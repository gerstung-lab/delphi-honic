"""Age/sex-stratified Mann-Whitney AUC for a pretrained Delphi model on the honic cohort.

One forward pass per participant reads the model's logits at the input position
strictly before each target event (frozen history), then for every disease token
asks: within an age bin and sex, can the model rank the people who develop the
disease above those who will not.

All models are assumed to share the legacy interface -- model(idx, age) returning
(logits, loss, att) -- so scoring goes through eval.nearest_prediction (the legacy
model has no intensity() method). The newer models are wrapped to match.

Output: JSON logbook[icd][sex][age_bin] = {"auc", "ctl_count", "dis_count"}.

Run:
    mamba run -n <torch-env> python auc.py \
        --ckpt /path/ckpt.pt --data-dir data --out results/auc.json
"""

import argparse
import csv
import os
import pprint
from dataclasses import asdict, fields

import numpy as np
import torch
from tqdm import tqdm

from dataset import DAYS_PER_YEAR, Dataset, HonicReader
from eval import (
    AgeStratRatesCollator,
    DiseaseRatesCollator,
    batched_mann_whitney_auc,
    eval_iter,
    nearest_prediction,
)
from legacy_model import Delphi, DelphiConfig

NO_EVENT_TOKEN = 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="path to the pretrained checkpoint (.pt); not needed with --random-init")
    p.add_argument(
        "--random-init",
        action="store_true",
        help="skip the checkpoint and evaluate a randomly-initialized model (chance baseline); "
        "uses default DelphiConfig args, which match the pretrained architecture",
    )
    p.add_argument("--data-dir", default="data", help="dir holding df_event.parquet + df_meta.parquet")
    p.add_argument("--df-event", default=None, help="override df_event path (default <data-dir>/df_event.parquet)")
    p.add_argument("--df-meta", default=None, help="override df_meta path (default <data-dir>/df_meta.parquet)")
    p.add_argument("--labels", default=None, help="label index CSV (default <data-dir>/delphi_labels_index_name.csv)")
    p.add_argument("--fold", default=None, help="fold to evaluate (default: whole cohort)")
    p.add_argument("--subsample", type=int, default=None, help="randomly evaluate only this many participants (default: all)")
    p.add_argument("--by-region", action="store_true", help="add a region stratification layer (df_meta.region_bl); appends '_by_region' to --out")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--offset", type=float, default=0.0, help="forecast offset in YEARS (score at t1 - offset)")
    p.add_argument("--age-start", type=int, default=40)
    p.add_argument("--age-end", type=int, default=85)
    p.add_argument("--age-gap", type=int, default=5)
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="crop each sequence to this many tokens (default: checkpoint's; 0 = no crop)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="column-block size for the per-stratum AUC (masks+scores+ranking); caps the "
        "AUC-stage transient to ~(N, chunk). Default 128; 0 = no chunking (whole V at once)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not args.random_init and not args.ckpt:
        p.error("--ckpt is required unless --random-init is set")
    return args


def load_model(ckpt_path, device, random_init=False, seed=None):
    """Build a Delphi model. random_init: a fresh model with default DelphiConfig
    args (which match the pretrained architecture) and no weights loaded -- a
    chance baseline; seed makes the random weights reproducible."""
    if random_init:
        if seed is not None:
            torch.manual_seed(seed)
        cfg = DelphiConfig()
        model = Delphi(cfg)
        model_args = asdict(cfg)
    else:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        valid = {f.name for f in fields(DelphiConfig)}
        cfg = DelphiConfig(**{k: v for k, v in ck["model_args"].items() if k in valid})
        model = Delphi(cfg)
        model.load_state_dict(ck["model"])
        model_args = ck["model_args"]
    model.to(device).eval()
    return model, model_args


def _cell(res, d):
    """One logbook cell for disease token `d` from a batched_mann_whitney_auc result."""
    ctl_counts, case_counts, aucs = res
    auc = aucs[d]
    return {
        "auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "ctl_count": int(ctl_counts[d]),
        "dis_count": int(case_counts[d]),
    }


def _stratum_auc(is_case, dis_rates, dis_times, ctl_bin, is_gr, lo, hi, chunk):
    """Windowed AUC for one (region, sex, age-bin) stratum, built in COLUMN BLOCKS so
    no full (N, V) mask/score/rank array is ever materialized -- caps the transient to
    (N, chunk). controls = in-group, event-free, with a bin control rate; cases =
    in-group onsets in [lo, hi). Returns (n_ctl, n_case, auc), each (V,). chunk=None -> one block.
    """
    V = dis_rates.shape[1]
    step = V if chunk is None else chunk
    n_ctl = np.zeros(V, dtype=np.int64)
    n_case = np.zeros(V, dtype=np.int64)
    auc = np.full(V, np.nan, dtype=float)
    for s in range(0, V, step):
        e = min(s + step, V)
        ic, cs, dt = is_case[:, s:e], ctl_bin[:, s:e], dis_times[:, s:e]
        ctl_valid = (~ic) & ~np.isnan(cs) & is_gr
        # onset in [lo, hi): edges[i] <= age < edges[i+1] (matches searchsorted side="right")
        case_valid = ic & (dt >= lo) & (dt < hi) & is_gr
        scores = np.where(case_valid, dis_rates[:, s:e], np.where(ctl_valid, cs, np.nan))
        n_ctl[s:e], n_case[s:e], auc[s:e] = batched_mann_whitney_auc(scores, ctl_valid, case_valid)
    return n_ctl, n_case, auc


def main():
    args = parse_args()
    data_dir = args.data_dir
    df_event = args.df_event or f"{data_dir}/df_event.parquet"
    df_meta = args.df_meta or f"{data_dir}/df_meta.parquet"
    labels = args.labels or f"{data_dir}/delphi_labels_index_name.csv"
    device = args.device
    pprint.pp(vars(args))  # echo run config (no longer saved in the CSV)

    model, model_args = load_model(args.ckpt, device, random_init=args.random_init, seed=args.seed)
    vocab_size = model_args["vocab_size"]
    ignore_tokens = set(model_args.get("ignore_tokens", [])) | {NO_EVENT_TOKEN}
    # targets = everything the model scores as a disease: all ids minus ignored
    # covariates/sex/padding and the no_event augmentation. Matches delphi's
    # (model.targets - augmentation_tokens); includes the death token.
    targets = torch.tensor(
        sorted(v for v in range(vocab_size) if v not in ignore_tokens), device=device
    )

    # crop length: checkpoint's block_size unless overridden; 0/negative disables
    block_size = model_args.get("block_size") if args.block_size is None else args.block_size
    block_size = block_size if block_size and block_size > 0 else None

    reader = HonicReader(df_event, df_meta, labels)
    pids = reader.participants(args.fold)
    if args.subsample is not None and args.subsample < len(pids):
        sub = np.random.default_rng(args.seed).choice(len(pids), size=args.subsample, replace=False)
        pids = pids[np.sort(sub)]
    ds = Dataset(reader, pids, block_size=block_size, seed=args.seed)
    # score short sequences together to cut padding; rebind pids to the new row order
    pids = ds.sort_by_length(descending=True)

    offset_days = args.offset * DAYS_PER_YEAR
    age_group_edges = np.arange(args.age_start, args.age_end + args.age_gap, args.age_gap) * DAYS_PER_YEAR
    n_bins = len(age_group_edges) - 1

    gen = torch.Generator(device=device).manual_seed(args.seed)
    ctl_collator = AgeStratRatesCollator(
        age_groups=torch.from_numpy(age_group_edges).float().to(device), n_participants=len(ds), generator=gen
    )
    dis_collator = DiseaseRatesCollator(targets=targets, n_participants=len(ds))

    with torch.no_grad():
        for batch_idx in tqdm(
            eval_iter(len(ds), args.batch_size),
            total=int(np.ceil(len(ds) / args.batch_size)),
            leave=False,
        ):
            x0, t0, x1, t1 = (b.to(device) for b in ds.get_batch(batch_idx))
            logits, _, _ = model(x0, t0)  # legacy interface: (logits, loss, att)
            # scores at the input step strictly before each target's (t1 - offset)
            scores, nearest_t0 = nearest_prediction(x0, t0, logits, t1 - offset_days)
            scores = scores.half()
            ctl_collator.step(timesteps=nearest_t0, logits=scores)
            dis_collator.step(tokens=x1, timesteps=nearest_t0, logits=scores)

    ctl_rates, _ = ctl_collator.finalize()  # (N, n_bins, V)
    dis_rates, dis_times = dis_collator.finalize()  # (N, V), (N, V)
    ctl_rates = ctl_rates.numpy()
    dis_rates = dis_rates.numpy()
    dis_times = dis_times.numpy()
    del ctl_collator, dis_collator  # drop the per-batch lists finalize() kept alive (else ~2x memory)
    is_female = reader.is_female(pids)  # (N,) bool, aligned to the reordered rows

    # Region is just one more stratification dimension: a list of (label, row-mask)
    # groups. Turning it OFF doesn't drop the loop -- it collapses the dimension to
    # a single group covering everyone, so the compute + logbook loops below are
    # identical either way. (AUC can't be marginalized after the fact, so "off"
    # genuinely recomputes over the merged cohort rather than pooling regions.)
    if args.by_region:
        regions = np.array(["unknown" if r is None else str(r) for r in reader.region(pids)])
        region_groups = [(r, regions == r) for r in sorted(set(regions.tolist()))]  # missing -> "unknown"
    else:
        region_groups = [(None, np.ones(len(pids), dtype=bool))]

    # Bin each case by the age of its prediction position (== nearest_t0), matching
    # the control binning. For a fixed (region, sex, age bin), one column-wise AUC
    # pass: controls = matching participants who never develop the token (their bin
    # control rate); cases = matching participants whose onset falls in this bin.
    # Bin membership is compared inline against the two bin edges in the loop below
    # (a transient (N, V) bool, freed each iteration) rather than materializing a full
    # (N, V) dis_time_bin: np.searchsorted would build an int64 (N, V) temp (~21 GB at
    # 2M x 1300, ~40 GB with the `- 1`) before we could downcast it.
    is_case = ~np.isnan(dis_rates)  # (N, V)
    chunk = args.chunk_size if args.chunk_size > 0 else None  # column-block size (0/neg -> no chunking)

    results = {}
    pbar = tqdm(total=len(region_groups) * 2 * n_bins, desc="AUC (region x sex x age bin)", leave=False)
    for region, is_r in region_groups:
        for sex_label, is_g in [("female", is_female), ("male", ~is_female)]:
            is_gr = (is_g & is_r)[:, None]
            for i in range(n_bins):
                lo, hi = age_group_edges[i], age_group_edges[i + 1]
                results[(region, sex_label, i)] = _stratum_auc(
                    is_case, dis_rates, dis_times, ctl_rates[:, i, :], is_gr, lo, hi, chunk
                )
                pbar.update(1)
    pbar.close()

    age_group_keys = [
        f"{int(start / DAYS_PER_YEAR)}-{int(end / DAYS_PER_YEAR)}"
        for start, end in zip(age_group_edges[:-1], age_group_edges[1:])
    ]
    # Flatten to one row per (token, [region,] sex, age_bin): each nesting key is a
    # column. The region column exists only when --by-region -- off means that
    # stratification level (and column) is absent.
    fieldnames = ["token"] + (["region"] if args.by_region else []) + ["sex", "age_bin", "auc", "ctl_count", "dis_count"]
    rows = []
    for d in targets.cpu().numpy().tolist():
        icd = reader.detokenizer.get(int(d), str(d))
        for region, _ in region_groups:
            for sex_label in ("female", "male"):
                for i, age_grp in enumerate(age_group_keys):
                    row = {"token": icd, "sex": sex_label, "age_bin": age_grp, **_cell(results[(region, sex_label, i)], d)}
                    if args.by_region:
                        row["region"] = region
                    rows.append(row)

    out_path = args.out
    if args.by_region:
        root, ext = os.path.splitext(out_path)
        out_path = f"{root}_by_region{ext}"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    strata = f"{len(region_groups)} regions x " if args.by_region else ""
    print(f"Saved to {out_path}  ({len(rows)} rows = {len(targets)} tokens x {strata}{len(age_group_keys)} age bins x 2 sexes)")

    death_name = reader.detokenizer.get(reader.death_token, "death")
    print(f"\n=== {death_name} ===")
    pprint.pp([r for r in rows if r["token"] == death_name])


if __name__ == "__main__":
    main()
