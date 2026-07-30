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
from scipy.stats import rankdata
from tqdm import tqdm

from dataset import DAYS_PER_YEAR, Dataset, HonicReader
from eval import AgeStratRatesCollator, DiseaseRatesCollator, eval_iter, nearest_prediction
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


def _stratum_auc(is_case, dis_rates, dis_times, ctl_bin, g, lo, hi):
    """Windowed AUC for one (region, sex, age-bin) stratum, COLUMN BY COLUMN with row
    removal: for each disease we keep only the rows that are a valid control or case in
    this group and rank JUST those -- no full-N argsort over out-of-group / NaN rows.

    g: (N,) in-group bool (this region+sex). controls = in-group, not a case of the
    disease, with a bin control rate; cases = in-group onsets in [lo, hi).
    Returns (n_ctl, n_case, auc), each (V,). AUC = P(case score > control score), ties 0.5
    -- same statistic as batched_mann_whitney_auc.
    """
    V = dis_rates.shape[1]
    n_ctl = np.zeros(V, dtype=np.int64)
    n_case = np.zeros(V, dtype=np.int64)
    auc = np.full(V, np.nan, dtype=float)
    gi = np.flatnonzero(g)  # in-group row indices (drops out-of-group rows once)
    if gi.size == 0:
        return n_ctl, n_case, auc
    for d in range(V):
        cd = ctl_bin[gi, d]  # (n_g,) control rate for disease d in this bin
        ic = is_case[gi, d]  # (n_g,) does this in-group participant develop d at all
        cv = (~ic) & ~np.isnan(cd)  # controls: not a case, has a bin control rate
        dtd = dis_times[gi, d]
        kv = ic & (dtd >= lo) & (dtd < hi)  # cases: onset in [lo, hi) (matches searchsorted right)
        n1 = int(cv.sum())
        n2 = int(kv.sum())
        n_ctl[d] = n1
        n_case[d] = n2
        if n1 == 0 or n2 == 0:
            continue  # AUC undefined without both controls and cases
        valid = cv | kv
        sc = np.where(kv, dis_rates[gi, d], cd)[valid]  # scores of the k=n1+n2 real rows only
        ranks = rankdata(sc)  # average ties; k values, not N
        R1 = ranks[cv[valid]].sum()  # summed control ranks
        auc[d] = (n1 * n2 + 0.5 * n1 * (n1 + 1) - R1) / (n1 * n2)
    return n_ctl, n_case, auc


def main():
    args = parse_args()
    data_dir = args.data_dir
    df_event = args.df_event or f"{data_dir}/df_event.parquet"
    df_meta = args.df_meta or f"{data_dir}/df_meta.parquet"
    labels = args.labels or f"{data_dir}/delphi_labels_index_name.csv"
    device = args.device
    pprint.pp(vars(args))  # echo run config (no longer saved in the CSV)

    reader = HonicReader(df_event, df_meta, labels)
    age_group_edges = np.arange(args.age_start, args.age_end + args.age_gap, args.age_gap) * DAYS_PER_YEAR
    n_bins = len(age_group_edges) - 1

    model, model_args = load_model(args.ckpt, device, random_init=args.random_init, seed=args.seed)
    vocab_size = model_args["vocab_size"]
    ignore_tokens = set(model_args.get("ignore_tokens", [])) | {NO_EVENT_TOKEN}
    # targets = everything the model scores as a disease: all ids minus ignored
    # covariates/sex/padding and no_event. Matches delphi's model.targets - augmentation; incl. death.
    targets = torch.tensor(sorted(v for v in range(vocab_size) if v not in ignore_tokens), device=device)
    target_ids = targets.cpu().numpy()

    # crop length: checkpoint's block_size unless overridden; 0/negative disables
    block_size = model_args.get("block_size") if args.block_size is None else args.block_size
    block_size = block_size if block_size and block_size > 0 else None

    pids = reader.participants(args.fold)
    if args.subsample is not None and args.subsample < len(pids):
        sub = np.random.default_rng(args.seed).choice(len(pids), size=args.subsample, replace=False)
        pids = pids[np.sort(sub)]
    ds = Dataset(reader, pids, block_size=block_size, seed=args.seed)
    # score short sequences together to cut padding; rebind pids to the new row order
    pids = ds.sort_by_length(descending=True)

    offset_days = args.offset * DAYS_PER_YEAR
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
            # scores at the input step strictly before each target's (t1 - offset).
            # termination_token: reads past death -> NaN (death is last under the default
            # no_event config, so this only bites a negative --offset).
            scores, nearest_t0 = nearest_prediction(x0, t0, logits, t1 - offset_days, termination_token=reader.death_token)
            scores = scores.half()
            ctl_collator.step(timesteps=nearest_t0, logits=scores)
            dis_collator.step(tokens=x1, timesteps=nearest_t0, logits=scores)

    ctl_rates, _ = ctl_collator.finalize()  # (N, n_bins, V)
    dis_rates, dis_times = dis_collator.finalize()  # (N, V), (N, V)
    ctl_rates = ctl_rates.numpy()
    dis_rates = dis_rates.numpy()
    dis_times = dis_times.numpy()
    del ctl_collator, dis_collator

    is_female = reader.is_female(pids)  # (N,) bool, aligned to the array row order

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

    # is_case[:, d] marks participants who develop disease d at all; _stratum_auc bins
    # cases inline (dis_times in [lo, hi)) and ranks only the valid rows per column.
    is_case = ~np.isnan(dis_rates)  # (N, V)

    results = {}
    pbar = tqdm(total=len(region_groups) * 2 * n_bins, desc="AUC (region x sex x age bin)", leave=False)
    for region, is_r in region_groups:
        for sex_label, is_g in [("female", is_female), ("male", ~is_female)]:
            g = is_g & is_r  # (N,) in-group mask for this region+sex
            for i in range(n_bins):
                lo, hi = age_group_edges[i], age_group_edges[i + 1]
                results[(region, sex_label, i)] = _stratum_auc(
                    is_case, dis_rates, dis_times, ctl_rates[:, i, :], g, lo, hi
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
    for d in target_ids.tolist():
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
    print(f"Saved to {out_path}  ({len(rows)} rows = {len(target_ids)} tokens x {strata}{len(age_group_keys)} age bins x 2 sexes)")

    death_name = reader.detokenizer.get(reader.death_token, "death")
    print(f"\n=== {death_name} ===")
    pprint.pp([r for r in rows if r["token"] == death_name])


if __name__ == "__main__":
    main()
