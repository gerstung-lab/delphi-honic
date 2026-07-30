"""Dynamic (time-dependent) AUC / c-index for a pretrained Delphi model on honic.

Adapted from delphi apps/c-index-m4.py, restricted to the base setup (no expansion
packs / biomarkers / remap / adversarial swap) and using the legacy model interface.

Two passes over the frozen-history dataset:
  Phase 1 -- forward once; DiseaseRatesCollator collects each case's score (the model's
             rate at the input step just before the event).
  Phase 2 -- forward again; ConcordanceCollator scores every participant as a control
             at each case's ONSET age and counts, per case, the at-risk same-sex (and
             same-region, with --by-region) controls whose score is below the case's.

The c-index for a disease is sum(concordant)/sum(total_pairs) over its case events;
plot/dynamic_auc.py smooths concordant/total over case_time into the AUC(t) curve.

Output: CSV, one row per case event -- token, [region,] sex, participant_id, case_time,
concordant, total_pairs. --by-region adds region and appends '_by_region' to --out.

NOTE: the concordance is O(N x total_case_events); run on GPU, and use --subsample to
bound cost (it shrinks both factors).
"""

import argparse
import csv
import os
import pprint

import numpy as np
import torch
from tqdm import tqdm

from auc import load_model
from dataset import DAYS_PER_YEAR, Dataset, HonicReader
from eval import ConcordanceCollator, DiseaseRatesCollator, eval_iter, nearest_prediction

NO_EVENT_TOKEN = 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="pretrained checkpoint (.pt); not needed with --random-init")
    p.add_argument("--random-init", action="store_true", help="randomly-initialized model (chance baseline)")
    p.add_argument("--data-dir", default="data", help="dir holding df_event.parquet + df_meta.parquet")
    p.add_argument("--df-event", default=None)
    p.add_argument("--df-meta", default=None)
    p.add_argument("--labels", default=None, help="label index CSV (default <data-dir>/delphi_labels_index_name.csv)")
    p.add_argument("--fold", default=None, help="fold to evaluate (default: whole cohort)")
    p.add_argument("--subsample", type=int, default=None, help="randomly evaluate only this many participants")
    p.add_argument("--by-region", action="store_true", help="restrict controls to the case's region; appends '_by_region' to --out")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--offset", type=float, default=0.0, help="forecast offset in YEARS (score at onset - offset)")
    p.add_argument("--max-gap", type=float, default=5.0, help="control read must be within this many YEARS of the case onset")
    p.add_argument("--same-sex", action=argparse.BooleanOptionalAction, default=True, help="restrict controls to the case's sex")
    p.add_argument("--chunk-size", type=int, default=8192, help="case-event chunk size for the concordance inner loop")
    p.add_argument("--block-size", type=int, default=None, help="crop each sequence to this many tokens (default: checkpoint's; 0 = no crop)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not args.random_init and not args.ckpt:
        p.error("--ckpt is required unless --random-init is set")
    return args


def _forward_scores(model, x0, t0, x1, t1, offset_days):
    """(x1, nearest_t0, half scores) for one batch -- the frozen-history read before each target."""
    logits, _, _ = model(x0, t0)  # legacy interface
    scores, nearest_t0 = nearest_prediction(x0, t0, logits, t1 - offset_days)
    return scores.half(), nearest_t0, logits


def main():
    args = parse_args()
    data_dir = args.data_dir
    df_event = args.df_event or f"{data_dir}/df_event.parquet"
    df_meta = args.df_meta or f"{data_dir}/df_meta.parquet"
    labels = args.labels or f"{data_dir}/delphi_labels_index_name.csv"
    device = args.device
    pprint.pp(vars(args))

    model, model_args = load_model(args.ckpt, device, random_init=args.random_init, seed=args.seed)
    vocab_size = model_args["vocab_size"]
    ignore_tokens = set(model_args.get("ignore_tokens", [])) | {NO_EVENT_TOKEN}
    targets = torch.tensor(sorted(v for v in range(vocab_size) if v not in ignore_tokens), device=device)

    block_size = model_args.get("block_size") if args.block_size is None else args.block_size
    block_size = block_size if block_size and block_size > 0 else None

    reader = HonicReader(df_event, df_meta, labels)
    pids = reader.participants(args.fold)
    if args.subsample is not None and args.subsample < len(pids):
        pids = pids[np.sort(np.random.default_rng(args.seed).choice(len(pids), size=args.subsample, replace=False))]
    ds = Dataset(reader, pids, block_size=block_size, seed=args.seed)
    pids = ds.sort_by_length(descending=True)  # rebind to the batched row order

    offset_days = args.offset * DAYS_PER_YEAR

    # --- Phase 1: case scores (the model's rate just before each event) ---
    dis_collator = DiseaseRatesCollator(targets=targets, n_participants=len(ds))
    with torch.no_grad():
        for batch_idx in tqdm(eval_iter(len(ds), args.batch_size), total=int(np.ceil(len(ds) / args.batch_size)), desc="Phase 1", leave=False):
            x0, t0, x1, t1 = (b.to(device) for b in ds.get_batch(batch_idx))
            scores, nearest_t0, _ = _forward_scores(model, x0, t0, x1, t1, offset_days)
            dis_collator.step(tokens=x1, timesteps=nearest_t0, logits=scores)
    dis_rates, _ = dis_collator.finalize()  # (N, V) case scores, NaN where not a case
    del dis_collator

    # onset ages + strata, aligned to the reordered rows
    onset = reader.event_times(pids)  # (N, V) first-occurrence age (days), NaN if never
    is_female = torch.from_numpy(reader.is_female(pids)).to(device)
    regions = np.array(["unknown" if r is None else str(r) for r in reader.region(pids)]) if args.by_region else None
    region_codes = torch.from_numpy(np.unique(regions, return_inverse=True)[1]).to(device) if args.by_region else None

    dis_rates = dis_rates.to(device)
    case_times = (torch.from_numpy(onset).to(device) - offset_days)  # (N, V) onset - offset

    # --- Phase 2: concordance ---
    # covariates to control for: a control must match the case on each. Sex and region
    # are the same kind of constraint -- just entries in the list.
    covariates = []
    if args.same_sex:
        covariates.append(is_female.long())
    if args.by_region:
        covariates.append(region_codes)
    cc = ConcordanceCollator(
        dis_rates=dis_rates, case_times=case_times, covariates=covariates or None,
        chunk_size=args.chunk_size, max_gap_days=args.max_gap * DAYS_PER_YEAR,
    )
    del dis_rates, case_times  # the collator kept its own copies
    with torch.no_grad():
        for batch_idx in tqdm(eval_iter(len(ds), args.batch_size), total=int(np.ceil(len(ds) / args.batch_size)), desc="Phase 2", leave=False):
            x0, t0, _, _ = (b.to(device) for b in ds.get_batch(batch_idx))
            logits, _, _ = model(x0, t0)
            cc.step(x0, t0, logits)

    case_tokens, total_pairs, concordant = cc.finalize()
    case_time = cc.case_times.cpu().numpy()
    case_part = cc.case_participants.cpu().numpy()
    case_sex = is_female.cpu().numpy()  # per-participant sex, indexed by case_part below

    fieldnames = ["token"] + (["region"] if args.by_region else []) + ["sex", "participant_id", "case_time", "concordant", "total_pairs"]
    rows = []
    for k in range(len(case_tokens)):
        p = int(case_part[k])
        row = {
            "token": reader.detokenizer.get(int(case_tokens[k]), str(case_tokens[k])),
            "sex": "female" if case_sex[p] else "male",
            "participant_id": pids[p],
            "case_time": round(float(case_time[k]), 2),
            "concordant": float(concordant[k]),
            "total_pairs": int(total_pairs[k]),
        }
        if args.by_region:
            row["region"] = regions[p]
        rows.append(row)

    out_path = args.out
    if args.by_region:
        root, ext = os.path.splitext(out_path)
        out_path = f"{root}_by_region{ext}"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tot = sum(r["total_pairs"] for r in rows)
    con = sum(r["concordant"] for r in rows)
    print(f"Saved to {out_path}  ({len(rows)} case events; overall c-index = {con / tot:.4f})" if tot else f"Saved to {out_path}  ({len(rows)} case events)")


if __name__ == "__main__":
    main()
