"""Forecast evaluation: Harrell's C-index from a prompt anchor, hazards predictor.

Adapted from delphi's apps/forecast-m4.py, restricted to the "hazards" predictor:
one forward pass on each participant's prompt (history up to a cutoff `at` years
after baseline), then the model's prediction at the END of the prompt is used as
the per-disease risk score into the future. Valid when the intensity is roughly
time-homogeneous between events; with HONIC's ~5-year follow-up that's defensible.

Metric: Harrell's C-index per (token, [region,] sex) -- horizon-free, ranking
time-to-event from the anchor over the full follow-up (death censors, cause-specific).

Output: CSV with columns token, [region,] sex, cindex, n_event. --by-region adds
the region layer and appends '_by_region' to --out.
"""

import argparse
import csv
import os
import pprint

import numpy as np
import torch
from tqdm import tqdm

from auc import load_model  # shared legacy/random-init model loader
from dataset import Dataset, HonicReader
from eval import eval_iter, harrell_cindex

NO_EVENT_TOKEN = 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="path to the pretrained checkpoint (.pt); not needed with --random-init")
    p.add_argument("--random-init", action="store_true", help="evaluate a randomly-initialized model (chance baseline)")
    p.add_argument("--data-dir", default="data", help="dir holding df_event.parquet + df_meta.parquet")
    p.add_argument("--df-event", default=None)
    p.add_argument("--df-meta", default=None)
    p.add_argument("--labels", default=None, help="label index CSV (default <data-dir>/delphi_labels_index_name.csv)")
    p.add_argument("--fold", default=None, help="fold to evaluate (default: whole cohort)")
    p.add_argument("--subsample", type=int, default=None, help="randomly evaluate only this many participants")
    p.add_argument("--by-region", action="store_true", help="add a region stratification layer; appends '_by_region' to --out")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--at", type=float, default=0.0, help="prompt cutoff, YEARS after each participant's baseline age (age_bl)")
    p.add_argument("--block-size", type=int, default=None, help="crop each prompt to this many tokens (default: checkpoint's; 0 = no crop)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not args.random_init and not args.ckpt:
        p.error("--ckpt is required unless --random-init is set")
    return args


def main():
    args = parse_args()
    data_dir = args.data_dir
    df_event = args.df_event or f"{data_dir}/df_event.parquet"
    df_meta = args.df_meta or f"{data_dir}/df_meta.parquet"
    labels = args.labels or f"{data_dir}/delphi_labels_index_name.csv"
    device = args.device
    pprint.pp(vars(args))  # echo run config (not saved in the CSV)

    model, model_args = load_model(args.ckpt, device, random_init=args.random_init, seed=args.seed)
    vocab_size = model_args["vocab_size"]
    ignore_tokens = set(model_args.get("ignore_tokens", [])) | {NO_EVENT_TOKEN}
    disease_ids = np.array(sorted(v for v in range(vocab_size) if v not in ignore_tokens))  # {13..1269}

    block_size = model_args.get("block_size") if args.block_size is None else args.block_size
    block_size = block_size if block_size and block_size > 0 else None

    reader = HonicReader(df_event, df_meta, labels)
    # prompt cutoff `at` years after each baseline, keeping only pids with follow-up past it
    prompt_age = reader.resolve_prompt_age(args.at)
    fold_pids = set(reader.participants(args.fold).tolist())
    prompt_age = {p: c for p, c in prompt_age.items() if p in fold_pids}
    pids = np.array(list(prompt_age))
    if args.subsample is not None and args.subsample < len(pids):
        pids = pids[np.sort(np.random.default_rng(args.seed).choice(len(pids), size=args.subsample, replace=False))]

    # forecast mode: x0/t0 = prompt (events up to the cutoff). append_no_event=False
    # for hazards (the last real prompt position is the risk-read anchor).
    ds = Dataset(reader, pids, prompt_age=prompt_age, block_size=block_size, seed=args.seed)
    pids = ds.sort_by_length(descending=True)  # rebind to the batched row order

    # hazards predictor: the model's logits at the last prompt position (left-padded,
    # so index -1 is the most recent real event). One (N, V) score, used for all diseases.
    predictor = []
    with torch.no_grad():
        for batch_idx in tqdm(
            eval_iter(len(ds), args.batch_size), total=int(np.ceil(len(ds) / args.batch_size)), leave=False
        ):
            x0, t0, _, _ = (b.to(device) for b in ds.get_batch(batch_idx))
            logits, _, _ = model(x0, t0)  # legacy interface: (logits, loss, att)
            predictor.append(logits[:, -1, :].float().cpu().numpy())
    predictor = np.concatenate(predictor, axis=0)  # (N, V)

    # ground truth + anchor, aligned to the reordered rows
    event_times = reader.event_times(pids)  # (N, V) absolute first-occurrence age (days), NaN if none
    exit_time = reader.exit_times(pids)  # (N,) last-seen age (days) -> cause-specific censor
    is_female = reader.is_female(pids)  # (N,)
    anchor = np.array([prompt_age[p] for p in pids], dtype=np.float64)  # (N,) cutoff age (days)

    if args.by_region:
        regions = np.array(["unknown" if r is None else str(r) for r in reader.region(pids)])
        region_groups = [(r, regions == r) for r in sorted(set(regions.tolist()))]
    else:
        region_groups = [(None, np.ones(len(pids), dtype=bool))]

    fieldnames = ["token"] + (["region"] if args.by_region else []) + ["sex", "cindex", "n_event"]
    rows = []
    for region, is_r in region_groups:
        for sex_label, is_g in [("female", is_female), ("male", ~is_female)]:
            m = is_g & is_r
            est_g, occ_g, exit_g, anch_g = predictor[m], event_times[m], exit_time[m], anchor[m]
            for d in tqdm(disease_ids, desc=f"c-index {region or 'all'}/{sex_label}", leave=False):
                res = harrell_cindex(est_g[:, d], occ_g[:, d], exit_g, anch_g)
                if res["n_event"] == 0:
                    continue  # no cases in this stratum -> skip (undefined)
                row = {
                    "token": reader.detokenizer.get(int(d), str(d)), "sex": sex_label,
                    "cindex": None if np.isnan(res["cindex"]) else round(res["cindex"], 4),
                    "n_event": res["n_event"],
                }
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
    print(f"Saved to {out_path}  ({len(rows)} rows; {len(pids)} participants, anchor = age_bl + {args.at}y)")

    death_name = reader.detokenizer.get(reader.death_token, "death")
    print(f"\n=== {death_name} ===")
    pprint.pp([r for r in rows if r["token"] == death_name])


if __name__ == "__main__":
    main()
