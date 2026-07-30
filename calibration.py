"""Instant (frozen-history) reliability curves per sex and 5-year age bin (+ optional region).

The calibration sibling of auc.py: same single-pass frozen-history scoring, but
instead of a per-(sex, age-bin) AUC it emits reliability (calibration) curve data.
Adapted from delphi's apps/instant_calibration.py.

  - CASE score: the model's predicted rate at the input position immediately before
    the disease event (DiseaseRatesCollator; --offset rewinds it).
  - CONTROL score: one randomly-sampled position per age bin per participant
    (AgeStratRatesCollator).
  - Rate -> probability over a window W = age_gap years: p = 1 - exp(-rate * W).
  - Within each (token, [region,] sex, age bin), predicted probabilities are binned
    on a fixed power-law grid (PROB_BINS); per bin we report the mean predicted
    probability, the observed case fraction, and the count.

LEGACY-MODEL RATE (the key adaptation): the delphi app reads a per-year rate from
model.intensity(). The legacy model has no intensity() -- it returns raw logits.
Its generate() samples waiting times as t = -log(U)*exp(-logit) with t in DAYS, so
the per-token rate is lambda = exp(logit) per day; we convert to per-year with
*365.25 before the p = 1 - exp(-rate * W) window formula. VERIFY this matches the
legacy plot_calibration if exact reproduction matters.

Output: CSV, one row per (token, [region,] sex, age_bin, prob_bin) with columns
pred, obs, count (empty bins dropped). --by-region appends '_by_region' to --out.
"""

import argparse
import csv
import os
import pprint

import numpy as np
import torch
from tqdm import tqdm

from auc import load_model  # shared legacy/random-init model loader
from dataset import DAYS_PER_YEAR, Dataset, HonicReader
from eval import AgeStratRatesCollator, DiseaseRatesCollator, eval_iter, nearest_prediction

NO_EVENT_TOKEN = 1
# Power-law predicted-probability bins (identical to the legacy plot_calibration):
# 14 right-closed bins over (1e-6, ~31].
PROB_BINS = 10.0 ** np.arange(-6.0, 1.5, 0.5)


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
    p.add_argument("--offset", type=float, default=0.0, help="forecast offset in YEARS (score at t1 - offset)")
    p.add_argument("--age-start", type=int, default=40)
    p.add_argument("--age-end", type=int, default=85)
    p.add_argument("--age-gap", type=int, default=5, help="age-bin width in years; also the probability window W")
    p.add_argument("--block-size", type=int, default=None, help="crop each sequence to this many tokens (default: checkpoint's; 0 = no crop)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not args.random_init and not args.ckpt:
        p.error("--ckpt is required unless --random-init is set")
    return args


def reliability(p_ctl, p_case):
    """Reliability points over PROB_BINS for one (token, [region,] sex, age bin).

    Returns a list of (prob_bin_hi, pred, obs, count) for POPULATED bins only:
    pred = mean predicted prob in bin, obs = observed case fraction, count = n.
    Bins are right-closed (PROB_BINS[b-1], PROB_BINS[b]] (np.digitize right=True).
    """
    p = np.concatenate([p_ctl, p_case])
    y = np.concatenate([np.zeros(len(p_ctl)), np.ones(len(p_case))])
    idx = np.digitize(p, PROB_BINS, right=True)
    out = []
    for b in range(1, len(PROB_BINS)):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue  # drop empty bins (a reliability curve only has points where there's data)
        out.append((float(PROB_BINS[b]), round(float(p[m].mean()), 6), round(float(y[m].mean()), 6), c))
    return out


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
    targets = torch.tensor(sorted(v for v in range(vocab_size) if v not in ignore_tokens), device=device)

    block_size = model_args.get("block_size") if args.block_size is None else args.block_size
    block_size = block_size if block_size and block_size > 0 else None

    reader = HonicReader(df_event, df_meta, labels)
    pids = reader.participants(args.fold)
    if args.subsample is not None and args.subsample < len(pids):
        sub = np.random.default_rng(args.seed).choice(len(pids), size=args.subsample, replace=False)
        pids = pids[np.sort(sub)]
    ds = Dataset(reader, pids, block_size=block_size, seed=args.seed)
    pids = ds.sort_by_length(descending=True)  # rebind to the new row order

    offset_days = args.offset * DAYS_PER_YEAR
    window = float(args.age_gap)  # probability window in years (== age-bin width)
    age_group_edges = np.arange(args.age_start, args.age_end + args.age_gap, args.age_gap) * DAYS_PER_YEAR
    n_bins = len(age_group_edges) - 1

    gen = torch.Generator(device=device).manual_seed(args.seed)
    ctl_collator = AgeStratRatesCollator(
        age_groups=torch.from_numpy(age_group_edges).float().to(device), n_participants=len(ds), generator=gen
    )
    dis_collator = DiseaseRatesCollator(targets=targets, n_participants=len(ds))

    with torch.no_grad():
        for batch_idx in tqdm(
            eval_iter(len(ds), args.batch_size), total=int(np.ceil(len(ds) / args.batch_size)), leave=False
        ):
            x0, t0, x1, t1 = (b.to(device) for b in ds.get_batch(batch_idx))
            logits, _, _ = model(x0, t0)  # legacy interface
            scores, nearest_t0 = nearest_prediction(x0, t0, logits, t1 - offset_days)
            # legacy logit -> per-day rate exp(logit) -> per-year rate; occurred -> 0, invalid -> NaN.
            # half-overflow on huge rates saturates to +inf -> prob 1 downstream (the correct limit).
            rate = (scores.float().exp() * DAYS_PER_YEAR).half()
            ctl_collator.step(timesteps=nearest_t0, logits=rate)
            dis_collator.step(tokens=x1, timesteps=nearest_t0, logits=rate)

    ctl_rates, _ = ctl_collator.finalize()  # (N, n_bins, V) per-year rates
    dis_rates, dis_times = dis_collator.finalize()  # (N, V), (N, V)
    # keep the big arrays float16 (as auc.py does) -- upcasting to float32 here would
    # double a ~46GB (N, n_bins, V) array. The probability math below upcasts one
    # (N, V) slice at a time instead, which is identical numerically but bounded.
    ctl_rates = ctl_rates.numpy()  # float16
    dis_rates = dis_rates.numpy()  # float16
    dis_times = dis_times.numpy()
    del ctl_collator, dis_collator
    is_female = reader.is_female(pids)

    # region as an optional stratification dimension (see auc.py): a (label, mask)
    # grouping list that collapses to one all-rows group when off.
    if args.by_region:
        regions = np.array(["unknown" if r is None else str(r) for r in reader.region(pids)])
        region_groups = [(r, regions == r) for r in sorted(set(regions.tolist()))]
    else:
        region_groups = [(None, np.ones(len(pids), dtype=bool))]

    with np.errstate(over="ignore", invalid="ignore"):
        prob_dis = 1.0 - np.exp(-dis_rates.astype(np.float32) * window)  # (N, V) case probabilities
    is_case = ~np.isnan(dis_rates)  # (N, V)
    age_group_keys = [
        f"{int(start / DAYS_PER_YEAR)}-{int(end / DAYS_PER_YEAR)}"
        for start, end in zip(age_group_edges[:-1], age_group_edges[1:])
    ]

    fieldnames = (
        ["token"] + (["region"] if args.by_region else []) + ["sex", "age_bin", "prob_bin_hi", "pred", "obs", "count"]
    )
    target_ids = targets.detach().cpu().numpy().tolist()
    rows = []
    for i, bracket in enumerate(tqdm(age_group_keys, desc="age bins")):
        lo, hi = age_group_edges[i], age_group_edges[i + 1]  # bin i == [lo, hi) (matches searchsorted right)
        ctl_i = ctl_rates[:, i, :].astype(np.float32)  # (N, V) upcast one bin at a time
        ctl_here = (~is_case) & ~np.isnan(ctl_i)  # (N, V)
        case_here = is_case & (dis_times >= lo) & (dis_times < hi)  # (N, V)
        with np.errstate(over="ignore", invalid="ignore"):
            prob_ctl_i = 1.0 - np.exp(-ctl_i * window)  # (N, V)
        for region, is_r in region_groups:
            for sex_label, is_g in [("female", is_female), ("male", ~is_female)]:
                grp = is_g & is_r
                for d in target_ids:
                    token = reader.detokenizer.get(int(d), str(d))
                    cmask = ctl_here[:, d] & grp
                    kmask = case_here[:, d] & grp
                    for prob_hi, pred, obs, count in reliability(prob_ctl_i[cmask, d], prob_dis[kmask, d]):
                        row = {
                            "token": token, "sex": sex_label, "age_bin": bracket,
                            "prob_bin_hi": prob_hi, "pred": pred, "obs": obs, "count": count,
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
    print(f"Saved to {out_path}  ({len(rows)} non-empty reliability rows)")

    death_name = reader.detokenizer.get(reader.death_token, "death")
    print(f"\n=== {death_name} ({age_group_keys[0]}) ===")
    pprint.pp([r for r in rows if r["token"] == death_name and r["age_bin"] == age_group_keys[0]])


if __name__ == "__main__":
    main()
