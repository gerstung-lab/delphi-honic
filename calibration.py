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


def _scatter(acc_n, acc_case, acc_psum, base, rate, valid, window, is_case):
    """Bin one (chunk, age-bin) slice into flat (group, token, prob_bin) histograms.

    Only the `valid` cells are gathered before exp -- so we never build a full
    (C, V) probability array (dense for controls, sparse for cases; one code path).
    base: (C, V) int64 = (group*V + token)*N_PROB key without the prob-bin offset.
    """
    if not valid.any():
        return
    with np.errstate(over="ignore", invalid="ignore"):
        pv = 1.0 - np.exp(-rate[valid].astype(np.float32) * window)  # 1-D, valid cells
    keyv = base[valid] + np.digitize(pv, PROB_BINS, right=True)
    flat_n, flat_case, flat_psum = acc_n.reshape(-1), acc_case.reshape(-1), acc_psum.reshape(-1)
    m = flat_n.size
    flat_n += np.bincount(keyv, minlength=m)
    flat_psum += np.bincount(keyv, weights=pv, minlength=m)  # weights -> float64 sum
    if is_case:
        flat_case += np.bincount(keyv, minlength=m)


def accumulate_reliability(ctl_rates, dis_rates, dis_times, group_id, n_groups,
                           window, age_group_edges, chunk=None, progress=False):
    """Reliability histograms over PROB_BINS, one participant-chunk at a time.

    Replaces the per-(sex, token) column-gather loop: reads the big fp16 arrays in
    row-contiguous chunks (sequential, cache- and swap-friendly) and upcasts only the
    selected cells. Returns three (n_bins, n_groups, V, N_PROB) arrays -- count, case
    count, summed predicted prob -- giving pred = psum/count and obs = case/count.
    Semantics: control cell = not a case for that token AND a control was sampled in
    the bin; case cell = a case whose onset falls in the bin. Bins are right-closed
    (PROB_BINS[b-1], PROB_BINS[b]] via np.digitize(right=True); empty bins are dropped.
    """
    N, V = dis_rates.shape
    n_bins = len(age_group_edges) - 1
    n_prob = len(PROB_BINS) + 1  # np.digitize(., right=True) returns 0..len(PROB_BINS)
    shape = (n_bins, n_groups, V, n_prob)
    acc_n = np.zeros(shape, dtype=np.int64)
    acc_case = np.zeros(shape, dtype=np.int64)
    acc_psum = np.zeros(shape, dtype=np.float64)
    col_key = (np.arange(V, dtype=np.int64) * n_prob)[None, :]  # (1, V)
    if chunk is None:
        chunk = max(1, 20_000_000 // V)  # ~2e7 cells/pass -> sub-GB peak
    it = range(0, N, chunk)
    if progress:
        it = tqdm(it, total=int(np.ceil(N / chunk)), desc="chunks")
    for r0 in it:
        r1 = min(r0 + chunk, N)
        base = (group_id[r0:r1] * V * n_prob)[:, None] + col_key  # (C, V) key sans prob bin
        is_case_c = ~np.isnan(dis_rates[r0:r1])  # (C, V)
        dis_c = dis_rates[r0:r1]                 # (C, V) fp16
        dtimes_c = dis_times[r0:r1]              # (C, V)
        for i in range(n_bins):
            lo, hi = age_group_edges[i], age_group_edges[i + 1]
            ctl_c = ctl_rates[r0:r1, i, :]  # (C, V) fp16 (contiguous last axis)
            ctl_valid = (~is_case_c) & ~np.isnan(ctl_c)
            case_valid = is_case_c & (dtimes_c >= lo) & (dtimes_c < hi)
            _scatter(acc_n[i], acc_case[i], acc_psum[i], base, ctl_c, ctl_valid, window, is_case=False)
            _scatter(acc_n[i], acc_case[i], acc_psum[i], base, dis_c, case_valid, window, is_case=True)
    return acc_n, acc_case, acc_psum


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
            # termination_token: reads past death -> NaN (death is last under the default
            # no_event config, so this only bites a negative --offset).
            scores, nearest_t0 = nearest_prediction(x0, t0, logits, t1 - offset_days, termination_token=reader.death_token)
            # legacy logit -> per-day rate exp(logit) -> per-year rate; occurred -> 0,
            # invalid/post-death -> NaN. half-overflow on huge rates saturates to +inf ->
            # prob 1 downstream (the correct limit).
            rate = (scores.float().exp() * DAYS_PER_YEAR).half()
            ctl_collator.step(timesteps=nearest_t0, logits=rate)
            dis_collator.step(tokens=x1, timesteps=nearest_t0, logits=rate)

    ctl_rates, _ = ctl_collator.finalize()  # (N, n_bins, V) per-year rates
    dis_rates, dis_times = dis_collator.finalize()  # (N, V), (N, V)
    # keep the big arrays float16 (as auc.py does); accumulate_reliability walks them
    # in row-contiguous chunks and upcasts only the selected cells to float32.
    ctl_rates = ctl_rates.numpy()  # float16 (N, n_bins, V)
    dis_rates = dis_rates.numpy()  # float16 (N, V)
    dis_times = dis_times.numpy()  # (N, V)
    del ctl_collator, dis_collator
    is_female = reader.is_female(pids)

    # map every participant to one group id = region_idx * 2 + sex_id (0 female, 1
    # male). --by-region off -> a single region, so the id is just the sex.
    sex_id = np.where(is_female, 0, 1)
    if args.by_region:
        regions = np.array(["unknown" if r is None else str(r) for r in reader.region(pids)])
        region_labels = sorted(set(regions.tolist()))
        region_of = {r: k for k, r in enumerate(region_labels)}
        region_idx = np.array([region_of[r] for r in regions], dtype=np.int64)
    else:
        region_labels = [None]
        region_idx = np.zeros(len(pids), dtype=np.int64)
    group_id = region_idx * 2 + sex_id
    n_groups = 2 * len(region_labels)

    acc_n, acc_case, acc_psum = accumulate_reliability(
        ctl_rates, dis_rates, dis_times, group_id, n_groups, window, age_group_edges, progress=True
    )

    age_group_keys = [
        f"{int(start / DAYS_PER_YEAR)}-{int(end / DAYS_PER_YEAR)}"
        for start, end in zip(age_group_edges[:-1], age_group_edges[1:])
    ]
    fieldnames = (
        ["token"] + (["region"] if args.by_region else []) + ["sex", "age_bin", "prob_bin_hi", "pred", "obs", "count"]
    )
    target_ids = targets.detach().cpu().numpy().tolist()
    sexes = ["female", "male"]  # index == sex_id
    rows = []
    for i, bracket in enumerate(age_group_keys):
        for region_k, region in enumerate(region_labels):
            for sex_k, sex_label in enumerate(sexes):
                g = region_k * 2 + sex_k
                for d in target_ids:
                    cnt, csum, kcase = acc_n[i, g, d], acc_psum[i, g, d], acc_case[i, g, d]  # (N_PROB,) each
                    token = reader.detokenizer.get(int(d), str(d))
                    for b in range(1, len(PROB_BINS)):  # populated bins only (b in 1..len-1)
                        c = int(cnt[b])
                        if c == 0:
                            continue
                        row = {
                            "token": token, "sex": sex_label, "age_bin": bracket,
                            "prob_bin_hi": float(PROB_BINS[b]),
                            "pred": round(float(csum[b] / c), 6),
                            "obs": round(float(kcase[b] / c), 6),
                            "count": c,
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
