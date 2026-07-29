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
import json
from dataclasses import fields

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
    p.add_argument("--ckpt", required=True, help="path to the pretrained checkpoint (.pt)")
    p.add_argument("--data-dir", default="data", help="dir holding df_event.parquet + df_meta.parquet")
    p.add_argument("--df-event", default=None, help="override df_event path (default <data-dir>/df_event.parquet)")
    p.add_argument("--df-meta", default=None, help="override df_meta path (default <data-dir>/df_meta.parquet)")
    p.add_argument("--labels", default=None, help="label index CSV (default <data-dir>/delphi_labels_index_name.csv)")
    p.add_argument("--fold", default=None, help="fold to evaluate (default: whole cohort)")
    p.add_argument("--out", required=True, help="output JSON path")
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
    return p.parse_args()


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    valid = {f.name for f in fields(DelphiConfig)}
    cfg = DelphiConfig(**{k: v for k, v in ck["model_args"].items() if k in valid})
    model = Delphi(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck["model_args"]


def main():
    args = parse_args()
    data_dir = args.data_dir
    df_event = args.df_event or f"{data_dir}/df_event.parquet"
    df_meta = args.df_meta or f"{data_dir}/df_meta.parquet"
    labels = args.labels or f"{data_dir}/delphi_labels_index_name.csv"
    device = args.device

    model, model_args = load_model(args.ckpt, device)
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
    ds = Dataset(reader, reader.participants(args.fold), block_size=block_size, seed=args.seed)
    # score short sequences together to cut padding; rebind pids to the new row order
    pids = ds.sort_by_length(descending=True)

    offset_days = args.offset * DAYS_PER_YEAR
    age_group_edges = np.arange(args.age_start, args.age_end + args.age_gap, args.age_gap) * DAYS_PER_YEAR
    n_bins = len(age_group_edges) - 1

    gen = torch.Generator(device=device).manual_seed(args.seed)
    ctl_collator = AgeStratRatesCollator(
        age_groups=torch.from_numpy(age_group_edges).float().to(device), generator=gen
    )
    dis_collator = DiseaseRatesCollator(targets=targets)

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
    is_female = reader.is_female(pids)  # (N,) bool, aligned to the reordered rows

    # Bin each case by the age of its prediction position (== nearest_t0), matching
    # the control binning. For a fixed (sex, age bin), one column-wise AUC pass:
    # controls = same-sex who never develop the token (their bin control rate),
    # cases = same-sex whose onset falls in this bin (their pre-onset rate).
    dis_time_bin = np.searchsorted(age_group_edges, dis_times, side="right") - 1  # (N, V)
    is_case = ~np.isnan(dis_rates)  # (N, V)

    results = {}
    for sex_label, is_g in [("female", is_female), ("male", ~is_female)]:
        for i in range(n_bins):
            ctl_score = ctl_rates[:, i, :]
            ctl_valid = (~is_case) & ~np.isnan(ctl_score) & is_g[:, None]
            case_valid = is_case & (dis_time_bin == i) & is_g[:, None]
            scores = np.where(case_valid, dis_rates, np.where(ctl_valid, ctl_score, np.nan))
            results[(sex_label, i)] = batched_mann_whitney_auc(scores, ctl=ctl_valid, case=case_valid)

    age_group_keys = [
        f"{int(start / DAYS_PER_YEAR)}-{int(end / DAYS_PER_YEAR)}"
        for start, end in zip(age_group_edges[:-1], age_group_edges[1:])
    ]
    logbook = {}
    for d in targets.cpu().numpy().tolist():
        icd = reader.detokenizer.get(int(d), str(d))
        logbook[icd] = {"female": {}, "male": {}}
        for sex_label in ("female", "male"):
            for i, age_grp in enumerate(age_group_keys):
                ctl_counts, case_counts, aucs = results[(sex_label, i)]
                auc = aucs[d]
                logbook[icd][sex_label][age_grp] = {
                    "auc": round(float(auc), 4) if not np.isnan(auc) else None,
                    "ctl_count": int(ctl_counts[d]),
                    "dis_count": int(case_counts[d]),
                }

    out_path = args.out
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "logbook": logbook}, f, indent=4)
    print(f"Saved to {out_path}  ({len(logbook)} tokens x {len(age_group_keys)} age bins x 2 sexes)")


if __name__ == "__main__":
    main()
