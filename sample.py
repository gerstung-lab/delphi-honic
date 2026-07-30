"""Sample future trajectories from the legacy Delphi model, matched to each
participant's ground-truth follow-up duration, and save them to a .npz.

Same prompt setup as forecast.py: x0/t0 = each participant's history up to a cutoff
`at` years after baseline. generate() (no KV cache) then continues each prompt until
the sampled age reaches that participant's ground-truth exit age (max_age = exit_time),
terminating early on death -- so each sampled trajectory spans the same age window as
the real one. stop_at_block_size is OFF (age governs the horizon, not token count);
--max-new-tokens is an optional runaway cap.

Output .npz, all arrays LEFT-padded (idx/mask pad 0, age pad -1e4) and row-aligned to
`pids`:
  pids, is_female, region, anchor (cutoff age, days), exit_age (days),
  gen_idx, gen_age, gen_mask (0 pad / 1 prompt / 2 continuation / 3 censored),
  n_prompt, n_gen, true_idx, true_age (the dataset ground-truth trajectory).

Note: sampling re-runs a full forward per step (no KV cache) and trajectories past the
model's trained block_size are extrapolation -- subsample for anything large.
"""

import argparse
import pprint

import numpy as np
import torch
from tqdm import tqdm

from auc import load_model  # shared legacy/random-init model loader
from dataset import Dataset, HonicReader
from eval import eval_iter
from generate import generate

NO_EVENT_TOKEN = 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="path to the pretrained checkpoint (.pt); not needed with --random-init")
    p.add_argument("--random-init", action="store_true", help="sample from a randomly-initialized model")
    p.add_argument("--data-dir", default="data", help="dir holding df_event.parquet + df_meta.parquet")
    p.add_argument("--df-event", default=None)
    p.add_argument("--df-meta", default=None)
    p.add_argument("--labels", default=None, help="label index CSV (default <data-dir>/delphi_labels_index_name.csv)")
    p.add_argument("--fold", default=None, help="fold to sample (default: whole cohort)")
    p.add_argument("--subsample", type=int, default=None, help="randomly sample only this many participants")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--at", type=float, default=0.0, help="prompt cutoff, YEARS after each participant's baseline age (age_bl)")
    p.add_argument("--max-new-tokens", type=int, default=None, help="safety cap on generated events per participant (default: none; age governs)")
    p.add_argument("--block-size", type=int, default=None, help="crop each prompt to this many tokens (default: checkpoint's; 0 = no crop)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not args.random_init and not args.ckpt:
        p.error("--ckpt is required unless --random-init is set")
    return args


def _stack_left(arrays, pad):
    """Left-pad a list of (b, l) arrays to the global max width, then vstack -> (N, L)."""
    L = max(a.shape[1] for a in arrays)
    out = [
        a if a.shape[1] == L
        else np.concatenate([np.full((a.shape[0], L - a.shape[1]), pad, dtype=a.dtype), a], axis=1)
        for a in arrays
    ]
    return np.concatenate(out, axis=0)


def main():
    args = parse_args()
    data_dir = args.data_dir
    df_event = args.df_event or f"{data_dir}/df_event.parquet"
    df_meta = args.df_meta or f"{data_dir}/df_meta.parquet"
    labels = args.labels or f"{data_dir}/delphi_labels_index_name.csv"
    device = args.device
    pprint.pp(vars(args))  # echo run config (not saved in the npz)

    model, model_args = load_model(args.ckpt, device, random_init=args.random_init, seed=args.seed)
    block_size = model_args.get("block_size") if args.block_size is None else args.block_size
    block_size = block_size if block_size and block_size > 0 else None

    reader = HonicReader(df_event, df_meta, labels)
    # prompt cutoff `at` years after baseline; resolve_prompt_age drops empty prompts
    prompt_age = reader.resolve_prompt_age(args.at)
    fold_pids = set(reader.participants(args.fold).tolist())
    prompt_age = {p: c for p, c in prompt_age.items() if p in fold_pids}
    pids = np.array(list(prompt_age))
    if args.subsample is not None and args.subsample < len(pids):
        pids = pids[np.sort(np.random.default_rng(args.seed).choice(len(pids), size=args.subsample, replace=False))]

    # forecast mode: x0/t0 = prompt, x1/t1 = ground-truth trajectory. crop_mode=left so a
    # block-size crop keeps the earliest tokens (never strips the pre-cutoff prompt history).
    # ponytail: left crop also truncates a >block_size ground truth to its first block_size
    # tokens; the horizon still uses the full exit age. Read raw events if you need full GT.
    ds = Dataset(reader, pids, prompt_age=prompt_age, block_size=block_size, crop_mode="left", seed=args.seed)
    pids = ds.sort_by_length(descending=True)  # rebind to the batched row order
    exit_age = reader.exit_times(pids).astype(np.float64)  # (N,) ground-truth end age (days) = generation horizon
    death = reader.death_token

    torch.manual_seed(args.seed)  # sampling RNG
    gi, ga, gm, npr, ng, ti, ta = [], [], [], [], [], [], []
    for batch_idx in tqdm(eval_iter(len(ds), args.batch_size), total=int(np.ceil(len(ds) / args.batch_size)), leave=False):
        x0, t0, x1, t1 = (b.to(device) for b in ds.get_batch(batch_idx))
        max_age = torch.as_tensor(exit_age[batch_idx], dtype=t0.dtype, device=device)  # per-participant horizon
        out_idx, out_age, misc = generate(
            model, x0, t0, termination_tokens=[death], max_age=max_age,
            max_new_tokens=args.max_new_tokens, stop_at_block_size=False,
        )
        gi.append(out_idx.cpu().numpy()); ga.append(out_age.cpu().numpy()); gm.append(misc["mask"].cpu().numpy())
        npr.append(misc["n_prompt"]); ng.append(misc["n_gen"])
        ti.append(x1.cpu().numpy()); ta.append(t1.cpu().numpy())

    regions = np.array(["unknown" if r is None else str(r) for r in reader.region(pids)])
    anchor = np.array([prompt_age[p] for p in pids], dtype=np.float64)
    np.savez_compressed(
        args.out,
        pids=pids.astype(str), is_female=reader.is_female(pids), region=regions,
        anchor=anchor, exit_age=exit_age,
        gen_idx=_stack_left(gi, 0), gen_age=_stack_left(ga, -1e4), gen_mask=_stack_left(gm, 0),
        n_prompt=np.concatenate(npr), n_gen=np.concatenate(ng),
        true_idx=_stack_left(ti, 0), true_age=_stack_left(ta, -1e4),
    )
    out_path = args.out if args.out.endswith(".npz") else args.out + ".npz"
    print(f"Saved to {out_path}  ({len(pids)} trajectories, prompt = age_bl + {args.at}y, horizon = ground-truth exit)")


if __name__ == "__main__":
    main()
