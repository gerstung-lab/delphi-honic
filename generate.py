"""Autoregressive generation for the legacy Delphi model.

Adapted from the clean delphi/model/transformer.py generate(), but WITHOUT KV
caching: the legacy model (legacy_model.Delphi, forward -> (logits, loss, att)) has
no incremental-decode path, so each step re-runs a full forward over the whole
running sequence. Everything else -- per-sequence early stop, age/block/count caps,
the 0/1/2/3 mask, age-cap censoring, and the sort+trim assembly -- is kept.

Sampling is the legacy competing-exponentials scheme: token k has per-day rate
exp(logit_k), so its waiting time is t_k = -log(U_k) * exp(-logit_k); the soonest
(argmin over k) wins, at age = latest real age + t_k. ignore_tokens and already-seen
tokens (no_repeat, keeping no_event=1 repeatable) are masked to -inf -> t_k = t_max.
"""

import torch

T_MAX = 365 * 80.0  # waiting-time clamp (days), matches the legacy generate


def sample_next(logits, idx, ignore_tokens, no_repeat=True, t_max=T_MAX):
    """One legacy generation step.

    logits: (B, T, V) from Delphi.forward; idx: (B, T) running token history.
    Returns (idx_next (B, 1) long, time_til_next (B, 1) days) -- one event per row.
    """
    logits = logits[:, -1, :].clone()  # (B, V) prediction at the last position
    logits[:, ignore_tokens] = -float("inf")
    if no_repeat:  # forbid repeating a token already in history; no_event (1) stays repeatable
        fill = idx.clone()
        fill[fill == 1] = 0  # remap no_event to pad-col 0 (already -inf) so it is NOT forbidden
        logits = logits.scatter(1, fill, -float("inf"))
    # competing exponentials: t_k = -exp(-logit_k) * log(U_k); masked -> +inf -> clamps to t_max
    u = torch.rand_like(logits)
    t = torch.clamp(-torch.exp(-logits) * u.log(), min=0.0, max=t_max)
    time_til_next, idx_next = t.min(dim=1)
    return idx_next[:, None], time_til_next[:, None]


def generate(
    model,
    idx,
    age,
    termination_tokens,
    max_new_tokens=None,
    max_age=85 * 365.25,
    stop_at_block_size=True,
    exclude_pad=True,
    censor=True,
    no_repeat=True,
):
    """Continue each (idx, age) prompt until it terminates, ages out, fills the block,
    or hits max_new_tokens. idx: (B, T) left-padded token ids (0=pad); age: (B, T) ages
    in DAYS (-1e4 pad). Returns (idx, age, misc) with misc = {n_prompt, n_gen, mask};
    mask is (B, L) with 0=pad, 1=prompt, 2=continuation, 3=censored (age-capped)."""
    device = idx.device
    termination_tokens = torch.as_tensor(termination_tokens, dtype=torch.int64, device=device)
    if max_new_tokens is None:
        max_new_tokens = float("inf")

    if max_age is None:
        pass
    elif isinstance(max_age, torch.Tensor):
        assert max_age.shape == (age.shape[0],)
        max_age = max_age.unsqueeze(1)  # (B, 1)
    else:
        max_age = torch.full((age.shape[0], 1), float(max_age), device=device)

    ignore_tokens = [0] + list(model.config.ignore_tokens or [])

    batch_size = idx.shape[0]
    active_indices = torch.arange(batch_size, device=device)
    completed_idx, completed_age, completed_mask = {}, {}, {}
    cur_idx = idx.clone()
    cur_age = age.clone()
    cur_mask = (cur_idx > 0).long()  # 1=prompt (real token), 0=pad
    pmt_cnt = (idx > 0).sum(dim=1)
    gen_cnt = torch.zeros_like(pmt_cnt)

    with torch.no_grad():
        while len(active_indices) > 0:
            logits, _, _ = model(cur_idx, cur_age)  # full forward every step (no KV cache)
            idx_next, time_til_next = sample_next(logits, cur_idx, ignore_tokens, no_repeat=no_repeat)
            # baseline = latest real age per row (max is robust to -1e4 pads in the last column)
            age_next = cur_age.max(dim=1, keepdim=True).values + time_til_next

            gen_cnt[active_indices] += (idx_next > 0).sum(dim=1)
            cur_idx = torch.cat((cur_idx, idx_next), dim=1)
            cur_age = torch.cat((cur_age, age_next), dim=1)
            cur_mask = torch.cat((cur_mask, (idx_next > 0).long() * 2), dim=1)

            terminated = torch.isin(idx_next, termination_tokens).any(-1)
            if max_age is None:
                aged_out = torch.zeros_like(terminated)
            else:
                aged_out = (age_next > max_age[active_indices]).any(-1)
            if stop_at_block_size and model.config.block_size is not None:
                length = (cur_idx != 0).sum(dim=1) if exclude_pad else torch.full_like(active_indices, cur_idx.shape[1])
                reached_block = length >= model.config.block_size
            else:
                reached_block = torch.zeros_like(terminated)
            maxed_out = gen_cnt[active_indices] >= max_new_tokens
            should_stop = terminated | aged_out | reached_block | maxed_out

            if should_stop.any():
                for local_i in torch.where(should_stop)[0]:
                    global_i = active_indices[local_i].item()
                    completed_idx[global_i] = cur_idx[local_i]
                    completed_age[global_i] = cur_age[local_i]
                    completed_mask[global_i] = cur_mask[local_i]
                keep = ~should_stop
                cur_idx, cur_age, cur_mask = cur_idx[keep], cur_age[keep], cur_mask[keep]
                active_indices = active_indices[keep]

    max_len = max(t.numel() for t in completed_idx.values())
    final_idx = torch.full((batch_size, max_len), 0, dtype=idx.dtype, device=device)
    final_age = torch.full((batch_size, max_len), -1e4, dtype=age.dtype, device=device)
    final_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i in range(batch_size):  # left-pad each completed row into the block
        ii, ai, mi = completed_idx[i], completed_age[i], completed_mask[i]
        final_idx[i, -ii.numel():] = ii
        final_age[i, -ai.numel():] = ai
        final_mask[i, -mi.numel():] = mi

    if max_age is not None and censor:
        # the overflow event (age > max_age) becomes a no_event clamped to max_age, marked censored
        censored = final_age > max_age
        final_idx[censored] = 1
        final_mask[censored] = 3
        final_age = torch.clamp(final_age, max=max_age)

    order = torch.argsort(final_age, dim=1)  # ascending -> -1e4 pads sort left
    final_age = torch.take_along_dim(final_age, order, dim=1)
    final_idx = torch.take_along_dim(final_idx, order, dim=1)
    final_mask = torch.take_along_dim(final_mask, order, dim=1)

    margin = torch.min((final_idx == 0).sum(dim=1)).item()  # trim the common left-pad
    final_idx, final_age, final_mask = final_idx[:, margin:], final_age[:, margin:], final_mask[:, margin:]
    final_mask = final_mask.masked_fill((final_idx == 0) | (final_age == -1e4), 0)  # re-assert pad-is-0

    return final_idx, final_age, {
        "n_prompt": pmt_cnt.detach().cpu().numpy(),
        "n_gen": gen_cnt.detach().cpu().numpy(),
        "mask": final_mask,
    }
