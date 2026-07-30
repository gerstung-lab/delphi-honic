"""Eval components for the honic AUC pipeline.

Ported from delphi/eval/auc.py, delphi/eval/utils.py, delphi/experiment.py and
delphi/model/utils.py so honic runs without importing delphi. Two changes vs upstream:
  * sample_boolean_mask takes an optional torch.Generator (reproducible eval);
  * nearest_prediction is a NEW legacy-model stand-in for model.intensity() -- the
    legacy Delphi model returns plain (logits, loss, att) and has no intensity()/
    tpp head, so we recover "the prediction at the input step strictly before each
    query time, with already-occurred marks extinguished" from its raw logits.

Conventions (same as the model + dataset.py): ages in DAYS, sequences age-sorted and
LEFT-padded with -1e4, mark 0 = padding, mark 1 = no_event.
"""

import numpy as np
import torch
from scipy.stats import rankdata

# Tokens exempt from already-occurred extinguishment: padding (0) and no_event (1)
# are structural / recurring, not diseases, so they are never masked out.
DEFAULT_TERMINATE_EXCEPT = (0, 1)


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #
def eval_iter(total_size: int, batch_size: int):
    """Yield contiguous index blocks covering [0, total_size) once (no shuffling)."""
    batch_start_pos = np.arange(0, total_size, batch_size)
    batch_end_pos = batch_start_pos + batch_size
    batch_end_pos[-1] = total_size
    for start, end in zip(batch_start_pos, batch_end_pos):
        yield np.arange(start, end)


# --------------------------------------------------------------------------- #
# AUC
# --------------------------------------------------------------------------- #
def batched_mann_whitney_auc(
    scores: np.ndarray, ctl: np.ndarray, case: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise AUC over an (N, V) score matrix.

    Scores outside `ctl | case` or NaN are excluded from ranking (per column).
    Returns (ctl_counts, case_counts, auc), each of shape (V,). Callers that need to
    cap memory should pass column blocks (rankdata allocates a full float64 rank
    matrix + int64 argsort); per-column AUC is independent, so block-wise is exact.
    """
    assert scores.shape == ctl.shape == case.shape
    masked = np.where(ctl | case, scores, np.nan)
    ranks = rankdata(masked, method="average", axis=0, nan_policy="omit")

    valid = ~np.isnan(masked)
    n1 = (ctl & valid).sum(axis=0)
    n2 = (case & valid).sum(axis=0)
    R1 = np.where(ctl & valid, ranks, 0).sum(axis=0)

    U1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    denom = n1 * n2
    auc = np.full(denom.shape, np.nan, dtype=float)
    np.divide(U1, denom, out=auc, where=denom > 0)
    return n1, n2, auc


def harrell_cindex(estimate, event_times, censor_times, anchor):
    """Harrell's C-index for a forecast anchor (horizon-free). All args are (N,).

    estimate    : risk score (higher = earlier event expected),
    event_times : absolute first-occurrence age (NaN if the event never occurs),
    censor_times: absolute last-seen age (exit) -- death censors here (cause-specific),
    anchor      : t0, the prompt cutoff age.
    Prevalent participants (event strictly before the anchor) are dropped; the clock
    is re-zeroed at the anchor. Returns {"cindex", "n_event"} (cindex NaN if undefined).
    Uses sksurv's C-optimized concordance (lazy import so `import eval` stays light).
    """
    from sksurv.metrics import concordance_index_censored

    occ = event_times
    keep = ~(~np.isnan(occ) & (occ < anchor))  # drop prevalent (event before t0)
    event = ~np.isnan(occ[keep]) & (occ[keep] >= anchor[keep])
    time = np.where(event, occ[keep], censor_times[keep]) - anchor[keep]
    time = np.maximum(time, 1e-3)  # sksurv requires strictly positive times
    est = estimate[keep]

    out = {"cindex": float("nan"), "n_event": int(event.sum())}
    if out["n_event"] == 0:
        return out  # no cases -> concordance undefined
    try:
        out["cindex"] = float(concordance_index_censored(event, time, est)[0])
    except Exception:  # e.g. no comparable pairs -- NaN, don't kill the run
        pass
    return out


# --------------------------------------------------------------------------- #
# collators
# --------------------------------------------------------------------------- #
def sample_boolean_mask(mask: torch.Tensor, generator: torch.Generator | None = None):
    """Sample one True value per row from a boolean mask (vectorized).

    Pass a torch.Generator for a reproducible pick (upstream delphi is unseeded).
    """
    n_rows = mask.shape[0]
    result = torch.zeros_like(mask).bool()

    counts = mask.sum(dim=1)
    has_true = counts > 0
    if not has_true.any():
        return result

    # random score per position on the mask's own device; -inf out the False slots
    random_positions = torch.rand(mask.shape, device=mask.device, generator=generator)
    random_positions[~mask] = -torch.inf

    selected_cols = torch.argmax(random_positions, dim=1)
    result[torch.arange(n_rows, device=mask.device), selected_cols] = has_true
    return result


class _BufferedCollator:
    """Shared accumulation: with n_participants, write each batch into a preallocated
    (N, ...) CPU buffer at a running row offset -- no per-batch list and no
    torch.concat at the end (concat transiently doubles memory: list + result). With
    n_participants=None, fall back to appending + concat (old behaviour)."""

    def __init__(self, n_participants: int | None):
        self.n = n_participants
        self._rates = self._times = None
        self._pos = 0
        self._rate_list = list()
        self._time_list = list()

    def _store(self, rates: torch.Tensor, times: torch.Tensor):
        if self.n is None:
            self._rate_list.append(rates)
            self._time_list.append(times)
            return
        if self._rates is None:  # preallocate once shapes are known -> no concat later
            self._rates = torch.empty((self.n, *rates.shape[1:]), dtype=rates.dtype)
            self._times = torch.empty((self.n, *times.shape[1:]), dtype=times.dtype)
        b = rates.shape[0]
        self._rates[self._pos : self._pos + b] = rates
        self._times[self._pos : self._pos + b] = times
        self._pos += b

    def finalize(self):
        if self.n is not None:
            return self._rates[: self._pos], self._times[: self._pos]
        return torch.concat(self._rate_list), torch.concat(self._time_list)


class AgeStratRatesCollator(_BufferedCollator):

    def __init__(self, age_groups: torch.Tensor, n_participants: int | None = None, generator: torch.Generator | None = None):
        super().__init__(n_participants)
        self.age_groups = age_groups
        self.generator = generator

    def step(self, timesteps: torch.Tensor, logits: torch.Tensor):
        batch_size = logits.shape[0]
        n_age_bins = len(self.age_groups) - 1
        bin_assignments = torch.searchsorted(self.age_groups, timesteps, right=True)
        bin_assignments -= 1

        ctl_rates = list()
        ctl_times = list()
        for bin_idx in range(n_age_bins):
            bin_mask = sample_boolean_mask(bin_assignments == bin_idx, generator=self.generator)
            ctl_rate = torch.full(
                (batch_size, logits.shape[-1]),
                dtype=logits.dtype,
                fill_value=torch.nan,
            ).to(logits.device)
            ctl_time = torch.full(
                (batch_size,), dtype=timesteps.dtype, fill_value=torch.nan
            ).to(logits.device)
            ctl_rate[bin_mask.any(dim=-1)] = logits[bin_mask, :]
            ctl_time[bin_mask.any(dim=-1)] = timesteps[bin_mask]
            ctl_rates.append(ctl_rate)
            ctl_times.append(ctl_time)
        ctl_rates = torch.stack(ctl_rates, dim=1)
        ctl_times = torch.stack(ctl_times, dim=1)
        self._store(ctl_rates.detach().cpu(), ctl_times.detach().cpu())


class DiseaseRatesCollator(_BufferedCollator):

    def __init__(self, targets: torch.Tensor, n_participants: int | None = None):
        super().__init__(n_participants)
        self.targets = targets

    def step(self, tokens: torch.Tensor, timesteps: torch.Tensor, logits: torch.Tensor):
        dis_time = torch.full(
            (logits.shape[0], logits.shape[-1]),
            dtype=timesteps.dtype,
            fill_value=torch.nan,
        ).to(logits.device)
        dis_time.scatter_(index=tokens, src=timesteps, dim=1)

        dis_rate = torch.full(
            (logits.shape[0], logits.shape[-1]),
            dtype=logits.dtype,
            fill_value=torch.nan,
        ).to(logits.device)
        uniq_tokens = torch.unique(tokens)
        uniq_tokens = uniq_tokens[torch.isin(uniq_tokens, self.targets)]
        for token in uniq_tokens:
            have_disease = tokens == token
            dis_rate[have_disease.any(dim=1), token] = logits[have_disease][:, token]
        # finalize() returns (rates, times) -> store in that order
        self._store(dis_rate.detach().cpu(), dis_time.detach().cpu())


# --------------------------------------------------------------------------- #
# legacy-model intensity bridge
# --------------------------------------------------------------------------- #
def lookup(t0: torch.Tensor, query_t: torch.Tensor):
    """Strict-before nearest-input lookup (ports delphi model.utils.lookup).

    t0: (B, L0) age-sorted ascending, LEFT-padded with -1e4. query_t: (B, Q).
    For each query age, find the nearest input event in t0 *strictly before* it.
    Returns (idx, nearest_t, invalid), each (B, Q): idx clamped >=0; nearest_t is
    that input's age (-1e4 where invalid); invalid marks queries with no non-pad
    input strictly before them (no such input, or the strict-before lands on a pad).
    """
    B = t0.shape[0]
    q_shape = query_t.shape[1:]
    t0 = t0.contiguous()
    q_flat = query_t.reshape(B, -1).contiguous()
    # right=False -> first i with t0[i] >= q  ->  idx-1 = last t0[i] < q (strict before)
    idx = torch.searchsorted(t0, q_flat, right=False) - 1
    idx = idx.reshape(B, *q_shape)
    invalid = idx == -1
    idx = idx.clamp(min=0)
    nearest_t = torch.take_along_dim(t0, idx.reshape(B, -1), dim=1).reshape(idx.shape)
    invalid = invalid | (nearest_t == -1e4)
    nearest_t = nearest_t.masked_fill(invalid, -1e4)
    return idx, nearest_t, invalid


def have_occurred(history_x: torch.Tensor, terminate_except, vocab_size: int):
    """Per-history cumulative-seen mask (ports delphi model.utils.have_occurred).

    Returns (B, L, V) bool: [b, j, v] is True iff token v appeared in
    history_x[b, 0..j] (tokens in terminate_except are ignored / mapped to pad).
    """
    terminate_except = torch.as_tensor(terminate_except, device=history_x.device)
    fill = history_x.clone()
    fill[torch.isin(fill, terminate_except)] = 0
    B, L = fill.shape
    one_hot = torch.zeros(B, L, vocab_size, device=fill.device)
    one_hot.scatter_(2, fill.unsqueeze(-1).long(), 1.0)
    return one_hot.cumsum(dim=1) > 0


def nearest_prediction(
    x0: torch.Tensor,
    t0: torch.Tensor,
    logits: torch.Tensor,
    query_t: torch.Tensor,
    terminate_except=DEFAULT_TERMINATE_EXCEPT,
    termination_token: int | None = None,
):
    """Legacy-model stand-in for model.intensity().

    The legacy Delphi forward returns plain (logits, loss, att); this recovers the
    model's per-token prediction at the input step *strictly before* each query
    time, with already-occurred marks extinguished -- the eval read the newer
    model does inside intensity(), minus the death-termination branch the legacy
    model has no notion of.

    Args (all share batch B and input length L0; ages in days, left-padded -1e4):
        x0:      (B, L0) input token ids.
        t0:      (B, L0) input ages, ascending.
        logits:  (B, L0, V) model logits at each input position.
        query_t: (B, Q) query ages (e.g. next-event ages t1, minus any offset).
    Returns (scores, nearest_t):
        scores:    (B, Q, V) logits gathered at the strict-before position; marks
                   already occurred by that position -> -inf (rank lowest / rate 0),
                   queries with no strict-before history -> NaN (excluded downstream).
        nearest_t: (B, Q) age of that input step (-1e4 where invalid).
    termination_token: if given, positions whose history already includes this token
        (e.g. death) -> the whole prediction is NaN (the model never saw tokens after it).
    Rank-identical to exp'd intensities since AUC is rank-based.
    """
    V = logits.shape[-1]
    idx, nearest_t, invalid = lookup(t0, query_t)
    scores = torch.take_along_dim(logits, idx.unsqueeze(-1), dim=1)  # (B, Q, V)
    occurred = torch.take_along_dim(
        have_occurred(x0, terminate_except, V), idx.unsqueeze(-1), dim=1
    )  # (B, Q, V)
    scores = scores.masked_fill(occurred, float("-inf"))
    if termination_token is not None:  # after death: no valid prediction
        scores = scores.masked_fill(occurred[..., termination_token].unsqueeze(-1), float("nan"))
    scores = scores.masked_fill(invalid.unsqueeze(-1), float("nan"))
    return scores, nearest_t


def nearest_prediction_at(
    x0: torch.Tensor,
    t0: torch.Tensor,
    logits: torch.Tensor,
    query_t: torch.Tensor,
    tokens: torch.Tensor,
    terminate_except=DEFAULT_TERMINATE_EXCEPT,
    termination_token: int | None = None,
):
    """Per-(query, token) nearest_prediction: the legacy model's logit for a specific
    token at the input step strictly before each query time -- avoids the (B, Q, V)
    intermediate (only one token per query is needed, e.g. the concordance read).

    x0, t0: (B, L0); logits: (B, L0, V); query_t: (B, Q); tokens: (Q,) or (B, Q).
    Returns (scores (B, Q), nearest_t (B, Q)): the token's logit, -inf if already
    occurred by that step, NaN if there is no strict-before history. termination_token:
    if given, reads whose history already includes it (death) -> NaN (post-termination).
    """
    V = logits.shape[-1]
    idx, nearest_t, invalid = lookup(t0, query_t)  # (B, Q)
    B, Q = idx.shape
    tok = torch.broadcast_to(tokens, idx.shape)
    b = torch.arange(B, device=logits.device).unsqueeze(1).expand(B, Q)
    occ = have_occurred(x0, terminate_except, V)  # (B, L0, V)
    scores = logits[b, idx, tok]  # (B, Q)
    scores = scores.masked_fill(occ[b, idx, tok], float("-inf"))  # token already occurred
    if termination_token is not None:
        scores = scores.masked_fill(occ[b, idx, termination_token], float("nan"))  # after death
    scores = scores.masked_fill(invalid, float("nan"))
    return scores, nearest_t


class ConcordanceCollator:
    """Dynamic-AUC (time-dependent) concordance for the legacy model. Ported from
    delphi eval.ConcordanceCollator, using nearest_prediction_at (no intensity_at) and
    with an optional same-region constraint.

    A case event = each (participant, disease) pair the participant develops, with its
    frozen-history score (case_scores) and onset age (case_times). For each case, over
    every control batch, it counts at-risk controls -- matched to the case on every
    `covariates` code (e.g. sex, region) -- whose model score for the case's disease AT
    THE CASE'S ONSET AGE is below the case's score. Accumulates concordant/total pairs
    per case event; c-index = sum/sum.
    """

    def __init__(
        self,
        dis_rates: torch.Tensor,  # (N, V) case scores; NaN where not a case
        case_times: torch.Tensor,  # (N, V) onset-age matrix (days), NaN if never
        last_t: torch.Tensor,  # (N,) each participant's last recorded age (days); the follow-up end
        covariates: list[torch.Tensor] | None = None,  # (N,) code tensors; a control must MATCH the case on each
        chunk_size: int = 8192,
        max_lag: float = 365.25,  # days (1 year)
        termination_token: int | None = None,  # death: controls read after it -> NaN -> dropped
    ):
        # flatten case events; chunk rows so no nonzero() call exceeds INT_MAX elements
        mask = ~torch.isnan(dis_rates)
        step = max(1, (2**31 - 1) // dis_rates.shape[1])
        idx = [(r + s, c) for s in range(0, dis_rates.shape[0], step)
               for r, c in [mask[s:s + step].nonzero(as_tuple=True)]]
        cp = torch.cat([r for r, _ in idx])
        ct = torch.cat([c for _, c in idx])
        self.case_scores = dis_rates[cp, ct].float()
        self.case_times_mat = case_times  # (N, V) onset matrix (for the at-risk check)
        self.case_times = case_times[cp, ct].float()
        self.case_tokens = ct
        self.case_participants = cp
        self.last_t = last_t
        self.covariates = covariates or []  # controls matched to the case on each (sex, region, ...)
        self.chunk_size = chunk_size
        self.max_lag = max_lag
        self.termination_token = termination_token
        E = len(cp)
        self.concordant_pairs = np.zeros(E, dtype=np.float64)
        self.total_pairs = np.zeros(E, dtype=np.float64)
        self.participant_offset = 0
        self.device = self.case_scores.device

    def step(self, x0: torch.Tensor, t0: torch.Tensor, logits: torch.Tensor):
        """Score this batch's participants as controls against every case event."""
        B = t0.shape[0]
        E_total = len(self.case_tokens)
        j = torch.arange(B, device=self.device) + self.participant_offset  # control global ids
        for s in range(0, E_total, self.chunk_size):
            e = min(s + self.chunk_size, E_total)
            cct = self.case_times[s:e]  # (E_c,)
            ctok = self.case_tokens[s:e]
            cpart = self.case_participants[s:e]
            cscore = self.case_scores[s:e]
            # legacy: control's rate for the case's token at the case's onset age
            ctrl, t_at = nearest_prediction_at(
                x0, t0, logits, cct.unsqueeze(0).expand(B, -1), ctok, termination_token=self.termination_token
            )  # (B, E_c); dead controls (read after death) -> NaN -> dropped below
            valid = t_at >= 0  # has strict-before history (not padding)
            valid &= ~torch.isnan(ctrl)  # no history / occurred-NaN controls drop out
            # drop controls whose last record ended > max_lag before the case onset (stale / not
            # observed near the case time). last_t[j] is the control's follow-up end.
            valid &= (cct.unsqueeze(0) - self.last_t[j].unsqueeze(1)) <= self.max_lag
            j_onset = self.case_times_mat[j.unsqueeze(1), ctok.unsqueeze(0).expand(B, -1)]
            valid &= j_onset.isnan() | (j_onset > cct.unsqueeze(0))  # at-risk: control has not developed it yet
            valid &= j.unsqueeze(1) != cpart.unsqueeze(0)  # not the case itself
            for cov in self.covariates:  # control must match the case on every covariate (sex, region, ...)
                valid &= cov[j].unsqueeze(1) == cov[cpart].unsqueeze(0)
            self.concordant_pairs[s:e] += (valid & (ctrl.float() < cscore.unsqueeze(0))).sum(0).cpu().numpy()
            self.total_pairs[s:e] += valid.sum(0).cpu().numpy()
        self.participant_offset += B

    def finalize(self):
        return self.case_tokens.cpu().numpy(), self.total_pairs, self.concordant_pairs
