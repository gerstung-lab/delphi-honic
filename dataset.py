"""honic data loading for a pretrained delphi-m4 forward + AUC/c-index eval.

De-abstracted port of delphi's TokenReader + MultimodalUKBReader (-> HonicReader)
and MultimodalDataset (-> Dataset). No ABCs, no subclassing, no Transform class,
no biomarker/expansion-pack machinery (honic has neither). Transforms are applied
as plain functions from utils.py.

Conventions the pretrained model depends on (get these wrong -> silently wrong
metrics, no crash):
  * time is in DAYS. df_event.age is YEARS, so the reader multiplies by 365.25 once,
    at load. Everything downstream (no_event's 36525-day range, the model's /365.25
    age encoding, the -1e4 age pad) is day-space.
  * token ids are the CSV `index` verbatim (0=padding, 1=no_event, 2=female, 3=male,
    1269=death, 13..1268=ICD). Never renumber.
  * sex is read from the token stream (id 2/3), not df_meta.gender_bl.
  * df_event is pre-validated (data/tests.py): age-ascending within contiguous
    patient blocks, no duplicate token per patient, >=2 events/patient. The reader
    relies on that order and never re-sorts the stored stream.
"""

import numpy as np
import pandas as pd
import torch

from utils import append_no_event, collate_batch, crop_contiguous, exclude_tokens, sort_by_time

DAYS_PER_YEAR = 365.25
NO_EVENT_TOKEN = 1


class HonicReader:
    """Single-participant (x, t) store + trajectory queries over the honic cohort.

    x = token ids (uint32), t = age in DAYS (float32). Keyed by STRING patient_id.
    """

    def __init__(self, df_event_path, df_meta_path, labels_csv_path="data/delphi_labels_index_name.csv"):
        labels = pd.read_csv(labels_csv_path)
        # lowercase keys so 'female'/'male'/'death' resolve (CSV labels_short are capitalised)
        self.tokenizer = {name.lower(): int(i) for name, i in zip(labels["labels_short"], labels["index"])}
        # id-keyed so duplicate labels_short (e.g. O01, see tests.py warning) can't collide
        self.detokenizer = dict(zip(labels["index"].astype(int), labels["labels_short"]))
        self.vocab_size = len(labels)
        self.female_token = self.tokenizer["female"]
        self.death_token = self.tokenizer["death"]
        self.biomarker2idx = {}  # honic has no biomarkers; kept so downstream reads don't KeyError

        df = pd.read_parquet(df_event_path, columns=["patient_id", "age", "idx"])
        self.tokens = df["idx"].to_numpy(np.uint32)
        self.timesteps = (df["age"].to_numpy(np.float32) * DAYS_PER_YEAR).astype(np.float32)  # YEARS -> DAYS
        pids = df["patient_id"].to_numpy()  # string/object
        # df_event is validated contiguous+sorted, so first-occurrence gives block starts
        uniq, first_idx, counts = np.unique(pids, return_index=True, return_counts=True)
        self.start_pos = dict(zip(uniq, first_idx))
        self.seq_len = dict(zip(uniq, counts))

        meta = pd.read_parquet(df_meta_path, columns=None)
        self._participants = meta["patient_id"].to_numpy()
        self._fold_of = meta["fold"].to_numpy() if "fold" in meta.columns else None
        # per-patient baseline region (the one nullable covariate); NA -> None
        self._region_of = {
            pid: (None if pd.isna(r) else r)
            for pid, r in zip(meta["patient_id"], meta["region_bl"])
        }
        self._age_bl_of = dict(zip(meta["patient_id"], meta["age_bl"]))  # baseline age (years)

    def __getitem__(self, pid):
        i = self.start_pos[pid]
        l = self.seq_len[pid]
        x = self.tokens[i : i + l].astype(np.uint32)
        t = self.timesteps[i : i + l].astype(np.float32)
        return x, t

    def participants(self, fold=None) -> np.ndarray:
        if fold is None:
            return self._participants
        # ponytail: fold source = a 'fold' column in df_meta; repoint here if splits live elsewhere.
        if self._fold_of is None:
            raise ValueError("fold requested but df_meta has no 'fold' column")
        return self._participants[self._fold_of == fold]

    def is_female(self, pids) -> np.ndarray:
        out = np.zeros(len(pids), dtype=bool)
        for i, pid in enumerate(pids):
            j = self.start_pos[pid]
            out[i] = (self.tokens[j : j + self.seq_len[pid]] == self.female_token).any()
        return out

    def region(self, pids) -> np.ndarray:
        """(N,) baseline region (df_meta.region_bl) per pid, aligned to `pids`.
        Object array; None where region is missing (region_bl is the one nullable column)."""
        return np.array([self._region_of.get(p) for p in pids], dtype=object)

    def exit_times(self, pids) -> np.ndarray:
        """(N,) last-token age in DAYS = censor/exit age (stream is age-ascending)."""
        out = np.empty(len(pids), dtype=np.float32)
        for i, pid in enumerate(pids):
            out[i] = self.timesteps[self.start_pos[pid] + self.seq_len[pid] - 1]
        return out

    def event_times(self, pids) -> np.ndarray:
        """(N, max_id+1) first-occurrence age in DAYS per token; NaN where absent.
        Width is max token id + 1 (== vocab_size for a dense id space) so a token id
        indexes its own column even if the id space is not 0..N-1 contiguous."""
        width = max(self.tokenizer.values()) + 1
        out = np.full((len(pids), width), np.nan, dtype=np.float32)
        for i, pid in enumerate(pids):
            x, t = self[pid]
            uniq, first_idx = np.unique(x, return_index=True)  # x is time-ordered -> earliest per token
            out[i, uniq] = t[first_idx]
        return out

    def resolve_prompt_age(self, at: float) -> dict:
        """Per-participant forecast cutoff for `at` YEARS after each baseline.

        cutoff_age (DAYS) = (age_bl + at) * 365.25. Returns {pid: cutoff_days} for
        participants with follow-up past the cutoff (exit age > cutoff) -- i.e.
        someone to forecast into; others are dropped. The dict keys define the
        forecast cohort; pass it straight to Dataset(prompt_age=...)."""
        pids = self._participants
        age_bl = np.array([self._age_bl_of[p] for p in pids], dtype=np.float64)
        cutoff = (age_bl + at) * DAYS_PER_YEAR  # (N,) days
        keep = self.exit_times(pids) > cutoff
        return {p: float(c) for p, c, k in zip(pids, cutoff, keep) if k}


class Dataset:
    """Builds the padded batch a pretrained forward consumes.

    Transforms are applied inline as functions (no Transform class). Matches
    delphi's eval config: no_event insertion ON, deterministic per-pid. Emits the
    slim token-only batch (X0, T0, X1, T1) -- honic's model.py takes no biomarkers.

    Two modes for the (x0, x1) split in __getitem__:
      * frozen-history (default): next-token shift x0=x[:-1], x1=x[1:].
      * forecast (prompt_age set): x0/t0 = events up to each pid's cutoff (the
        prompt), x1/t1 = the full trajectory (ground truth). prompt_age is a
        {pid: cutoff_days} dict, e.g. from HonicReader.resolve_prompt_age.
    """

    def __init__(
        self,
        reader: HonicReader,
        pids: np.ndarray,
        *,
        no_event_interval: float | None = 5 * DAYS_PER_YEAR,
        no_event_mode: str = "legacy-random",
        blacklist_tokens: list | None = None,
        block_size: int | None = None,
        crop_mode: str = "right",
        prompt_age: dict | None = None,
        append_no_event_at_cutoff: bool = False,
        seed: int = 42,
        deterministic: bool = True,
    ):
        self.reader = reader
        self.participants = pids
        self.tokenizer = reader.tokenizer
        self.no_event_interval = no_event_interval
        self.no_event_mode = no_event_mode
        self.blacklist_tokens = None if blacklist_tokens is None else np.asarray(blacklist_tokens)
        self.block_size = block_size
        self.crop_mode = crop_mode
        self.prompt_age = prompt_age
        self.append_no_event_at_cutoff = append_no_event_at_cutoff
        self.seed = seed
        self.deterministic = deterministic
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return self.participants.size

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def _transform(self, x, t):
        # per-pid deterministic rng (matches delphi eval): reproducible across the
        # repeated passes AUC/c-index make over the same participant.
        rng = np.random.default_rng(int(x.sum()) + self.seed) if self.deterministic else self._rng
        if self.blacklist_tokens is not None:
            x, t = exclude_tokens(x, t, self.blacklist_tokens)
        if self.no_event_interval is not None:
            x, t = append_no_event(x, t, rng, self.no_event_interval, self.no_event_mode, token=NO_EVENT_TOKEN)
        t, x = sort_by_time(t, x, stable=self.deterministic)
        if self.block_size is not None:
            x, t = crop_contiguous(x, t, block_size=self.block_size, rng=rng, mode=self.crop_mode)
        return x, t

    def __getitem__(self, idx: int):
        pid = self.participants[idx]
        x, t = self.reader[pid]
        x, t = self._transform(x, t)
        if self.prompt_age is not None:
            # forecast: prompt = events up to the pid's cutoff; ground truth = full trajectory
            cutoff = self.prompt_age[pid]
            mask = t <= cutoff
            x0, t0 = x[mask], t[mask]  # boolean-mask indexing already copies
            if self.append_no_event_at_cutoff:
                x0 = np.append(x0, NO_EVENT_TOKEN)
                t0 = np.append(t0, np.float32(cutoff))
            return x0, t0, x.copy(), t.copy()
        # frozen-history next-token shift; .copy() so x0/x1 don't alias the same buffer
        x0, x1 = x[:-1].copy(), x[1:].copy()
        t0, t1 = t[:-1].copy(), t[1:].copy()
        return x0, t0, x1, t1

    def sort_by_length(self, descending: bool = True) -> np.ndarray:
        """Reorder participants by post-transform length (minimises padding) and
        RETURN the new order. Callers must rebind their per-pid arrays to it."""
        lengths = np.array([self[i][0].size for i in range(len(self))])
        order = np.argsort(lengths, kind="stable")
        if descending:
            order = order[::-1]
        self.participants = self.participants[order]
        return self.participants

    def get_batch(self, batch_idx):
        return self.collate([self[i] for i in batch_idx])

    def collate(self, samples):
        """LEFT-pad a list of (x0,t0,x1,t1) into (X0,T0,X1,T1) torch tensors.
        Tokens -> long (pad 0); ages -> float32 (pad -1e4, day-space sentinel)."""
        X0, T0, X1, T1 = zip(*samples)
        return (
            torch.tensor(collate_batch(list(X0)), dtype=torch.long),
            torch.tensor(collate_batch(list(T0), fill_value=-1e4), dtype=torch.float32),
            torch.tensor(collate_batch(list(X1)), dtype=torch.long),
            torch.tensor(collate_batch(list(T1), fill_value=-1e4), dtype=torch.float32),
        )
