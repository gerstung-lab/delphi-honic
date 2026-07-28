"""Preprocessing-correctness gate for the honic (German cohort) dataframes.

Validates the two preprocessed parquet files the honic reader will consume before
a *pretrained* Delphi model is evaluated on them:

    df_meta.parquet  : one row per patient  -> patient_id, age_bl, gender_bl, region_bl
    df_event.parquet : the full event stream -> patient_id, value, age, idx

`idx` is the integer Delphi token id and `value` its short label; both must be
consistent with the pretrained label index (delphi_labels_index_name.csv, where
`labels_short` = name and `index`/`token` = id). Token semantics come from that
file's `type` column: covars (padding/no_event/sex/lifestyle), icdcodes (disease),
death.

Ages here are in YEARS. The model consumes DAYS, so the honic reader must apply the
x365.25 conversion; that (and patient_id id-mapping) is the reader's job, not this
gate's. Each assert below guards an invariant the delphi reader/model/eval relies on
and that would silently corrupt results if violated -- see the inline notes.
"""

from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
META_FILE = "df_meta.parquet"
EVENT_FILE = "df_event.parquet"
LABELS_FILE = "delphi_labels_index_name.csv"

MAX_AGE_YEARS = 120

# stated schema; is_string_dtype also accepts pandas "string[python]"
EVENT_COLUMNS = {"patient_id": "string", "value": "string", "age": "float", "idx": "integer"}
META_COLUMNS = {"patient_id": "string", "age_bl": "float", "gender_bl": "string", "region_bl": "string"}

_DTYPE_CHECK = {
    "string": pd.api.types.is_string_dtype,
    "float": pd.api.types.is_float_dtype,
    "integer": pd.api.types.is_integer_dtype,
}

Labels = namedtuple("Labels", "id2name ids sex disease death dup_labels")


def load_labels(csv_path) -> Labels:
    """Derive the token-id sets from the pretrained label index CSV."""
    t = pd.read_csv(csv_path)
    low = t["labels_short"].astype(str).str.lower()
    return Labels(
        id2name=dict(zip(t["index"], t["labels_short"])),
        ids=set(t["index"]),
        sex=set(t.loc[low.isin(["female", "male"]), "index"]),
        disease=set(t.loc[t["type"] == "icdcodes", "index"]),
        death=set(t.loc[t["type"] == "death", "index"]),
        dup_labels=sorted(t["labels_short"][t["labels_short"].duplicated(keep=False)].unique()),
    )


def has_columns_and_dtypes(df: pd.DataFrame, spec: dict) -> bool:
    if not set(spec).issubset(df.columns):
        return False
    return all(_DTYPE_CHECK[kind](df[col]) for col, kind in spec.items())


def no_nan(df: pd.DataFrame, allow: tuple = ()) -> bool:
    cols = [c for c in df.columns if c not in allow]
    return not df[cols].isna().any().any()


def age_in_years_range(s: pd.Series) -> bool:
    # <=120 also rejects a column accidentally left in days (would be ~10^4).
    return bool(s.notna().all() and np.isfinite(s).all() and (s >= 0).all() and (s <= MAX_AGE_YEARS).all())


def idx_in_vocab(idx: pd.Series, labels: Labels) -> bool:
    # Every id must be a real token in the pretrained vocab (idx indexes nn.Embedding);
    # an unknown id is an out-of-range lookup -> crash.
    return set(idx.unique()).issubset(labels.ids)


def idx_not_reserved(idx: pd.Series) -> bool:
    # 0 = padding, 1 = no_event: injected by the pipeline, never a real event.
    return bool((idx >= 2).all())


def value_matches_idx(df_event: pd.DataFrame, labels: Labels) -> bool:
    # Keyed on idx (not label): robust to duplicate labels_short in the CSV.
    mapped = df_event["idx"].map(labels.id2name)
    bad = df_event.loc[mapped != df_event["value"], ["patient_id", "idx", "value"]]
    if not bad.empty:
        raise AssertionError(f"{len(bad)} rows where value != labels_short[idx], e.g.\n{bad.head()}")
    return True


def referential_integrity(df_meta: pd.DataFrame, df_event: pd.DataFrame) -> bool:
    return bool(
        df_meta["patient_id"].is_unique
        and set(df_event["patient_id"]) == set(df_meta["patient_id"])
        and df_meta["patient_id"].dtype == df_event["patient_id"].dtype
    )


def sorted_by_patient_then_age(df_event: pd.DataFrame) -> bool:
    # The reader never re-sorts: it reads the last row as exit time and the first
    # occurrence as onset, so rows must be age-ascending within contiguous patient blocks.
    pid = df_event["patient_id"]
    ascending = df_event.groupby("patient_id", sort=False)["age"].apply(lambda s: s.is_monotonic_increasing).all()
    # fillna(True): row 0's shift is NA (and under nullable "string" dtype != yields NA, not True)
    contiguous = (pid != pid.shift()).fillna(True).sum() == pid.nunique()
    return bool(ascending and contiguous)


def no_duplicate_token_per_patient(df_event: pd.DataFrame) -> bool:
    dups = df_event[df_event.duplicated(["patient_id", "value"], keep=False)]
    if not dups.empty:
        raise AssertionError(
            f"{len(dups)} duplicate (patient_id, value) rows, e.g.\n{dups[['patient_id', 'value', 'idx', 'age']].head()}"
        )
    return True


def one_sex_token_per_patient(df_event: pd.DataFrame, labels: Labels) -> bool:
    # Sex is read from the stream (not df_meta) and drives every sex-stratified metric.
    counts = df_event[df_event["idx"].isin(labels.sex)].groupby("patient_id").size()
    missing = set(df_event["patient_id"].unique()) - set(counts.index)
    extra = counts.index[counts != 1].tolist()
    if missing or extra:
        raise AssertionError(
            f"sex token: {len(missing)} patients missing (e.g. {list(missing)[:5]}), "
            f"{len(extra)} with !=1 (e.g. {extra[:5]})"
        )
    return True


def every_patient_has_disease_token(df_event: pd.DataFrame, labels: Labels) -> bool:
    has = df_event.assign(_d=df_event["idx"].isin(labels.disease)).groupby("patient_id")["_d"].any()
    missing = has.index[~has].tolist()
    if missing:
        raise AssertionError(f"{len(missing)} patients with no disease token, e.g. {missing[:10]}")
    return True


def death_single_and_terminal(df_event: pd.DataFrame, labels: Labels) -> bool:
    d = df_event[df_event["idx"].isin(labels.death)]
    if d.empty:
        return True
    once = d.groupby("patient_id").size().le(1).all()
    max_age = df_event.groupby("patient_id")["age"].transform("max")
    return bool(once and (d["age"] == max_age.loc[d.index]).all())


def _norm_sex(x):
    x = str(x).strip().lower()
    return "female" if x.startswith("f") else "male" if x.startswith("m") else x


def gender_bl_sex_mismatches(df_meta: pd.DataFrame, df_event: pd.DataFrame, labels: Labels) -> int:
    tok = df_event[df_event["idx"].isin(labels.sex)].groupby("patient_id")["idx"].first().map(labels.id2name).str.lower()
    meta = df_meta.set_index("patient_id")["gender_bl"].map(_norm_sex).reindex(tok.index)
    return int((tok != meta).sum())


def test_data(data_dir=DATA_DIR, labels_path=None):
    data_dir = Path(data_dir)
    labels = load_labels(labels_path or data_dir / LABELS_FILE)
    df_meta = pd.read_parquet(data_dir / META_FILE)
    df_event = pd.read_parquet(data_dir / EVENT_FILE)

    # schema + non-empty
    assert len(df_meta) and len(df_event)
    assert has_columns_and_dtypes(df_meta, META_COLUMNS)
    assert has_columns_and_dtypes(df_event, EVENT_COLUMNS)

    # no NaN -- region_bl is the only nullable column
    assert no_nan(df_event)
    assert no_nan(df_meta, allow=("region_bl",))

    # age is finite years in [0, 120], both tables
    assert age_in_years_range(df_event["age"])
    assert age_in_years_range(df_meta["age_bl"])

    # token ids valid + consistent with the pretrained label index
    assert idx_in_vocab(df_event["idx"], labels)
    assert idx_not_reserved(df_event["idx"])
    assert value_matches_idx(df_event, labels)

    # cohort / stream integrity
    assert referential_integrity(df_meta, df_event)
    assert sorted_by_patient_then_age(df_event)
    assert no_duplicate_token_per_patient(df_event)

    # every patient needs a sex token and >=1 disease token or eval is meaningless
    assert one_sex_token_per_patient(df_event, labels)
    assert every_patient_has_disease_token(df_event, labels)
    assert death_single_and_terminal(df_event, labels)

    # non-fatal warnings (don't gate the dataframes on tokenizer / gender-encoding issues)
    if labels.dup_labels:
        print(f"WARNING: duplicate labels_short in {LABELS_FILE}: {labels.dup_labels} -- value->id is ambiguous by label")
    mism = gender_bl_sex_mismatches(df_meta, df_event, labels)
    if mism:
        print(f"WARNING: {mism} patients where df_meta.gender_bl disagrees with the df_event sex token")


# ponytail: self-check so the logic is verifiable before the real parquet files exist.
def _selfcheck():
    import tempfile

    labels = pd.DataFrame(
        {
            "token": [0, 1, 2, 3, 13, 14, 1269],
            "index": [0, 1, 2, 3, 13, 14, 1269],
            "labels_short": ["Padding", "No event", "Female", "Male", "A00", "A01", "Death"],
            "type": ["covars", "covars", "covars", "covars", "icdcodes", "icdcodes", "death"],
        }
    )
    meta = pd.DataFrame(
        {"patient_id": ["1", "2"], "age_bl": [30.0, 40.0], "gender_bl": ["female", "male"], "region_bl": ["BW", None]}
    ).astype({"patient_id": "string", "gender_bl": "string", "region_bl": "string"})
    event = pd.DataFrame(
        {
            "patient_id": ["1", "1", "2", "2", "2"],
            "value": ["Female", "A00", "Male", "A01", "Death"],
            "age": [0.0, 30.0, 0.0, 40.0, 80.0],
            "idx": [2, 13, 3, 14, 1269],
        }
    ).astype({"patient_id": "string", "value": "string", "age": "float64", "idx": "int64"})

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        labels.to_csv(d / LABELS_FILE, index=False)
        meta.to_parquet(d / META_FILE)
        event.to_parquet(d / EVENT_FILE)
        test_data(d)  # valid -> passes

        # each perturbation must trip its assert
        mutations = (
            lambda e: pd.concat([e, e.iloc[[1]]]),                # duplicate token
            lambda e: e[e["idx"] != 13].reset_index(drop=True),   # patient 1 loses its only disease
            lambda e: e.assign(age=e["age"] * 365.25),            # age in days
            lambda e: e.assign(idx=e["idx"].replace(2, 999)),     # idx out of vocab
        )
        for i, mut in enumerate(mutations):
            e2 = mut(event.copy())
            e2.to_parquet(d / EVENT_FILE)
            try:
                test_data(d)
            except AssertionError:
                pass
            else:
                raise SystemExit(f"expected AssertionError not raised for mutation {i}")

    print("selfcheck OK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate the honic df_meta/df_event parquet files (plain asserts, no pytest)."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help=f"directory holding {META_FILE} and {EVENT_FILE} (default: this script's dir)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=f"path to {LABELS_FILE} (default: <data-dir>/{LABELS_FILE})",
    )
    parser.add_argument("--selfcheck", action="store_true", help="run the built-in synthetic self-check and exit")
    args = parser.parse_args()

    if args.selfcheck:
        _selfcheck()
    else:
        test_data(args.data_dir, labels_path=args.labels)
        print(f"OK: all checks passed for {args.data_dir}")
