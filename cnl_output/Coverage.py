# Analyze coverage of test templates by train templates using exact, forward (first-N words),
# and backward (last-N words) matching strategies. Save coverage results and matched rows to CSV files.
# RTSREC001 - Rector Ratsaka

import re
import pandas as pd
from typing import List, Tuple, Dict

# convert EC1, PC3 to EC, PC
ECPC_RE = re.compile(r'\b(EC|PC)\d+\b', re.IGNORECASE)

def normalize_template(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text).strip()
    t = ECPC_RE.sub(lambda m: m.group(1).upper(), t)  
    return re.sub(r'\s+', ' ', t)

def first_n_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n]) if n > 0 else ""

def last_n_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[-n:]) if n > 0 else ""

# lookup builder
def build_lookup(series: pd.Series, key_fn) -> Dict[str, str]:
    """
    Map key -> one representative normalized train template.
    key_fn transforms a normalized template into the matching key
    (e.g., first_n_words, last_n_words, or identity for full).
    """
    out = {}
    for val in series.dropna().map(normalize_template).unique():
        key = key_fn(val)
        # keep the first representative encountered
        if key and key not in out:
            out[key] = val
    return out

# Save only matched rows
def save_matches_filtered(test_df: pd.DataFrame,
                          templ_col: str,
                          key_series: pd.Series,
                          matches: pd.Series,
                          matched_train_map: Dict[str, str],
                          extra_col_name: str,
                          out_path: str):
    """
    Save only rows where matches == True, with:
      - original test templated_question
      - normalized_template
      - key segment (first/last n words or full)
      - matched_train_template (representative from train)
    """
    df = test_df.copy()
    df["normalized_template"] = test_df[templ_col].map(normalize_template)
    df[extra_col_name] = key_series
    df["match"] = matches

    # keep only matched
    matched_df = df[df["match"]].copy()

    # attach representative matched train template
    def map_train(tkey: str) -> str:
        return matched_train_map.get(tkey, "")

    matched_df["matched_train_template"] = matched_df[extra_col_name].map(map_train)

    # Keep useful columns only (if present)
    keep_cols = [c for c in [templ_col, "normalized_template", extra_col_name,
                             "matched_train_template"] if c in matched_df.columns]
    matched_df[keep_cols].to_csv(out_path, index=False)

# Core coverage
def coverage_exact(train_series: pd.Series, test_series: pd.Series) -> Tuple[pd.Series, pd.Series, Dict[str,str]]:
    # key is the full normalized template
    key_fn = lambda x: x
    train_map = build_lookup(train_series, key_fn)
    test_keys = test_series.dropna().map(normalize_template)
    matches = test_keys.map(lambda k: k in train_map)
    return matches, test_keys, train_map

def coverage_first_n(train_series: pd.Series, test_series: pd.Series, n: int) -> Tuple[pd.Series, pd.Series, Dict[str,str]]:
    key_fn = lambda x: first_n_words(x, n)
    train_map = build_lookup(train_series, key_fn)
    test_keys = test_series.dropna().map(normalize_template).map(key_fn)
    matches = test_keys.map(lambda k: k in train_map)
    return matches, test_keys, train_map

def coverage_last_n(train_series: pd.Series, test_series: pd.Series, n: int) -> Tuple[pd.Series, pd.Series, Dict[str,str]]:
    key_fn = lambda x: last_n_words(x, n)
    train_map = build_lookup(train_series, key_fn)
    test_keys = test_series.dropna().map(normalize_template).map(key_fn)
    matches = test_keys.map(lambda k: k in train_map)
    return matches, test_keys, train_map

# ---------- Runner ----------
def run_coverage(train_csv: str, test_csv: str,
                 templ_col: str = "templated_question",
                 forward_levels: List[int] = (11, 10, 9, 8, 7, 6, 5),
                 backward_levels: List[int] = (5, 6, 7, 8, 9, 10, 11)) -> pd.DataFrame:

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    if templ_col not in train_df.columns or templ_col not in test_df.columns:
        raise ValueError(f"Both CSV files must contain '{templ_col}' column.")

    train_t = train_df[templ_col]
    test_t  = test_df[templ_col]

    rows = []

    # Full exact (normalized) match — also save only matched rows to CSV
    full_matches, full_keys, full_map = coverage_exact(train_t, test_t)
    matched = int(full_matches.sum())
    total = len(full_matches)
    pct = (matched / total * 100.0) if total > 0 else 0.0
    rows.append({"match_type": "full_exact", "n_words": None, "matched": matched, "total": total, "coverage_pct": round(pct, 2)})

    save_matches_filtered(
        test_df=test_df, templ_col=templ_col,
        key_series=full_keys, matches=full_matches,
        matched_train_map=full_map,
        extra_col_name="full_normalized",
        out_path="full_exact_matches.csv"
    )

    # Forward (first-N words)
    for n in forward_levels:
        f_matches, f_keys, f_map = coverage_first_n(train_t, test_t, n)
        matched = int(f_matches.sum())
        total = len(f_matches)
        pct = (matched / total * 100.0) if total > 0 else 0.0
        rows.append({"match_type": "forward_first_n", "n_words": n, "matched": matched, "total": total, "coverage_pct": round(pct, 2)})

        # Save ONLY for n in {10, 11}, matched rows only
        if n in (10, 11):
            save_matches_filtered(
                test_df=test_df, templ_col=templ_col,
                key_series=f_keys, matches=f_matches,
                matched_train_map=f_map,
                extra_col_name=f"first_{n}_words",
                out_path=f"forward_first_{n}_matches.csv"
            )

    # Backward (last-N words)
    for n in backward_levels:
        b_matches, b_keys, b_map = coverage_last_n(train_t, test_t, n)
        matched = int(b_matches.sum())
        total = len(b_matches)
        pct = (matched / total * 100.0) if total > 0 else 0.0
        rows.append({"match_type": "backward_last_n", "n_words": n, "matched": matched, "total": total, "coverage_pct": round(pct, 2)})

        # Save ONLY for n in {10, 11}, matched rows only
        if n in (10, 11):
            save_matches_filtered(
                test_df=test_df, templ_col=templ_col,
                key_series=b_keys, matches=b_matches,
                matched_train_map=b_map,
                extra_col_name=f"last_{n}_words",
                out_path=f"backward_last_{n}_matches.csv"
            )

    result_df = pd.DataFrame(rows)

    # print summary
    print("\n=== Coverage Summary ===")
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', 120):
        print(result_df.sort_values(by=["match_type", "n_words"],
                                    key=lambda s: s.map(lambda x: -1 if x is None else x)
                                    ).to_string(index=False))
    return result_df

if __name__ == "__main__":
    TRAIN = "mistral_train.csv"
    TEST  = "mistral_test.csv"
    run_coverage(TRAIN, TEST)