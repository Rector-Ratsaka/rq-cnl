# analyse_mappings_by_marker.py
import pandas as pd
from pathlib import Path
from collections import Counter
from itertools import zip_longest

INPUT  = Path("./cnl_output/mistral_train.csv")
OUTPUT = Path("mistral_train_chunk_patterns.csv")

MARKERS = (
    [f"EC{i}" for i in range(1, 6)] +  # EC1 … EC5
    [f"PC{i}" for i in range(1, 3)]    # PC1 … PC2
)                                       

# load templated questions 
if not INPUT.exists():
    raise FileNotFoundError(f"Cannot locate {INPUT}")

df = pd.read_csv(INPUT)

# count frequencies for each marker column separately 
def count_per_column(col: str, frame: pd.DataFrame) -> list[tuple[str, int]]:
    """Return [(chunk, freq), …] with freq > 1, sorted desc."""
    c = Counter(
        x.strip().lower()
        for x in frame[col].dropna()
        if isinstance(x, str) and x.strip()
    )
    return sorted(
        [(chunk, freq) for chunk, freq in c.items() if freq > 1],
        key=lambda t: t[1],
        reverse=True
    )

counts_by_marker = {m: count_per_column(m, df) for m in MARKERS}

max_rows = max(len(v) for v in counts_by_marker.values())

records = {}
for marker in MARKERS:
    pairs = counts_by_marker[marker]                 

    # split into two separate lists
    chunks = [c for c, _ in pairs]
    freqs  = [f for _, f in pairs]

    # pad each list to max_rows
    chunks.extend([""] * (max_rows - len(chunks)))
    freqs.extend([""]  * (max_rows - len(freqs)))

    records[f"{marker} chunks"]    = chunks
    records[f"{marker} frequency"] = freqs

result = pd.DataFrame(records)

# save 
OUTPUT.parent.mkdir(exist_ok=True)
result.to_csv(OUTPUT, index=False, encoding="utf-8")
print(f"Saved per-marker frequencies to {OUTPUT}")