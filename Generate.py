# Generate CNL templates and structured EC/PC mappings from research questions (RQs)
# RTSREC001 - Rector Ratsaka

import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from ChunkingLib import extract_EC_chunks, extract_PC_chunks
from ChunkingLib import (
    extract_EC_chunks_with_mapping,  
    extract_PC_chunks_with_mapping,
)


@dataclass
class CNLTemplateGenerator:
    """Generate CNL templates *and* structured EC/PC mappings."""
    def extract_template(self, question: str):
        # fast path (no mapping)
        return extract_PC_chunks(extract_EC_chunks(question))

    def extract_template_with_mapping(self, question: str):
        """
        Returns (templated_question, mapping_dict)
        where mapping_dict has keys EC1 … ECn, PC1 … PCm.
        """
        cq_with_ec, map_dict = extract_EC_chunks_with_mapping(question)
        templated, map_dict  = extract_PC_chunks_with_mapping(cq_with_ec, map_dict)
        return templated, map_dict


# Columns for final output CSV
OUTPUT_COLS = [
    "research_question", "templated_question",
    "EC1", "EC2", "EC3", "EC4", "EC5",
    "PC1", "PC2"
]


def create_output_directory() -> Path:
    out_dir = Path.cwd() / "cnl_output"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def process_csv_file(file_path: Path) -> None:
    print(f"\nProcessing: {file_path} ***")
    df = pd.read_csv(file_path)

    if "research_question" not in df.columns:
        print("missing ‘research_question’ column")
        return

    df = df[["research_question"]].dropna().drop_duplicates()

    gen = CNLTemplateGenerator()

    # Apply extraction with mapping
    results = df["research_question"].apply(gen.extract_template_with_mapping)

    df["templated_question"] = results.apply(lambda x: x[0])
    df["mapping_dict"]       = results.apply(lambda x: x[1])

    df = (df.assign(template_len=df["templated_question"].str.len())
             .sort_values("template_len", ascending=True)
             .drop(columns="template_len"))

    # Expand EC / PC columns (pad with “” if missing)
    for i in range(1, 6):
        df[f"EC{i}"] = df["mapping_dict"].apply(lambda d: d.get(f"EC{i}", ""))
    for i in range(1, 3):
        df[f"PC{i}"] = df["mapping_dict"].apply(lambda d: d.get(f"PC{i}", ""))

    df = df[OUTPUT_COLS]

    out_path = create_output_directory() / "mistral_test.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"saved to {out_path}")


# CLI entry-point
def main():
    src = Path("research_questions/mistral_rqs_20test.csv")
    if not src.exists():
        print(f"File not found: {src}")
        return
    process_csv_file(src)


if __name__ == "__main__":
    main()