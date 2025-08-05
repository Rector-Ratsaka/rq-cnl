import re
import os
import csv
import spacy

nlp = spacy.load("en_core_web_sm")

def mark_chunk(cq: str,
               spans: list[tuple[int, int]],
               chunktype: str,
               offset: int,
               counter: int) -> tuple[str, int]:
    """
    Replace each (start,end) span with `chunktype{counter}`.
    `offset` tracks how much the CQ has shortened so later spans stay valid.
    """
    for (start, end) in spans:
        cq = (cq[:start - offset] +
              f"{chunktype}{counter}" +
              cq[end - offset:])
        offset += (end - start) - len(chunktype) - len(str(counter))
    return cq, offset

# “Do-not-touch” prefixes (loaded from CSV)
def _load_prefix_patterns() -> list[str]:
    """
    Read patterns/llama_rqs_patterns.csv and return the lower-cased “lead-in”
    strings that must never be chunk-replaced when they appear right at the
    beginning of the question.
    The CSV needs a header with a column named `pattern`.
    """
    patterns = []
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "patterns", "llama_rqs_patterns.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("pattern"):
                    patterns.append(row["pattern"].strip().lower())
    return patterns

_PREFIX_PATTERNS = _load_prefix_patterns()

def _prefix_end_index(question: str) -> int | None:
    """
    If `question` starts with any protected pattern (ignoring leading spaces),
    return the character index *immediately after* that pattern;
    otherwise return None.
    """
    leading_ws = len(question) - len(question.lstrip())
    q_lc = question.lower()
    for pat in _PREFIX_PATTERNS:
        if q_lc.startswith(pat, leading_ws):
            return leading_ws + len(pat)
    return None

# Extract entity chunks (ECs) from a question
def extract_EC_chunks(cq: str) -> str:
    """
    Replace every eligible entity chunk with EC1, EC2 …
    (Chunks that overlap the protected prefix are skipped.)
    """
    def _fix_span(chunk):
        # Trim WH-starters and quantifiers that should not enter the EC label.
        if (len(chunk) > 1 and
            (chunk[0].text.lower().startswith("wh") or chunk[0].text.lower() == "how")):
            chunk = chunk[1:]
        if len(chunk) and chunk[0].text.lower() in {"any", "some", "many", "well", "its"}:
            chunk = chunk[1:]
        return (chunk.start_char, chunk.end_char) if len(chunk) else None

    rejecting_ec = {
        "does", "do", "can", "could", "will", "would", "should", "shall", "may", "might",
        "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
        "type", "types", "kinds", "kind", "category", "categories", "difference",
        "differences", "extent", "i", "we", "respect", "there", "this", "that",
        "these", "those", "not", "the main types", "the possible types",
        "the types", "the difference", "the differences", "the main categories",
        "is", "are", "was", "were", "have", "has", "had", "been", "being", "be",
        "how can", "what can", "when can", "where can", "why can", "which can",
        'the performance', 'the use','the context','the accuracy','terms', 'it',
        'impact'
    }

    doc        = nlp(cq)
    prefix_end = _prefix_end_index(cq)

    counter = 1
    offset  = 0

    # Special case: “How + ADJ + VERB …”
    if (len(doc) > 2 and doc[0].text.lower() == "how" and
            doc[1].pos_ == "ADJ" and doc[2].pos_ == "VERB"):
        start = doc[1].idx
        end   = start + len(doc[1])
        cq, offset = mark_chunk(cq, [(start, end)], "EC", offset, counter)
        counter += 1

    # General noun-chunk pass
    for chunk in doc.noun_chunks:
        span = _fix_span(chunk)
        if span is None:
            continue
        start, end = span
        # Skip if inside protected prefix
        if prefix_end is not None and start < prefix_end:
            continue

        ec_text = cq[start - offset:end - offset].strip().lower()
        if ec_text in rejecting_ec or not ec_text:
            continue

        # Skip single AUX / VERB at beginning (“Do”, “Is” …)
        if (chunk.start == 0 and len(chunk) == 1 and
            chunk[0].pos_ in {"AUX", "VERB"}):
            continue

        # Handle the special “the thing” split
        if "the thing" in ec_text and (end - start) > len("the thing"):
            cq = (cq[:start - offset] + f"EC{counter}" +
                  " " + f"EC{counter + 1}" +
                  cq[end - offset:])
            offset += (end - start) - 7
            counter += 2
        else:
            cq, offset = mark_chunk(cq, [(start, end)], "EC", offset, counter)
            counter += 1

    # Final adjective/adverb at sentence end (“… are reliable?”)
    if len(doc) >= 2 and doc[-1].text == "?":
        penult = doc[-2]
        if ((penult.pos_ == "VERB" and len(doc) >= 3 and
             doc[-3].text.lower() in {"are", "is", "were", "was"}) or
            penult.pos_ in {"ADJ", "ADV"}):
            if penult.text.lower() not in rejecting_ec:
                start, end = penult.idx, penult.idx + len(penult)
                cq, offset = mark_chunk(cq, [(start, end)], "EC", offset, counter)

    return cq

def get_PCs_as_spans(cq: str):
    """
    Return a list of (begin, end, auxiliary_token_or_None) spans that look like
    predicate chunks.  Implementation unchanged, but trimmed comments here for
    brevity.
    """
    def _is_aux(token, main_ids):
        return (token.head.i in main_ids and token.dep_ == "aux" and
                token.i not in main_ids)

    def _span_from_group(group, doc):
        ids = [int(t.split("::")[0]) for t in group.split(",")]
        aux = next((t for t in doc if _is_aux(t, ids)), None)
        return (doc[ids[0]].idx,
                doc[ids[-1]].idx + len(doc[ids[-1]]),
                aux)

    def _reject_subspans(spans):
        keep = []
        for i, span in enumerate(spans):
            if not any(span[0] >= other[0] and span[1] <= other[1]
                       for j, other in enumerate(spans) if i != j):
                keep.append(span)
        return keep

    doc = nlp(cq)
    pos_text = ",".join(f"{i}::{t.pos_}" for i, t in enumerate(doc))
    regexes = [
        r"([0-9]+::(PART|VERB),?)*([0-9]+::VERB)",
        r"([0-9]+::(PART|VERB),?)+([0-9]+::AD(J|V),)+([0-9]+::ADP)",
        r"([0-9]+::(PART|VERB),?)+([0-9]+::ADP)",
    ]
    spans = []
    for r in regexes:
        spans.extend(_span_from_group(m.group(), doc)
                     for m in re.finditer(r, pos_text))
    return _reject_subspans(spans)

# Extract predicate chunks (PCs) from a question
def extract_PC_chunks(cq: str) -> str:
    rejecting_pc = {
        'is', 'are', 'was','can', 'were', 'do', 'does', 'did', 'have', 'had', 'has',
        'can', 'could', 'will', 'would', 'should', 'shall', 'may', 'might',
        'must', 'be', 'been', 'being', 'categorise', 'regarding', 'is of',
        'are of', 'are in', 'given', 'is there', 'are there', 'was there',
        'were there','using','compared to','compare to', 'affect', 'improving', 'improve',
        'based on','incorporating','impact', 'used', 'achieve', 
        'identifying', 'considering', 'achieve'
    }

    doc        = nlp(cq)
    prefix_end = _prefix_end_index(cq)

    offset  = 0
    counter = 1

    for begin, end, aux in get_PCs_as_spans(cq):
        if prefix_end is not None and begin < prefix_end:
            continue

        pc_text = cq[begin - offset:end - offset].strip().lower()
        if not pc_text or pc_text in rejecting_pc:
            continue

        spans = [(begin, end)]
        # include auxiliary if it exists and is not itself rejected
        if aux and aux.text.lower() not in rejecting_pc:
            spans.insert(0, (aux.idx, aux.idx + len(aux)))

        cq, offset = mark_chunk(cq, spans, "PC", offset, counter)
        counter += 1

    return cq

# Re-export mapping helpers 
try:
    from Mappings import (
        extract_EC_chunks_with_mapping,
        extract_PC_chunks_with_mapping,
    )
except Exception as _err: 
    pass