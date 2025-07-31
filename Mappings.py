
from typing import Tuple, Dict
from ChunkingLib import (
    nlp,
    mark_chunk,                   
    _prefix_end_index             
)

def _mark_chunk_with_mapping(
    cq: str,
    spans: list[tuple[int, int]],
    chunktype: str,
    offset: int,
    counter: int,
    mappings: Dict[str, str]
) -> Tuple[str, int]:
    """
    Same as mark_chunk() but also stores the text→marker mapping.
    """
    for (start, end) in spans:
        key = f"{chunktype}{counter}"
        mappings[key] = cq[start - offset:end - offset]
        cq = cq[:start - offset] + key + cq[end - offset:]
        offset += (end - start) - len(key)
    return cq, offset


# EC extraction with mapping 
def extract_EC_chunks_with_mapping(
    question: str,
    mappings: Dict[str, str] | None = None
) -> Tuple[str, Dict[str, str]]:
    """
    Return (templated_question, mappings_dict)

    This is the same algorithm that used to live in ChunkingLib, but moved
    here so the main library stays clean.
    """
    if mappings is None:
        mappings = {}

    cq          = question
    doc         = nlp(cq)
    prefix_end  = _prefix_end_index(cq)
    offset      = 0
    counter     = 1

    rejecting_ec = {
        "does", "do", "can", "could", "will", "would", "should", "shall",
        "may", "might", "what", "which", "when", "where", "who", "whom",
        "whose", "why", "how", "type", "types", "kinds", "kind",
        "category", "categories", "difference", "differences", "extent",
        "i", "we", "respect", "there", "this", "that", "these", "those",
        "not", "the main types", "the possible types", "the types",
        "the difference", "the differences", "the main categories",
        "is", "are", "was", "were", "have", "has", "had", "been",
        "being", "be", "how can", "what can", "when can", "where can",
        "why can", "which can"
    }

    def _clean_span(chunk):
        if (len(chunk) > 1 and
            (chunk[0].text.lower().startswith("wh") or
             chunk[0].text.lower() == "how")):
            chunk = chunk[1:]
        if len(chunk) and chunk[0].text.lower() in {
            "any", "some", "many", "well", "its"
        }:
            chunk = chunk[1:]
        return (chunk.start_char, chunk.end_char) if len(chunk) else None

    # “How + ADJ + VERB …” special-case
    if (len(doc) > 2 and doc[0].text.lower() == "how" and
            doc[1].pos_ == "ADJ" and doc[2].pos_ == "VERB"):
        start = doc[1].idx
        end   = start + len(doc[1])
        cq, offset = _mark_chunk_with_mapping(
            cq, [(start, end)], "EC", offset, counter, mappings
        )
        counter += 1

    # general noun-chunk pass
    for chunk in doc.noun_chunks:
        span = _clean_span(chunk)
        if span is None:
            continue
        start, end = span
        if prefix_end is not None and start < prefix_end:
            continue
        text = cq[start - offset:end - offset].strip()
        if not text or text.lower() in rejecting_ec:
            continue
        # skip leading single auxiliaries (Do/Is …)
        if (chunk.start == 0 and len(chunk) == 1 and
            chunk[0].pos_ in {"AUX", "VERB"}):
            continue

        cq, offset = _mark_chunk_with_mapping(
            cq, [(start, end)], "EC", offset, counter, mappings
        )
        counter += 1

    return cq, mappings


# PC extraction with mapping
def extract_PC_chunks_with_mapping(
    question_with_ec: str,
    mappings: Dict[str, str] | None = None
) -> Tuple[str, Dict[str, str]]:
    """
    Second pass: replace predicate chunks and extend the same `mappings` dict.
    """
    if mappings is None:
        mappings = {}

    cq          = question_with_ec
    doc         = nlp(cq)
    prefix_end  = _prefix_end_index(cq)
    offset      = 0
    counter     = 1

    rejecting_pc = {
        'is', 'are', 'was', 'were', 'do', 'does', 'did', 'have', 'had', 'has',
        'can', 'could', 'will', 'would', 'should', 'shall', 'may', 'might',
        'must', 'be', 'been', 'being', 'categorise', 'regarding',
        'is of', 'are of', 'are in', 'given',
        'is there', 'are there', 'was there', 'were there'
    }

    from ChunkingLib import get_PCs_as_spans  # local import to avoid cycle
    for begin, end, aux in get_PCs_as_spans(cq):
        if prefix_end is not None and begin < prefix_end:
            continue
        pc_text = cq[begin - offset:end - offset].strip()
        if not pc_text or pc_text.lower() in rejecting_pc:
            continue

        spans = [(begin, end)]
        if aux and aux.text.lower() not in rejecting_pc:
            spans.insert(0, (aux.idx, aux.idx + len(aux)))

        cq, offset = _mark_chunk_with_mapping(
            cq, spans, "PC", offset, counter, mappings
        )
        counter += 1

    return cq, mappings