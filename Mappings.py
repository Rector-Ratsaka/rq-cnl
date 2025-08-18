from typing import Tuple, Dict
from ChunkingLib import (
    nlp,
    mark_chunk,                   
    _prefix_end_index,
    _is_question_starter_token  # Import the new function
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

    # "How + ADJ + VERB …" special-case
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
        
        # Skip if this chunk is within the protected prefix
        if prefix_end is not None and start < prefix_end:
            continue
            
        text = cq[start - offset:end - offset].strip()
        if not text or text.lower() in rejecting_ec:
            continue
            
        # Check if any token in this chunk is a question starter that should be preserved
        chunk_tokens = [token for token in doc if token.idx >= start and token.idx < end]
        if any(_is_question_starter_token(token, doc) for token in chunk_tokens):
            continue
            
        # skip leading single auxiliaries (Do/Is …)
        if (chunk.start == 0 and len(chunk) == 1 and
            chunk[0].pos_ in {"AUX", "VERB"} and
            chunk[0].text.lower() in {'do', 'does', 'can', 'could', 'will', 'would', 'should', 'is', 'are', 'was', 'were'}):
            continue

        # Handle "the thing" special case
        if "the thing" in text and end - start > len("the thing"):
            # Split into two ECs
            mappings[f"EC{counter}"] = "the"
            mappings[f"EC{counter + 1}"] = "thing"
            cq = cq[:start - offset] + f"EC{counter}" + f" EC{counter + 1}" + cq[end - offset:]
            counter += 2
            offset += (end - start) - len(f"EC{counter-2} EC{counter-1}")
        else:
            cq, offset = _mark_chunk_with_mapping(
                cq, [(start, end)], "EC", offset, counter, mappings
            )
            counter += 1

    # Handle end-of-sentence adjectives/adverbs
    try:
        if len(doc) >= 2:
            if ((doc[-2].pos_ == 'VERB' and len(doc) >= 3 and 
                 doc[-3].text in ['are', 'is', 'were', 'was'] and doc[-1].text == '?') or 
                (doc[-2].pos_ in ['ADJ', 'ADV'] and doc[-1].text == '?')):
                if (doc[-2].text.lower() not in rejecting_ec and
                    not _is_question_starter_token(doc[-2], doc)):
                    start = doc[-2].idx
                    end = start + len(doc[-2])
                    cq, offset = _mark_chunk_with_mapping(
                        cq, [(start, end)], "EC", offset, counter, mappings
                    )
                    counter += 1
    except Exception as e:
        print('Error processing end-of-sentence. Doc = ', doc, 'Error:', e)

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
        'is', 'are', 'was','can', 'were', 'do', 'does', 'did', 'have', 'had', 'has',
        'can', 'could', 'will', 'would', 'should', 'shall', 'may', 'might',
        'must', 'be', 'been', 'being', 'categorise', 'regarding', 'is of',
        'are of', 'are in', 'given', 'is there', 'are there', 'was there',
        'were there','using','compared to', 'affect', 'improving', 'improve',
        'based on','incorporating','impact', 'used', 'achieve', 
        'identifying', 'considering', 'achieve', 'compare to'
    }

    from ChunkingLib import get_PCs_as_spans  # local import to avoid cycle
    for begin, end, aux in get_PCs_as_spans(cq):
        # Skip if this PC chunk is within the protected prefix
        if prefix_end is not None and begin < prefix_end:
            continue
            
        pc_text = cq[begin - offset:end - offset].strip()
        if not pc_text or pc_text.lower() in rejecting_pc:
            continue

        # Check if any token in this PC chunk is a question starter
        pc_tokens = [token for token in doc if token.idx >= begin and token.idx < end]
        if any(_is_question_starter_token(token, doc) for token in pc_tokens):
            continue

        spans = [(begin, end)]
        
        # Handle auxiliary verbs, but check if they're question starters
        if aux and aux.text.lower() not in rejecting_pc:
            if not _is_question_starter_token(aux, doc):
                spans.insert(0, (aux.idx, aux.idx + len(aux)))

        cq, offset = _mark_chunk_with_mapping(
            cq, spans, "PC", offset, counter, mappings
        )
        counter += 1

    return cq, mappings