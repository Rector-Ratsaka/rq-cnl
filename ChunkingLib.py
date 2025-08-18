import re
import spacy
import csv, os 
nlp = spacy.load("en_core_web_sm")

def mark_chunk(cq, spans, chunktype, offset, counter):
    for (start, end) in spans:  # for each span of EC/PC candidate
        cq = cq[:start - offset] + chunktype + str(counter) + \
             cq[end - offset:]  # substitute that candidate with EC/PC marker
        offset += (end - start) - len(chunktype) - len(str(counter)) 
    return cq, offset

def _load_prefix_patterns() -> list[str]:
    """
    Load the list of fixed question-starter patterns (1-gram … n-gram) that
    must *never* be chunk-replaced when they appear right at the beginning of
    the research-question string.
    """
    patterns = []
    base_dir  = os.path.dirname(__file__)
    csv_path  = os.path.join(base_dir, "patterns/mistral_rqs_patterns.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("pattern"):
                    patterns.append(row["pattern"].strip().lower())
    
    # Add common question starters if not already in patterns
    default_patterns = [
        "can", "do", "does", "did", "will", "would", "could", "should", "shall", "may", "might",
        "is", "are", "was", "were", "has", "have", "had",
        "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
        "can a", "can the", "do the", "does the", "did the", 
        "will a", "will the", "would a", "would the",
        "could a", "could the", "should a", "should the",
        "how can", "how do", "how does", "how will", "how would", "how could", "how should",
        "what can", "what do", "what does", "what will", "what would",
        "when can", "when do", "when does", "when will", "when would",
        "where can", "where do", "where does", "where will", "where would",
        "why can", "why do", "why does", "why will", "why would",
        "which can", "which do", "which does", "which will", "which would"
    ]
    
    for pattern in default_patterns:
        if pattern not in patterns:
            patterns.append(pattern)
    
    return patterns

_PREFIX_PATTERNS = _load_prefix_patterns()

def _prefix_end_index(question: str) -> int | None:
    """Return the character index *after* a matched prefix or None."""
    q_lc = question.strip().lower()
    for pat in _PREFIX_PATTERNS:
        if q_lc.startswith(pat + " "):  # Ensure we match complete words with space after
            return len(pat)
        elif q_lc.startswith(pat) and len(q_lc) == len(pat):  # Exact match for single word
            return len(pat)
    return None

def _is_question_starter_token(token, doc):
    """Check if a token is part of a question starter that should be preserved"""
    # Get position of token in sentence
    token_pos = token.i
    
    # Check if this token starts a common question pattern
    question_starters = ["can", "do", "does", "did", "will", "would", "could", "should", 
                        "is", "are", "was", "were", "has", "have", "had",
                        "what", "which", "when", "where", "who", "whom", "whose", "why", "how"]
    
    if token_pos == 0 and token.text.lower() in question_starters:
        return True
    
    # Check for two-word question starters like "how can", "what do", etc.
    if token_pos <= 1:
        two_word_patterns = [
            ["how", "can"], ["how", "do"], ["how", "does"], ["how", "will"], ["how", "would"],
            ["how", "should"], ["how", "could"], ["how", "might"], ["how", "may"],
            ["what", "can"], ["what", "do"], ["what", "does"], ["what", "will"], ["what", "would"],
            ["when", "can"], ["when", "do"], ["when", "does"], ["when", "will"], ["when", "would"],
            ["where", "can"], ["where", "do"], ["where", "does"], ["where", "will"], ["where", "would"],
            ["why", "can"], ["why", "do"], ["why", "does"], ["why", "will"], ["why", "would"],
            ["which", "can"], ["which", "do"], ["which", "does"], ["which", "will"], ["which", "would"]
        ]
        
        for pattern in two_word_patterns:
            if (len(doc) > 1 and token_pos < len(pattern) and
                token_pos < len(doc) and doc[token_pos].text.lower() == pattern[token_pos]):
                # Check if the full pattern matches
                pattern_matches = True
                for i, word in enumerate(pattern):
                    if i >= len(doc) or doc[i].text.lower() != word:
                        pattern_matches = False
                        break
                if pattern_matches:
                    return True
    
    return False

def extract_EC_chunks(cq):
    """
        Find EC chunks and replace their occurrences with EC tags
    """

    def _get_EC_span_reject_wh_starters(chunk):
        """
            By default, SpaCy treats question words (wh- pronouns starting questions: where, what,...)
            as nouns, so often if questions starts with wh- pronoun + noun (like: What software is the best?)
            the whole "what software" is interpreted as EC chunk - this function tries to fix that issue by
            omitting wh- word if EC candidate consists of multiple tokens, and first token in question is wh- word.
            The same issue occurs for "How" starter.
            Moreover, chunks extracted with SpaCy enclose words like 'any', 'some' which are important for us, so
            they shouldn't be substituted with 'EC' marker. Thus we remove such words if they prepend the EC.
            The result is returned as the span - the position of beginning of the fixed EC and position of end.
        """
        if (len(chunk) > 1 and
                (chunk[0].text.lower().startswith("wh")) or
                (chunk[0].text.lower() == 'how')):
            chunk = chunk[1:]

        if len(chunk) > 0 and chunk[0].text.lower() in ['any', 'some', 'many', 'well', 'its']:
            chunk = chunk[1:]

        return (chunk.start_char, chunk.end_char) if len(chunk) > 0 else None

    doc = nlp(cq)
    prefix_end = _prefix_end_index(cq)

    #  things classified as ECs which shouldn't be interpreted that way
    rejecting_ec = ["does", "do", "can", "could", "will", "would", "should", "shall", "may", "might",
                    "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
                    "type", "types", "kinds", "kind", "category", "categories", "difference",
                    "differences", "extent", "i", "we", "respect", "there", "this", "that", "these", "those",
                    "not", "the main types", "the possible types", "the types",
                    "the difference", "the differences", "the main categories",
                    "is", "are", "was", "were", "have", "has", "had", "been", "being", "be",
                    "how can", "what can", "when can", "where can", "why can", "which can"]

    counter = 1
    offset = 0

    # we decided to treat qualities defined as adjectives in: How + Quality(adjective) + Verb as EC
    if (len(doc) > 2 and
            doc[0].text.lower() == 'how' and
            doc[1].pos_ == 'ADJ' and
            doc[2].pos_ == 'VERB'):

        start = doc[1].idx
        end = start + len(doc[1])

        cq, offset = mark_chunk(cq, [(start, end)], "EC", offset, counter)
        counter += 1

    for chunk in doc.noun_chunks:
        span_result = _get_EC_span_reject_wh_starters(chunk)
        
        if span_result is None:
            continue
            
        (start, end) = span_result
        
        # Skip if this chunk is within the protected prefix
        if prefix_end is not None and start < prefix_end:
            continue

        ec = cq[start - offset:end - offset]

        if (ec.lower().strip() in rejecting_ec or
            ec.strip() == "" or
            len(ec.strip()) == 0):
            continue

        # Check if any token in this chunk is a question starter that should be preserved
        chunk_tokens = [token for token in doc if token.idx >= start and token.idx < end]
        if any(_is_question_starter_token(token, doc) for token in chunk_tokens):
            continue

        # Additional check for single auxiliary verbs at start of question
        if (chunk.start == 0 and len(chunk) == 1 and 
            chunk[0].pos_ in ['AUX', 'VERB'] and 
            chunk[0].text.lower() in ['do', 'does', 'can', 'could', 'will', 'would', 'should', 'is', 'are', 'was', 'were']):
            continue

        if "the thing" in ec and end - start > len("the thing"):
            cq = cq[:start - offset] + "EC" + str(counter) + \
                 " EC" + str(counter + 1) + cq[end - offset:]
            counter += 2
            offset += (end - start) - 7
        else:
            cq, offset = mark_chunk(cq, [(start, end)], "EC", offset, counter)
            counter += 1

    # Handle end-of-sentence adjectives/adverbs
    try:
        if len(doc) >= 2:
            if (doc[-2].pos_ == 'VERB' and len(doc) >= 3 and doc[-3].text in ['are', 'is', 'were', 'was'] and doc[-1].text == '?') or (doc[-2].pos_ in ['ADJ', 'ADV'] and doc[-1].text == '?'):
                if (doc[-2].text.lower() not in rejecting_ec and
                    not _is_question_starter_token(doc[-2], doc)):
                    start = doc[-2].idx
                    end = start + len(doc[-2])

                    cq, offset = mark_chunk(
                        cq, [(start, end)], "EC", offset, counter)
                    counter += 1
    except Exception as e:
        print('Error processing. Doc = ', doc, 'Error:', e)

    return cq

def get_PCs_as_spans(cq):
    def _is_auxilary(token, chunk_token_ids):
        """
            Check if given token is an auxiliary verb of detected PC.
            The auxiliary verb can be in a different place than the main part
            of the PC, so pos-tag-sequence based rules don't work here.
            For example in "What system does Weka require?" - the main part
            of PC is the word 'required'. The auxiliary verb 'does' is separated
            from the main part by 'Weka' noun. Thus dependency tree is used
            to identify auxiliaries.
        """
        if (token.head.i in chunk_token_ids and
                token.dep_ == 'aux' and
                token.i not in chunk_token_ids):
            return True
        else:
            return False

    def _get_span(group, doc):
        id_tags = group.split(",")
        ids = [int(id_tag.split("::")[0]) for id_tag in id_tags]
        aux = None
        for token in doc:
            if _is_auxilary(token, ids):
                aux = token

        return (doc[ids[0]].idx, doc[ids[-1]].idx + len(doc[ids[-1]]),
                aux)

    def _reject_subspans(spans):
        """
            Given list of (chunk begin index, chunk end index) spans,
            return only those spans that aren't sub-spans of any other span.
            For instance form list [(1,10), (2,5)], the second span
            will be rejected because it is a subspan of the first one.
        """
        filtered = []
        for i, span in enumerate(spans):
            subspan = False
            for j, other in enumerate(spans):
                if i == j:
                    continue

                if span[0] >= other[0] and span[1] <= other[1]:
                    subspan = True
                    break
            if subspan is False:
                filtered.append(span)
        return filtered

    doc = nlp(cq)

    """
        Transform CQ into a form of POS-tags with token sequence identifier.
        Each token is described with "{ID}::{POS_TAG}".
        Tokens are separated with ","
        Having that form, we can extract longest sequences of expected pos-tags
        using regexes. The extracted parts can be explored to collect identifiers
        of tokens, so we know where they are located in text.
        Ex: "Kate owns a cat" should be translated into: "1::NOUN,2::VERB,3::DET,4::NOUN"
    """
    pos_text = ",".join(
        ["{id}::{pos}".format(id=id, pos=t.pos_) for id, t in enumerate(doc)])

    regexes = [  # rules describing PCs
        r"([0-9]+::(PART|VERB),?)*([0-9]+::VERB)",
        r"([0-9]+::(PART|VERB),?)+([0-9]+::AD(J|V),)+([0-9]+::ADP)",
        r"([0-9]+::(PART|VERB),?)+([0-9]+::ADP)",
    ]

    spans = []
    for regex in regexes:
        for m in re.finditer(regex, pos_text):
            spans.append(_get_span(m.group(), doc))
    spans = _reject_subspans(spans)
    return spans

def extract_PC_chunks(cq):
    rejecting_pc = ['is', 'are', 'was', 'were', 'do', 'does', 'did', 
                    'have', 'had', 'has', 'can', 'could', 'will', 'would', 
                    'should', 'shall', 'may', 'might', 'must', 'be', 'been', 'being',
                    'categorise', 'regarding', 'is of', 'are of', 'are in', 
                    'given', 'is there', 'are there', 'was there', 'were there']

    doc = nlp(cq)
    prefix_end = _prefix_end_index(cq)

    offset = 0
    counter = 1

    for begin, end, aux in get_PCs_as_spans(cq):
        # Skip if this PC chunk is within the protected prefix
        if prefix_end is not None and begin < prefix_end:
            continue
            
        pc_text = cq[begin - offset:end - offset]
        
        if (pc_text.lower().strip() in rejecting_pc or
            pc_text.strip() == "" or
            len(pc_text.strip()) == 0):
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