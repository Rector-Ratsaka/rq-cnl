import re
import spacy
nlp = spacy.load("en_core_web_sm")

def mark_chunk(cq, spans, chunktype, offset, counter):
    for (start, end) in spans:  # for each span of EC/PC candidate
        cq = cq[:start - offset] + chunktype + str(counter) + \
             cq[end - offset:]  # substitute that candidate with EC/PC marker
        offset += (end - start) - len(chunktype) - len(str(counter)) # Correct offset calculation
    return cq, offset


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

    doc = nlp(cq) # Assuming nlp is defined globally

    #  things classified as ECs which shouldn't be interpreted that way
    # FIXED: Added more auxiliary verbs and question words, made sure they're lowercase
    rejecting_ec = ["does", "do", "can", "could", "will", "would", "should", "shall", "may", "might",
                    "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
                    "type", "types", "kinds", "kind", "category", "categories", "difference",
                    "differences", "extent", "i", "we", "respect", "there", "this", "that", "these", "those",
                    "not", "the main types", "the possible types", "the types",
                    "the difference", "the differences", "the main categories",
                    "is", "are", "was", "were", "have", "has", "had", "been", "being", "be",
                    "how can", "what can", "when can", "where can", "why can", "which can"]
    
    # FIXED: Define question starter patterns to reject entirely
    question_starter_patterns = [
        ['how', 'can'], ['how', 'do'], ['how', 'does'], ['how', 'will'], ['how', 'would'],
        ['how', 'should'], ['how', 'could'], ['how', 'might'], ['how', 'may'],
        ['what', 'can'], ['what', 'do'], ['what', 'does'], ['what', 'will'], ['what', 'would'],
        ['when', 'can'], ['when', 'do'], ['when', 'does'], ['when', 'will'], ['when', 'would'],
        ['where', 'can'], ['where', 'do'], ['where', 'does'], ['where', 'will'], ['where', 'would'],
        ['why', 'can'], ['why', 'do'], ['why', 'does'], ['why', 'will'], ['why', 'would'],
        ['which', 'can'], ['which', 'do'], ['which', 'does'], ['which', 'will'], ['which', 'would']
    ]

    counter = 1  # counter indicating current chunk id (EC1, EC2, ...)
    offset = 0     # if we replace for example "Weka" with 'EC1' then the new CQ will be shorter by one char the offset remembers by how much we have shortened the current CQ with EC markers, so the new substitutions can be applied in correct places.

    # we decided to treat qualities defined as adjectives in: How + Quality(adjective) + Verb as EC
    if (len(doc) > 2 and  # FIXED: Added length check
            doc[0].text.lower() == 'how' and
            doc[1].pos_ == 'ADJ' and
            doc[2].pos_ == 'VERB'):

        start = doc[1].idx  # mark where quality starts
        end = start + len(doc[1])   # mark where quality ends

        cq, offset = mark_chunk(cq, [(start, end)], "EC", offset, counter)  # substitute quality with EC identifier
        counter += 1  # the next EC chunk should have a new, bigger identifier, for example EC2

    for chunk in doc.noun_chunks:  # for each EC chunk candidate detected
        span_result = _get_EC_span_reject_wh_starters(chunk)  # check where chunk begins and ends
        
        # FIXED: Check if span_result is None (empty chunk after filtering)
        if span_result is None:
            continue
            
        (start, end) = span_result

        ec = cq[start - offset:end - offset]  # extract text of potential EC, apply offsets

        # FIXED: More thorough rejection check
        if (ec.lower().strip() in rejecting_ec or  # Check lowercase and stripped version
            ec.strip() == "" or  # Skip empty chunks
            len(ec.strip()) == 0):  # Skip whitespace-only chunks
            continue

        # FIXED: Check if this EC chunk is part of a question starter pattern
        # Find the token position of this EC chunk
        ec_start_token_idx = None
        for i, token in enumerate(doc):
            if token.idx >= start and token.idx < end:
                ec_start_token_idx = i
                break
        
        # Skip if this is part of a question starter pattern at the beginning
        if ec_start_token_idx is not None and ec_start_token_idx <= 2:
            skip_chunk = False
            for pattern in question_starter_patterns:
                if (len(doc) > len(pattern) and 
                    ec_start_token_idx < len(pattern) and
                    all(doc[j].text.lower() == pattern[j] for j in range(len(pattern)) if j < len(doc))):
                    skip_chunk = True
                    break
            if skip_chunk:
                continue

        # FIXED: Additional check for single auxiliary verbs at start of question
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

    try:
        if len(doc) >= 2:  # FIXED: Added length check
            if (doc[-2].pos_ == 'VERB' and len(doc) >= 3 and doc[-3].text in ['are', 'is', 'were', 'was'] and doc[-1].text == '?') or (doc[-2].pos_ in ['ADJ', 'ADV'] and doc[-1].text == '?'):
                # if CQ ends with are/is/were/was + VERB + ? or the last token is ADJective or ADVerb, treat the
                # verb / adverb / adjective as EC
                # Which animals are endangered -> endangered is EC
                # Which animals are quick -> quick is EC
                if doc[-2].text.lower() not in rejecting_ec:
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
        if (token.head.i in chunk_token_ids and  # if dep-tree current token's parent (head) is somewhere inside the main part of PC represented as chunk_token_ids (sequence of numeric token identifiers)
                token.dep_ == 'aux' and  # if the dep-tree label on the edge between some word from main part of PC and current token is AUX (auxiliary)
                token.i not in chunk_token_ids):  # if token is outside of detected main part of PC
            return True  # yep, it's auxiliary
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

    doc = nlp(cq) # Assuming nlp is defined globally

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

    spans = []  # list of beginnings and endings of each chunk
    for regex in regexes:  # try to extract chunks with regexes
        for m in re.finditer(regex, pos_text):
            spans.append(_get_span(m.group(), doc))  # get chunk begin and end if matched
    spans = _reject_subspans(spans)  # reject sub-spans
    return spans


def extract_PC_chunks(cq):
    # FIXED: Enhanced rejecting_pc list with more auxiliary verbs and modal verbs
    rejecting_pc = ['is', 'are', 'was', 'were', 'do', 'does', 'did', 
                    'have', 'had', 'has', 'can', 'could', 'will', 'would', 
                    'should', 'shall', 'may', 'might', 'must', 'be', 'been', 'being',
                    'categorise', 'regarding', 'is of', 'are of', 'are in', 
                    'given', 'is there', 'are there', 'was there', 'were there']

    # FIXED: Add specific rejection patterns for question starters
    doc = nlp(cq)
    question_starter_patterns = [
        ['how', 'can'], ['how', 'do'], ['how', 'does'], ['how', 'will'], ['how', 'would'],
        ['how', 'should'], ['how', 'could'], ['how', 'might'], ['how', 'may'],
        ['what', 'can'], ['what', 'do'], ['what', 'does'], ['what', 'will'], ['what', 'would'],
        ['when', 'can'], ['when', 'do'], ['when', 'does'], ['when', 'will'], ['when', 'would'],
        ['where', 'can'], ['where', 'do'], ['where', 'does'], ['where', 'will'], ['where', 'would'],
        ['why', 'can'], ['why', 'do'], ['why', 'does'], ['why', 'will'], ['why', 'would'],
        ['which', 'can'], ['which', 'do'], ['which', 'does'], ['which', 'will'], ['which', 'would']
    ]

    offset = 0
    counter = 1

    for begin, end, aux in get_PCs_as_spans(cq):
        pc_text = cq[begin - offset:end - offset]
        
        # FIXED: More thorough rejection check for PC chunks
        if (pc_text.lower().strip() in rejecting_pc or
            pc_text.strip() == "" or
            len(pc_text.strip()) == 0):
            continue

        # FIXED: Check if this PC chunk is part of a question starter pattern
        # Find the token position of this PC chunk
        pc_start_token_idx = None
        for i, token in enumerate(doc):
            if token.idx >= begin and token.idx < end:
                pc_start_token_idx = i
                break
        
        # Skip if this is part of a question starter pattern at the beginning
        if pc_start_token_idx is not None and pc_start_token_idx <= 2:
            skip_chunk = False
            for pattern in question_starter_patterns:
                if (len(doc) > len(pattern) and 
                    pc_start_token_idx < len(pattern) and
                    all(doc[j].text.lower() == pattern[j] for j in range(len(pattern)) if j < len(doc))):
                    skip_chunk = True
                    break
            if skip_chunk:
                continue

        spans = [(begin, end)]

        # FIXED: Also check if auxiliary should be rejected
        if aux and aux.text.lower() not in rejecting_pc:
            # Additional check: don't include aux if it's part of question starter
            aux_skip = False
            if aux.i <= 2:  # auxiliary is in first few tokens
                for pattern in question_starter_patterns:
                    if (len(doc) > len(pattern) and
                        aux.i < len(pattern) and
                        all(doc[j].text.lower() == pattern[j] for j in range(len(pattern)) if j < len(doc))):
                        aux_skip = True
                        break
            
            if not aux_skip:
                spans.insert(0, (aux.idx, aux.idx + len(aux)))

        cq, offset = mark_chunk(cq, spans, "PC", offset, counter)
        counter += 1

    return cq