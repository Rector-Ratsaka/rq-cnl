# Analyze the quality of generated templates using corpus BLEU score.
# RTSREC001 - Rector Ratsaka

import csv
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Input file
input_file = "mistral_templates.csv"

# Load templates
train_templates = []
test_templates = []

with open(input_file, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tr = row.get("train_templates", "").strip()
        te = row.get("test_templates", "").strip()
        if tr:
            train_templates.append(tr.split())
        if te:
            test_templates.append(te.split())

# BLEU smoothing
smooth = SmoothingFunction().method1


# Interpret all train templates as references for every test template
if test_templates and train_templates:
    references_per_hyp = [train_templates] * len(test_templates)  # multi-reference
    corpus_score = corpus_bleu(
        references_per_hyp,
        test_templates,
        smoothing_function=smooth
    )
else:
    corpus_score = 0.0

print(f" Corpus BLEU Score: {corpus_score:.4f}")
