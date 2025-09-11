import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Input file
input_file = "mistral_templates.csv"

# Load templates
train_templates = []
test_templates = []

with open(input_file, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["train_templates"].strip():
            train_templates.append(row["train_templates"].split())
        if row["test_templates"].strip():
            test_templates.append(row["test_templates"].split())

# BLEU smoothing
smooth = SmoothingFunction().method1

# For each test template, find the highest BLEU against any train template
max_scores = []
for idx, te in enumerate(test_templates):
    scores = [sentence_bleu([tr], te, smoothing_function=smooth) for tr in train_templates]
    best_score = max(scores)
    max_scores.append(best_score)

# Compute average of the max scores
average_bleu = sum(max_scores) / len(max_scores) if max_scores else 0.0

print(f"Sentence BLEU Score: {average_bleu:.4f}")
