import csv
import random

files_and_counts = [
    ("research_questions/rqs_llama_3.csv", 25),
    ("research_questions/rqs_mistral_1.csv", 25),
]
target_rqs = set(range(2, 500)) | set(range(500, 1500)) | set(range(1500, 2500)) | set(range(2500, 3500)) | set(range(3500, 4650))

collected = []

for filename, count in files_and_counts:
    with open(filename, newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        filtered = [row for idx, row in enumerate(reader, 2) if idx in target_rqs]
        sampled = random.sample(filtered, min(count, len(filtered)))
        collected.extend(sampled)

# Write to new CSV
with open('rqs_50/eval_50_rqs.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(collected)

