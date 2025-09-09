import csv
import re

# Input files
train_file = "mistral_train.csv"
test_file = "mistral_test.csv"
output_file = "mistral_templates.csv"

# Regex to strip numbers from EC/PC terms
pattern = re.compile(r"(EC|PC)\d+")

def load_and_clean(filename):
    cleaned = []
    with open(filename, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            templated = row["templated_question"]
            # Remove chunk numbers
            templated = pattern.sub(r"\1", templated)
            cleaned.append(templated.strip())
    return cleaned

# Load both
train_templates = load_and_clean(train_file)
test_templates = load_and_clean(test_file)

# Pad shorter list with empty strings so lengths match
max_len = max(len(train_templates), len(test_templates))
train_templates += [""] * (max_len - len(train_templates))
test_templates += [""] * (max_len - len(test_templates))

# Write output
with open(output_file, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["train_templates", "test_templates"])
    for tr, te in zip(train_templates, test_templates):
        writer.writerow([tr, te])

print(f"✅ Done! Saved to {output_file}")
