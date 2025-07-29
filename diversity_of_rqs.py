import csv
import re
from collections import Counter

# Load RQs
input_file = "research_questions/mistral_rqs.csv"      # Replace with your actual input filename
output_file = "mistral_patterns.csv"  # Output filename

# Define pattern matcher (captures first 4 words for structure)
def extract_pattern(rq):
    rq = rq.strip()
    rq = re.sub(r'[^\w\s]', '', rq)  # Remove punctuation
    words = rq.split()
    if not words:
        return None
    # Lowercase pattern of first 3–5 words
    for n in range(5, 2, -1):  # Try from 5 to 3 words
        if len(words) >= n:
            return " ".join(words[:n]).lower()
    return " ".join(words).lower()

# Read RQs and extract patterns
patterns = []
with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pattern = extract_pattern(row["research_question"])
        if pattern:
            patterns.append(pattern)

# Count and filter patterns with frequency >= 2
pattern_counts = Counter(patterns)
filtered = [(p, c) for p, c in pattern_counts.items() if c >= 2]

# Sort by frequency descending
filtered.sort(key=lambda x: -x[1])

# Save to CSV
with open(output_file, mode="w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["pattern", "frequency"])
    for pattern, count in filtered:
        writer.writerow([pattern, count])
