import csv
from collections import Counter

# Input and output file paths
input_csv_path = 'cnl_output/llama_templates_mappings.csv'
output_csv_path = 'llama_template_patterns.csv'

# Count frequencies
template_counter = Counter()
with open(input_csv_path, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        template = row['templated_question'].strip()
        if template:
            template_counter[template] += 1

# Save to CSV
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['templated_question', 'frequency'])
    for template, count in template_counter.most_common():
        writer.writerow([template, count])

print(f"Saved template frequencies to '{output_csv_path}'")
