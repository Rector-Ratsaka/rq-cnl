import csv
import re

# Input and output file paths
input_file = 'research_questions/rqs_llama_3.csv'
output_file = 'research_questions/llama_sep_rqs.csv'

# Regex pattern for splitting compound questions
split_pattern = re.compile(r'\?\s*(and\s+(what|how|can|is))', re.IGNORECASE)

# Open input and output files
with open(input_file, mode='r', encoding='utf-8') as infile, \
     open(output_file, mode='w', encoding='utf-8', newline='') as outfile:

    reader = csv.DictReader(infile)
    fieldnames = ['url', 'research_question']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        url = row['url']
        rq = row['research_question'].strip()

        # Check if it is a compound question (2 question marks and "and what/how/can/is")
        if rq.count('?') >= 2 and re.search(split_pattern, rq):
            # Split and reconstruct individual questions
            parts = split_pattern.split(rq)
            # parts structure: [first part, 'and how', 'how', rest...]
            reconstructed = [parts[0] + '?']
            for i in range(1, len(parts), 3):
                clause = parts[i] + parts[i + 2]  # e.g. 'and how' + ' the rest...'
                reconstructed.append(clause.strip().capitalize() + '?')

            # Write each question as a new row
            for question in reconstructed:
                writer.writerow({'url': url, 'research_question': question})
        else:
            # Write the original row
            writer.writerow({'url': url, 'research_question': rq})