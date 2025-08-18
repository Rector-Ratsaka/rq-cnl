import csv

# Input and output file paths
input_file = 'research_questions/mistral_rqs.csv'
output_file = 'research_questions/mistral_filtered_rqs.csv'

MAX_CHARS = 250  # maximum allowed length

with open(input_file, mode='r', encoding='utf-8') as infile, \
     open(output_file, mode='w', encoding='utf-8', newline='') as outfile:

    reader = csv.DictReader(infile)
    fieldnames = ['url', 'research_question']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        url = row['url']
        rq = row['research_question'].strip()

        # Keep only research questions with <= 250 characters
        if len(rq) <= MAX_CHARS:
            writer.writerow({'url': url, 'research_question': rq})
