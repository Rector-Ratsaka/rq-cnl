import csv
import re

def extract_slots(template):
    # Find all slot tags like EC1, PC1, etc., in order of appearance
    return re.findall(r'\b(E?P?C?\d+)\b', template)

def escape_template(template):
    # Escape regex special characters and replace slots with capture groups
    pattern = re.escape(template)
    slots = extract_slots(template)
    for slot in slots:
        pattern = pattern.replace(re.escape(slot), f"(?P<{slot}>.+?)", 1)
    return pattern, slots

def map_slots(rq, template):
    pattern, slots = escape_template(template)
    match = re.match(pattern, rq)
    if not match:
        return None
    result = rq
    for slot in slots:
        value = match.group(slot)
        result = result.replace(value, f"[{value}] ({slot})", 1)
    return result

def process_csv(input_csv, output_txt):
    with open(input_csv, 'r', newline='', encoding='utf-8') as infile, open(output_txt, 'w', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        for row in reader:
            rq = row['research_question']
            template = row['templated_question']
            mapped = map_slots(rq, template)
            if mapped:
                outfile.write(mapped + "\n")
            else:
                outfile.write(f"Could not map: {rq}\n")

# Example usage
process_csv('cnl_output/mistral_mod.csv', 'mapped_output.txt')
