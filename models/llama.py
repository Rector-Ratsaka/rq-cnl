# Script to generate/extract research questions from abstracts using LLaMA 3.2
# Usage: python3 llama.py <input_file> <prompts_file> <output_file>
# RTSREC001 - Rector Ratsaka

import argparse
import json
import csv
import re
import torch
from transformers import AutoTokenizer, pipeline

# command line args
parser = argparse.ArgumentParser(description="Generate/Extract research questions from abstracts using LLaMA 3.2.")
parser.add_argument("input_file", type=str, help="Path to the input abstracts JSON file.")
parser.add_argument("prompts_file", type=str, help="Path to the prompts text file.")
parser.add_argument("output_file", type=str,  help="Path to the output CSV file.")
args = parser.parse_args()

# Assign command line arguments to variables
input_file = args.input_file
prompts_file = args.prompts_file
output_file = args.output_file

# Model name
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Target ID range
target_ids = set(range(1, 2501))

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id
)

# Load abstracts
with open(input_file, "r", encoding="utf-8") as f:
    abstracts = json.load(f)

# Load prompt template
with open(prompts_file, "r", encoding="utf-8") as f:
    prompt_template = f.read().strip()

# Process and collect research questions
rqs_dataset = []
for entry in abstracts:
    if entry["id"] not in target_ids or not entry.get("abstract", "").strip():
        continue

    abstract = entry["abstract"].strip()
    url = entry.get("url", "")

    # format the prompt
    prompt = prompt_template.format(abstract=abstract)

    # Generate output
    output = pipe(prompt, max_new_tokens=256)[0]["generated_text"]
    response = output[len(prompt):].strip()

    for line in response.split("\n"):
        # Clean and validate each line
        clean_line = re.sub(r'^\s*["\']?\d+[\.\)]\s*', '', line).strip(' "\'\n')
        if clean_line:
            rqs_dataset.append({
                "url": url,
                "research_question": clean_line
            })

# Save to CSV
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["url", "research_question"])
    writer.writeheader()
    writer.writerows(rqs_dataset)

print(f"Total RQs extracted: {len(rqs_dataset)}")