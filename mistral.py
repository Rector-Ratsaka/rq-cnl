import argparse
import json
import csv
import re
import torch
from transformers import AutoTokenizer, pipeline

# --- Command-line Arguments ---
parser = argparse.ArgumentParser(description="Extract or generate research questions from abstracts using Mistral 7B.")
parser.add_argument("input_file", type=str, help="Path to the input JSON file with abstracts.")
parser.add_argument("prompts_file", type=str, help="Path to the prompts text file.")
parser.add_argument("output_file", type=str, help="Path to the output CSV file.")
args = parser.parse_args()

input_file = args.input_file
prompts_file = args.prompts_file
output_file = args.output_file

# --- Model Setup ---
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id
)

# --- Load Abstracts ---
with open(input_file, "r", encoding="utf-8") as f:
    abstracts = json.load(f)

# --- Load Prompt Template ---
with open(prompts_file, "r", encoding="utf-8") as f:
    prompt_template = f.read().strip()

# --- Target Abstract IDs ---
target_ids = set(range(1, 2500))

# --- Generate Research Questions ---
rqs_dataset = []
for entry in abstracts:
    entry_id = entry.get("id")
    if entry_id not in target_ids:
        continue

    abstract = entry.get("abstract", "").strip()
    if not abstract:
        continue

    # prompt with abstract
    prompt = prompt_template.format(abstract=abstract)

    try:
        output = generator(prompt, max_new_tokens=256)[0]["generated_text"]
        response = output[len(prompt):].strip()

        for line in response.split("\n"):
            clean_line = re.sub(r'^\s*["\']?\d+[\.\)]?\s*', '', line).strip(' "\'\n')
            if clean_line:
                rqs_dataset.append({"research_question": clean_line})

    except Exception as e:
        print(f"Error processing entry ID {entry_id}: {e}")

# --- Write to CSV ---
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["research_question"])
    writer.writeheader()
    writer.writerows(rqs_dataset)

print(f"Extracted RQs saved to: {output_file}")
print(f"Total RQs extracted: {len(rqs_dataset)}")
