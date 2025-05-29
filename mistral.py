import json
import csv
from transformers import AutoTokenizer, pipeline
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id
)

# Load JSON
with open("abstracts_1k.json", "r", encoding="utf-8") as f:
    all_abstracts = json.load(f)

# ID ranges for testing
valid_ids = set(range(1, 6)) | set(range(251, 256)) | set(range(651, 656))

# Filter and process
rqs_dataset = []
for entry in all_abstracts:
    entry_id = entry.get("id")
    if entry_id not in valid_ids:
        continue

    abstract = entry.get("abstract", "").strip()
    if not abstract:
        continue

    prompt = (
        "[INST] Extract research question(s) from the following abstract. "
        "If none are explicitly stated, create new research question(s) for the abstract using cues such as "
        "‘this paper investigates’, ‘we aim to’, or ‘the objective is’.\n\n"
        "Abstract:\n"
        f"{abstract}\n\n"
        "Return the research questions only. The research question must be clear, relevant, feasible [/INST]"
    )
    output = generator(prompt, max_new_tokens=256)[0]["generated_text"]
    response = output[len(prompt):].strip()

    # Split into separate RQs by line if model outputs in bullet/numbered format
    for line in response.split("\n"):
        clean_line = line.strip("1234567890").strip()
        if clean_line:
            rqs_dataset.append({"research_question": clean_line})

# Write CSV
with open("rqs_dataset_15.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["research_question"])
    writer.writeheader()
    writer.writerows(rqs_dataset)

print(f"Extracted RQs saved to: rqs_dataset_15.csv")
print(f"Total RQs extracted: {len(rqs_dataset)}")