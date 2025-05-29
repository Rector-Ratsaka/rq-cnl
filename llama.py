import json
import csv
import re
import torch
from transformers import AutoTokenizer, pipeline

model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Target ID ranges
target_ids = set(range(1, 6)) | set(range(251, 256)) | set(range(651, 656))

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
with open("abstracts_1k.json", "r", encoding="utf-8") as f:
    abstracts = json.load(f)

# Process and collect research questions
rqs_dataset = []
for entry in abstracts:
    if entry["id"] not in target_ids or not entry.get("abstract", "").strip():
        continue

    abstract = entry["abstract"].strip()

    # LLaMA 3 prompt format
    prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are an expert CS/IT researcher. Your task is to extract or generate two clear, precise, and feasible research questions from the abstract below.\n"
    "Each research question should meet the following criteria (FINERMAPS subset) in CS/IT domain:\n"
    "- Feasible: Can be investigated using available tools, time, and data.\n"
    "- Relevant: Addresses a significant academic or applied research problem.\n"
    "- Measurable: Can be evaluated using specific methods or metrics.\n"
    "- Precise & Specific: Narrowly focused, avoids vague or broad terms.\n"
    "- Clear: Easy to understand, grammatically sound, and logically structured.\n"
    "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n"
    "Example Abstract:\n"
    "The volume of academic articles is increasing rapidly, reflecting the growing emphasis on research and scholarship across different science disciplines. This rapid growth necessitates the development of tools for more efficient and rapid understanding of these articles. Clear and well-defined Research Questions (RQs) in research articles can help guide scholarly inquiries. However, many academic studies lack a proper definition of RQs in their articles. This research addresses this gap by presenting a comprehensive framework for the systematic extraction, detection, and generation of RQs from scientific articles. The extraction component uses a set of regular expressions to identify articles containing well-defined RQs. The detection component aims to identify more complex RQs in articles, beyond those captured by the rule-based extraction method. The RQ generation focuses on creating RQs for articles that lack them. We integrate all these components to build a pipeline to extract RQs or generate them based on the articles’ full text. We evaluate the performance of the designed pipeline on a set of metrics designed to assess the quality of RQs. Our results indicate that the proposed pipeline can reliably detect RQs and generate high-quality ones.\n\n"
    "Example Research Questions:\n"
    "1. How effective is a text classification approach in detecting different research question patterns in research articles?\n"
    "2. How do different LLMs perform on the task of generating well-defined research questions?\n"
    "3. How does a unified pipeline combining research question extraction, detection, and generation components perform in forming well-structured research questions for scientific articles?\n\n"
    "How these meet the criteria:\n"
    "- Feasible: All three questions can be investigated using text classification, LLM evaluation, and pipeline experimentation.\n"
    "- Relevant: They target real problems in scholarly communication and automation.\n"
    "- Measurable: Each question allows empirical testing (e.g., F1 scores, BLEU scores, structural quality).\n"
    "- Precise & Specific: Each question focuses on a concrete aspect (text classification, LLMs, unified pipeline).\n"
    "- Clear: They are grammatically correct and logically phrased without ambiguity.\n\n"
    "Now apply the same structure to the following abstract:\n"
    f"{abstract}\n\n"
    "Return exactly two research questions. Do not include any preamble, explanation, numbering, quotation marks, or bullet points."
    "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
    )

    output = pipe(prompt, max_new_tokens=256)[0]["generated_text"]
    response = output[len(prompt):].strip()
    for line in response.split("\n"):
    	# Remove numbers and quotes
    	clean_line = re.sub(r'^\s*["\']?\d+[\.\)]\s*', '', line).strip(' "\'\n')
        if clean_line:
            rqs_dataset.append({"research_question": clean_line})

# Save to CSV
with open("second_itr.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["research_question"])
    writer.writeheader()
    writer.writerows(rqs_dataset)

print(f"Total RQs extracted: {len(rqs_dataset)}")