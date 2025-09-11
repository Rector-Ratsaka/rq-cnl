# RQ-CNL: Enhancing Research Question Quality with Controlled Natural Language

This project explores how large language models (LLMs) can extract research questions (RQs) from scientific abstracts and transform them into Controlled Natural Language (CNL) templates. CNL is a simplified form of natural language with restricted grammar and vocabulary, designed to make RQs clearer, more structured, and machine-readable while remaining human-friendly.

By combining LLM-based RQ extraction with CNL-guided template design, this work evaluates whether abstracts can serve as a reliable source for identifying reusable RQ patterns.

## To run models in /models folder follow this:

python3 <model_script> <abstracts_file(json)> <prompt_file(txt)> <output(reseach_questions_in_csv)>

e.g:
python3 llama.py abstracts/combined_abstracts.json prompts/llama_prompt1.txt research_questions/rqs_llama_itr1.csv
