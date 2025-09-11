# rq-cnl

Using llms to extract/create research questions from abstracts then designing a cnl

python3 <model_script> <abstracts_file(json)> <prompt_file(txt)> <output(reseach_questions_in_csv)>

e.g:
python3 llama.py abstracts/combined_abstracts.json prompts/llama_prompt1.txt research_questions/rqs_llama_itr1.csv
