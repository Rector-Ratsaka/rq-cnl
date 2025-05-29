import json

files_and_counts = [
    ("abstracts/cl_abstracts.json", 250),
    ("abstracts/conll_abstracts.json", 500),
    ("abstracts/ranlp_abstracts.json", 500),
    ("abstracts/wmt_abstracts.json", 500),
    ("abstracts/lrec_abstracts.json", 750)
]
collected = []

for filename, count in files_and_counts:
    with open(filename, "r") as f:
        abstracts = json.load(f)
        collected.extend(abstracts[:count])

# Assign id from 1 to 2500
for idx, item in enumerate(collected, 1):
    item["id"] = idx

with open("abstracts/combined_abstracts_2500.json", "w") as out_f:
    json.dump(collected, out_f, indent=2)

