import json

files_and_counts = [
    ("cl_abstracts.json", 262),
    ("conll_abstracts.json", 519),
    ("ranlp_abstracts.json", 717),
]
collected = []

for filename, count in files_and_counts:
    with open(filename, "r") as f:
        abstracts = json.load(f)
        collected.extend(abstracts[:count])

# Assign id from 1 to 1000
for idx, item in enumerate(collected, 1):
    item["id"] = idx

with open("abstracts_1498.json", "w") as out_f:
    json.dump(collected, out_f, indent=2)

