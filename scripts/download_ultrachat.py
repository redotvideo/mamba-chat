import json
from datasets import load_dataset

data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")


with open("../data/ultrachat.jsonl", "w") as f:
    for d in data:
        f.write(json.dumps(dict(messages=d["messages"]))+"\n")

