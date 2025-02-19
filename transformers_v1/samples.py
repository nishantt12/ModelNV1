from datasets import load_dataset

raw_datasets = load_dataset("conll2003", trust_remote_code=True)

print(raw_datasets["train"][0])
print(raw_datasets["train"][0]["tokens"])