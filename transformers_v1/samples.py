from datasets import load_dataset

ds = load_dataset("aidanzhou/tinderflirting")

print(ds['train'][:5])
