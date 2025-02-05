import validation
from datasets import load_dataset, DatasetDict, Dataset
from datasets.features import features
from transformers.commands import train
from transformers import AutoTokenizer


raw_datasets = load_dataset("glue", "mrpc")
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset.features)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

#
# DatasetDict({
#     train: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 3668
#     })
#     validation: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 408
#     })
#     test: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 1725
#     })
# })