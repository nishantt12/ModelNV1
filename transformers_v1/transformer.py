from transformers import pipeline
from transformers import BertConfig, BertModel


# classifier = pipeline("sentiment-analysis")
# print(classifier("I've been waiting for a HuggingFace course my whole life."))

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])

config = BertConfig()
model = BertModel(config)

print(config)