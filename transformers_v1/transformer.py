from transformers import pipeline
from transformers import BertConfig, BertModel
from transformers import AutoTokenizer
from datasets import load_dataset




# classifier = pipeline("sentiment-analysis")
# print(classifier("I've been waiting for a HuggingFace course my whole life."))

# unmasker = pipeline("fill-mask", model="bert-base-uncased")
# result = unmasker("This man works as a [MASK].")
# # print([r["token_str"] for r in result])
#
# result = unmasker("This woman works as a [MASK].")
# # print([r["token_str"] for r in result])
#
# config = BertConfig()
# model = BertModel(config)
#
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#
# sequence = "Using a Transformer network is simple"
# tokens = tokenizer.tokenize(sequence)
#
# ids = tokenizer.convert_tokens_to_ids(tokens)
#
# decoded_string = tokenizer.decode(ids)
#
# print(decoded_string)

from transformers import pipeline

camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("who are <mask> :)")
print(results)