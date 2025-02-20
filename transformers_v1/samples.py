# from datasets import load_dataset
#
# raw_datasets = load_dataset("conll2003", trust_remote_code=True)
#
# print(raw_datasets["train"][0])
# print(raw_datasets["train"][0]["tokens"])

# from transformers import pipeline
#
# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# translator = pipeline("translation", model=model_checkpoint)
# print(translator("Default to expanded threads"))

import gradio as gr


def greet(name):
    return "Hello " + name


demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch()