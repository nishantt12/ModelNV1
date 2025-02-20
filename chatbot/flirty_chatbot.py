from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# from huggingface_hub import login
# login(token)

# model_name = "mistralai/Mistral-7B-v0.1"
# # model_name = "deepseek-ai/DeepSeek-R1"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
# # Load model directly
# # from transformers import AutoModelForCausalLM
# # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
# def generate_reply(message):
#     input_text = f"Message: {message}\nFunny Reply:"
#     inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
#     output = model.generate(**inputs, max_new_tokens=50)
#     return tokenizer.decode(output[0], skip_special_tokens=True)
#
# print(generate_reply("Hey, you look cute!"))

# from transformers import pipeline

# chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

# response = chatbot("Tell me a joke", max_length=100, do_sample=True)
# print(response[0]["generated_text"])


import requests

API_KEY = token
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {API_KEY}"}

def chat(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response.json()

# Example
user_input = "What is the capital of France?"
response = chat(user_input)
print(response)
