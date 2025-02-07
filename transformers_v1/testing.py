from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load DeepSeek model and tokenizer
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"  # Change this if using a different version
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Chatbot loop
print("DeepSeek Chatbot: Type 'exit' to end chat")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\nBot:", response)
