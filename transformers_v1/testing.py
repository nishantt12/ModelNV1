from datasets import load_dataset

# Load the dataset
ds = load_dataset("fka/awesome-chatgpt-prompts")

# Print a sample prompt
print(ds['train'][0])

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# model_name = "mistralai/Mistral-7B-Instruct"  # You can use other models as well
model_name = "gpt2"  # You can use other models as well

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Get a sample prompt from the dataset
prompt = ds['train'][9]['prompt']

# Generate a response
response = generator(prompt, max_length=200, temperature=0.7)

# Print the output
print(response[0]['generated_text'])



