from transformers import AutoModel, AutoTokenizer

# List of models to download
models = [
    "microsoft/codebert-base"
]

# Function to download and save models and tokenizers
def download_and_save_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(f"./{model_name}")
    tokenizer.save_pretrained(f"./{model_name}")

# Download and save each model and tokenizer
for model_name in models:
    download_and_save_model(model_name)

print("All models and tokenizers downloaded and saved.")
