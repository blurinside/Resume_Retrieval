import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your saved model weights
model_path = r'C:\Users\aryan\Downloads\model.pth'  # Adjust this to your model path

# Load the tokenizer (assuming you're using the same tokenizer as the model)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-3.1-8B")  # Adjust this to your specific model identifier

# Initialize the model architecture
model = AutoModelForCausalLM.from_pretrained("meta-llama/LLaMA-3.1-8B")  # Adjust this to your specific model identifier

# Load the state dict from the .pth file
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully!")
