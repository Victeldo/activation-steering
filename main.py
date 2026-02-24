from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
# IMPORTANT: GPT-2 doesn't have a pad_token by default
tokenizer.pad_token = tokenizer.eos_token
# Example: Single text (no padding needed)

# Load model and configure pad_token
lm_model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
lm_model.config.pad_token_id = tokenizer.pad_token_id

# text = "I want to tell you that"
# encoded_input = tokenizer(text, return_tensors='pt', return_attention_mask=True)

# print(f"\nInput: '{text}'")
# print("Generating...")

# output_ids = lm_model.generate(
#     encoded_input['input_ids'],
#     attention_mask=encoded_input['attention_mask'],  # Pass attention mask
#     max_new_tokens=10,
#     do_sample=False,
#     pad_token_id=tokenizer.pad_token_id  # Set pad_token_id
# )

# generated_text = tokenizer.decode(output_ids[0])
# print(f"Generated: '{generated_text}'")


def get_activations(model, tokenizer, text, layer_idx):
    encoded_input = tokenizer(text, return_tensors='pt', return_attention_mask=True)
    with torch.no_grad():
        output = model(**encoded_input, output_hidden_states=True)
    return output.hidden_states[layer_idx]

# print(get_activations(lm_model, tokenizer, "I love talking about weddings", 11))

def load_dataset(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

love_dataset = load_dataset("datasets/love.txt")
hate_dataset = load_dataset("datasets/hate.txt")

## get activations
hate_activations = get_activations(lm_model, tokenizer, hate_dataset[0], 11)
love_activations = get_activations(lm_model, tokenizer, love_dataset[0], 11)

## reduce dimensionality

hate_activations = hate_activations.squeeze(0)
love_activations = love_activations.squeeze(0)

## average over seq_len
hate_activations = hate_activations.mean(dim=0)
love_activations = love_activations.mean(dim=0)

steering_vector = hate_activations - love_activations