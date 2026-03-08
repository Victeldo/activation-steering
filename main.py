from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
# IMPORTANT: GPT-2 doesn't have a pad_token by default
tokenizer.pad_token = tokenizer.eos_token
# Example: Single text (no padding needed)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model and configure pad_token
lm_model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
lm_model.config.pad_token_id = tokenizer.pad_token_id
lm_model.to(device)


def get_activations(model, tokenizer, text, layer_idx):
    encoded_input = tokenizer(text, return_tensors='pt', return_attention_mask=True)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input, output_hidden_states=True)
    return output.hidden_states[layer_idx]

# print(get_activations(lm_model, tokenizer, "I love talking about weddings", 11))

def load_dataset(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

love_dataset = load_dataset("datasets/love.txt")
hate_dataset = load_dataset("datasets/hate.txt")

# ## get activations
# hate_activations = get_activations(lm_model, tokenizer, hate_dataset[0], 11) # (batch_size, seq_len, hidden_size)
# love_activations = get_activations(lm_model, tokenizer, love_dataset[0], 11) # (batch_size, seq_len, hidden_size)

# ## reduce dimensionality

# hate_activations = hate_activations.squeeze(0) # (seq_len, hidden_size)
# love_activations = love_activations.squeeze(0) # (seq_len, hidden_size)

# ## average over seq_len
# hate_activations = hate_activations.mean(dim=0) # (hidden_size,)
# love_activations = love_activations.mean(dim=0) # (hidden_size,)

# steering_vector = love_activations - hate_activations

def compute_steering_vector(love_dataset, hate_dataset, layer_idx):
    
    # get activations
    love_activations = get_activations(lm_model, tokenizer, love_dataset[0], layer_idx) # (batch_size, seq_len, hidden_size)
    hate_activations = get_activations(lm_model, tokenizer, hate_dataset[0], layer_idx) # (batch_size, seq_len, hidden_size)
    
    # reduce dimensionality
    love_activations = love_activations.squeeze(0) # (seq_len, hidden_size)
    hate_activations = hate_activations.squeeze(0) # (seq_len, hidden_size)

    ## average over seq_len
    love_activations = love_activations.mean(dim=0) # (hidden_size,)
    hate_activations = hate_activations.mean(dim=0) # (hidden_size,)
    
    return love_activations - hate_activations

def generate_with_steering(model, tokenizer, prompt, steering_vector, layer_idx, alpha):
    
    def hook_fn(module, input, output):
        # hook_fn can "see" steering_vector and alpha 
        # from the outer function's scope
        return (output[0] + alpha * steering_vector,) + output[1:]
    
    handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
    # run generate()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    output_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    handle.remove()
    return tokenizer.decode(output_ids[0])

# print(generate_with_steering(lm_model, tokenizer, "I think weddings are", steering_vector, 11, -5.0))


def layer_sweep(model, tokenizer, prompt, steering_vector, alpha):
    for layer in range(model.config.n_layers):
        steering_vector = compute_steering_vector(love_dataset, hate_dataset, layer)
        print(f"Layer {layer}:")
        print("Positive steering:")
        print(generate_with_steering(model, tokenizer, prompt, steering_vector, layer, alpha))
        print("Negative steering:")
        print(generate_with_steering(model, tokenizer, prompt, steering_vector, layer, -alpha))
        print("\n")

layer_sweep(lm_model, tokenizer, "I think weddings are", steering_vector, 5.0)