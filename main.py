from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

MAX_NEW_TOKENS = 30
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


def get_hidden_states(model, tokenizer, text):
    encoded_input = tokenizer(text, return_tensors='pt', return_attention_mask=True)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input, output_hidden_states=True)
    return output.hidden_states


def load_dataset(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

love_dataset = load_dataset("datasets/love.txt")
hate_dataset = load_dataset("datasets/hate.txt")


def compute_steering_vector(love_hidden_states, hate_hidden_states, layer_idx):
    
    # get activations
    love_activations = love_hidden_states[layer_idx] # (batch_size, seq_len, hidden_size)
    hate_activations = hate_hidden_states[layer_idx] # (batch_size, seq_len, hidden_size)
    
    # reduce dimensionality
    love_activations = love_activations.squeeze(0) # (seq_len, hidden_size)
    hate_activations = hate_activations.squeeze(0) # (seq_len, hidden_size)

    ## average over seq_len
    love_activations = love_activations.mean(dim=0) # (hidden_size,)
    hate_activations = hate_activations.mean(dim=0) # (hidden_size,)
    
    return love_activations - hate_activations

def compute_mean_steering_vector(love_hidden_states_list, hate_hidden_states_list, layer_idx):
    steering_vector_list = []
    for love_hidden_states, hate_hidden_states in zip(love_hidden_states_list, hate_hidden_states_list):
        steering_vector_list.append(compute_steering_vector(love_hidden_states, hate_hidden_states, layer_idx))
    return torch.stack(steering_vector_list).mean(dim=0)

def generate_with_steering(model, tokenizer, prompt, steering_vector, layer_idx, alpha):
    
    def hook_fn(module, input, output):
        # hook_fn can "see" steering_vector and alpha 
        # from the outer function's scope
        return (output[0] + alpha * steering_vector,) + output[1:]
    
    handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
    # run generate()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    output_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    handle.remove()
    return tokenizer.decode(output_ids[0])


def layer_sweep(model, tokenizer, prompt, alpha):
    love_hidden_states_list = []
    hate_hidden_states_list = []
    for love_text, hate_text in zip(love_dataset, hate_dataset):
        love_hidden_states_list.append(get_hidden_states(lm_model, tokenizer, love_text)) # (num_texts, seq_len, hidden_size)
        hate_hidden_states_list.append(get_hidden_states(lm_model, tokenizer, hate_text)) # (num_texts, seq_len, hidden_size)

    for layer in range(model.config.n_layer):
        # steering_vector = compute_steering_vector(love_hidden_states, hate_hidden_states, layer)
        steering_vector = compute_mean_steering_vector(love_hidden_states_list, hate_hidden_states_list, layer)
        print(f"Layer {layer}:")
        print("Positive steering:")
        print(generate_with_steering(model, tokenizer, prompt, steering_vector, layer, alpha))
        print("Negative steering:")
        print(generate_with_steering(model, tokenizer, prompt, steering_vector, layer, -alpha))
        print("\n")

layer_sweep(lm_model, tokenizer, "I think Mondays are", 5.0)