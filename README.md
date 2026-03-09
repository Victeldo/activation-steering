# Activation Steering – GPT-2-XL

A practice reproduction of activation steering on GPT-2-XL, based on the paper:

> [Steering GPT-2-XL by adding an activation vector](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector) – Turner et al., 2023

## How it works

A **steering vector** is computed by taking the difference between the hidden states of a positive/negative contrast pair (e.g. "I love talking about weddings" vs "I hate talking about weddings") at each transformer layer. This vector is then injected into the model's residual stream during generation via a PyTorch forward hook, nudging the model's outputs in the desired direction.

The script runs a **layer sweep** – testing the steering vector at every layer of the model – and prints the positive and negative steered outputs for each layer.

## Usage

Edit the three constants at the top of `main.py`:

```python
MAX_NEW_TOKENS = 30   # length of generated output
ALPHA = 5.0           # steering strength (try 1.0–20.0)
PROMPT = "I think Mondays are"  # prompt to steer
```

Then run:

```bash
python main.py
```

Positive steering pushes toward **love**, negative steering pushes toward **hate**. Layers in the middle of the network (~10–20 out of 48) tend to produce the most coherent steered outputs.

## Contrast pairs

The steering vector is averaged over 10 contrast pairs stored in `datasets/`:

- `datasets/love.txt` – positive examples (one per line)
- `datasets/hate.txt` – negative examples (one per line, paired with love.txt)

## Requirements

```bash
pip install torch transformers
```

GPT-2-XL (~6GB) will be downloaded automatically from Hugging Face on first run. A GPU (CUDA or Apple MPS) is recommended – CPU inference is very slow.
