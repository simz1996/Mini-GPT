# Transformer Model with Rotary Embeddings

Welcome to the repository for my Transformer model implementation with Rotary Embeddings! This project showcases a powerful Transformer architecture built with PyTorch. The code includes custom implementations of RMSNorm, Self-Attention with Rotary Embeddings, and Feed Forward layers, all packed into a versatile EncoderBlock and Transformer class. This README will guide you through the code, its functionality, and how to get started.

## Features

- **Custom Transformer Architecture**: A modular implementation of a Transformer model, designed for flexibility and extensibility.
- **Rotary Embeddings**: Integration of rotary positional embeddings to enhance the model's performance.
- **RMSNorm**: A custom normalization layer (Root Mean Square Normalization) for stabilizing the training.
- **Attention Mechanism**: Self-Attention with rotary embeddings for capturing long-range dependencies in the input data.
- **Feed Forward Network**: A sophisticated feed-forward network with optional dimension multipliers.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Code Overview](#code-overview)
    - [Model Arguments](#model-arguments)
    - [Positional Frequencies](#positional-frequencies)
    - [Rotary Embeddings](#rotary-embeddings)
    - [Normalization](#normalization)
    - [Self-Attention](#self-attention)
    - [Feed Forward Network](#feed-forward-network)
    - [Encoder Block](#encoder-block)
    - [Transformer](#transformer)
4. [Contributing](#contributing)
5. [License](#license)

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/transformer-rotary-embeddings.git
cd transformer-rotary-embeddings
```

Install the required dependencies:

```bash
pip install torch
```

## Usage

You can instantiate and use the Transformer model with the following code:

```python
import torch
from your_module import ModelArgs, Transformer

args = ModelArgs(
    dim=4096,
    n_layer=32,
    n_heads=32,
    vocab_size=30522,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

model = Transformer(args)

tokens = torch.tensor([[1]]).to(args.device)
start_pos = 0

output = model(tokens, start_pos)
print(output)
```

## Code Overview

### Model Arguments

`ModelArgs` is a dataclass that holds the hyperparameters for the model. Adjust these parameters to customize your Transformer model.

```python
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_heads: int = 32
    vocab_size: int = -1
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = 'cuda'
    # ... other parameters
```

### Positional Frequencies

`precompute_theta_pos_frequencies` precomputes the positional frequencies for rotary embeddings.

```python
def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta=10000):
    # ... implementation
```

### Rotary Embeddings

`apply_rotary_embeddings` applies the rotary positional embeddings to the input tensor.

```python
def apply_rotary_embeddings(x, freq_complex, device):
    # ... implementation
```

### Normalization

`RMSNorm` implements Root Mean Square Normalization.

```python
class RMSNorm(nn.Module):
    # ... implementation
```

### Self-Attention

`SelfAttentionBlock` implements the self-attention mechanism with rotary embeddings.

```python
class SelfAttentionBlock(nn.Module):
    # ... implementation
```

### Feed Forward Network

`FeedForward` implements the feed-forward network with optional dimension multipliers.

```python
class FeedForward(nn.Module):
    # ... implementation
```

### Encoder Block

`EncoderBlock` combines the self-attention and feed-forward network into a single block.

```python
class EncoderBlock(nn.Module):
    # ... implementation
```

### Transformer

`Transformer` is the main model class that stacks multiple encoder blocks.

```python
class Transformer(nn.Module):
    # ... implementation
```

## Contributing

Feel free to fork this repository, make improvements, and create pull requests. Contributions are always welcome!

## License

This project is not licesed and free to use.

---

Thank you for checking out my Transformer model implementation. I hope you find it useful and insightful! If you have any questions or feedback, feel free to reach out.