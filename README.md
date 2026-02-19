# Word2Vec with NumPy

A high-performance implementation of the Skip-gram with Negative Sampling (SGNS) architecture using only Python and NumPy. This project demonstrates the fundamental mechanics of word embeddings, including manual backpropagation, self-supervised label generation, and efficient sampling techniques.

## Overview

Unlike high-level frameworks (PyTorch/TensorFlow), this implementation manually handles the optimization procedure. It transforms raw text from the Text8 dataset into dense, 100-dimensional vectors that capture semantic relationships (e.g., "king" is mathematically similar to "queen").

## Features

- **Skip-gram Architecture**: Predicts surrounding context words given a target word.
- **Negative Sampling (SGNS)**: Replaces the expensive $O(V)$ Softmax with a binary classification task $O(K)$, significantly speeding up training.
- **Subsampling**: Reduces the influence of frequent "noise" words like "the" or "is" using probabilistic discarding.
- **Unigram Table**: Pre-computes word distributions raised to the $3/4$ power for efficient $O(1)$ noise sampling.
- **Pure NumPy**: No autograd libraries. All gradients and weight updates are derived and implemented manually.

## The Mathematics

### 1. Forward Pass
For a target word $v_t$ and a context word $u_c$, we calculate the probability of them being neighbors using the dot product and the Sigmoid function:

$$
P(1 | v_t, u_c) = \sigma(v_t \cdot u_c) = \frac{1}{1 + e^{-(v_t \cdot u_c)}}
$$

### 2. Loss Function (Binary Cross-Entropy)
The model maximizes the probability of real word pairs while minimizing the probability of random noise pairs:

$$
J = -\log\sigma(v_t \cdot u_{pos}) - \sum_{i=1}^{k} \log\sigma(-v_t \cdot u_{neg_i})
$$

### 3. Gradient Updates
We update the vectors using the error $\text{err} = (\hat{y} - y)$. For each step, the gradients are applied to the input ($W_{in}$) and output ($W_{out}$) matrices:

$$
\frac{\partial J}{\partial v_t} = (\sigma(v_t \cdot u_c) - y)u_c
$$

## Implementation Plan

- **Preprocessing**: Tokenization and cleaning of the Text8 dataset.
- **Vocabulary Building**: Mapping the top 10,000 words to unique IDs.
- **Subsampling**: Pruning frequent words to improve vector quality and training speed.
- **Negative Sampling Table**: Generating a table based on the $3/4$ power of word frequencies.
- **Training Loop**:
  1. Iterate through the corpus with a sliding window.
  2. Perform a forward pass for positive and negative samples.
  3. Update weights using manual backpropagation.
- **Inference**: Using Cosine Similarity to find the nearest neighbors for a given word.

## Hyperparameters

| Parameter     | Value   | Description                              |
|---------------|---------|------------------------------------------|
| **EMBED_DIM** | 100     | Size of the word vector.                 |
| **WINDOW_SIZE** | 2      | Number of neighbors to consider.         |
| **K_NEG**     | 5       | Number of negative samples per positive pair. |
| **LEARNING_RATE** | 0.025 | The step size for gradient updates.      |
| **VOCAB_SIZE** | 10,000  | Number of unique words in the dictionary. |

## Usage

### Prerequisites

- Python 3.x
- NumPy

### Running the model

```bash
cd src
python3 word2vec.py
```
### Expected Output
```
--- Word2Vec NumPy System ---
Enter 'T' to Train a new model or 'L' to Load an existing model: L
Model loaded successfully from word2vec_numpy.pkl

--- Model Inference Test ---
Words most similar to 'king': ['queen', 'elizabeth', 'henry', 'constantine', 'emperor']
Words most similar to 'queen': ['elizabeth', 'princess', 'prince', 'frederick', 'elector']
Words most similar to 'apple': ['macintosh', 'ibm', 'ms', 'amiga', 'dos']
Words most similar to 'france': ['spain', 'italy', 'austria', 'portugal', 'germany']
Words most similar to 'computer': ['console', 'digital', 'handheld', 'graphics', 'macintosh']
```
