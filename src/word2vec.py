import numpy as np
import collections
import urllib.request
import zipfile
import os
import pickle

class Word2VecNumPy:
    """
    A pure NumPy implementation of the Word2Vec Skip-gram model 
    with Negative Sampling (SGNS).
    """

    def __init__(self, vocab_size=10000, embed_dim=100, window_size=2, k_neg=5, lr=0.025):
        """
        Initializes the model with hyperparameters and random weight matrices.
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.k_neg = k_neg
        self.lr = lr
        
        # Matrix Initialization
        # W_in: Target word vectors (the ones we usually use for inference)
        # W_out: Context word vectors (used during training)
        self.W_in = np.random.uniform(-0.5/embed_dim, 0.5/embed_dim, (vocab_size, embed_dim))
        self.W_out = np.random.uniform(-0.5/embed_dim, 0.5/embed_dim, (vocab_size, embed_dim))
        
    def _sigmoid(self, x):
        """Stable sigmoid function to map values to [0, 1]."""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def train(self, train_data, unigram_table, epochs=1):
        """
        Core training loop implementing forward pass, loss, and backprop.
        
        Args:
            train_data (list): List of word IDs from the corpus.
            unigram_table (np.array): Precomputed table for negative sampling.
            epochs (int): Number of full passes over the dataset.
        """
        for epoch in range(epochs):
            total_loss = 0
            for i, target_id in enumerate(train_data):
                v_t = self.W_in[target_id]
                
                # Define window bounds around the target word
                start = max(0, i - self.window_size)
                end = min(len(train_data), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i == j: continue  # Skip the target word itself
                    context_id = train_data[j]
                    
                    # --- POSITIVE SAMPLE (Label = 1) ---
                    # We want to maximize the dot product of target and actual context
                    u_pos = self.W_out[context_id]
                    z_pos = np.dot(v_t, u_pos)
                    y_hat_pos = self._sigmoid(z_pos)
                    err_pos = y_hat_pos - 1 # Gradient wrt prediction
                    
                    # --- NEGATIVE SAMPLES (Label = 0) ---
                    # We want to minimize the dot product of target and random noise words
                    neg_indices = np.random.choice(unigram_table, self.k_neg)
                    u_negs = self.W_out[neg_indices] # Shape: (K, Dim)
                    z_negs = np.dot(u_negs, v_t)     # Shape: (K,)
                    y_hat_negs = self._sigmoid(z_negs)
                    err_negs = y_hat_negs - 0
                    
                    # --- GRADIENT CALCULATION ---
                    # dL/dv_t combines errors from the positive sample and all negative samples
                    grad_v_t = (err_pos * u_pos) + np.dot(err_negs, u_negs)
                    
                    # --- PARAMETER UPDATES ---
                    # Update Context vectors in W_out
                    self.W_out[context_id] -= self.lr * (err_pos * v_t)
                    self.W_out[neg_indices] -= self.lr * np.outer(err_negs, v_t)
                    
                    # Update Target vector in W_in
                    self.W_in[target_id] -= self.lr * grad_v_t
                
                if i % 100000 == 0 and i > 0:
                    print(f"Epoch {epoch+1} | Processed {i}/{len(train_data)} words")

    def get_similar(self, word, word_to_id, id_to_word, top_n=5):
        """
        Computes cosine similarity to find the most similar words to the input.
        """
        if word not in word_to_id: return "Word not in vocab"
        
        idx = word_to_id[word]
        vec = self.W_in[idx]
        
        # Normalize target vector and the entire matrix for cosine similarity calculation
        norm_v = vec / (np.linalg.norm(vec) + 1e-9)
        norm_W = self.W_in / (np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-9)
        
        # Dot product of normalized vectors yields cosine similarity
        similarities = np.dot(norm_W, norm_v)
        # Sort by similarity and return top N (excluding the word itself)
        closest = np.argsort(similarities)[::-1][1:top_n+1]
        return [id_to_word[c] for c in closest]

    def save_model(self, filepath, word_to_id, id_to_word):
        """
        Saves the model weights and vocabulary dictionaries to disk.
        """
        model_data = {
            'W_in': self.W_in,
            'W_out': self.W_out,
            'word_to_id': word_to_id,
            'id_to_word': id_to_word,
            'params': {
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved successfully to {filepath}")

    @staticmethod
    def load_model(filepath):
        """
        Loads a saved model and returns the model instance along with vocab maps.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct the class instance
        params = data['params']
        instance = Word2VecNumPy(vocab_size=params['vocab_size'], embed_dim=params['embed_dim'])
        instance.W_in = data['W_in']
        instance.W_out = data['W_out']
        
        print(f"Model loaded successfully from {filepath}")
        return instance, data['word_to_id'], data['id_to_word']

# --- DATA UTILITIES ---

def prepare_dataset(vocab_size=10000):
    """
    Downloads text8, cleans it, performs vocabulary mapping, and handles subsampling.
    """
    print("Downloading and processing data...")
    url = 'http://mattmahoney.net/dc/text8.zip'
    if not os.path.exists('text8.zip'):
        urllib.request.urlretrieve(url, 'text8.zip')
    
    with zipfile.ZipFile('text8.zip') as f:
        words = f.read(f.namelist()[0]).decode('utf-8').split()

    # Build Vocabulary: Keep top N most frequent words
    counts = collections.Counter(words).most_common(vocab_size - 1)
    word_to_id = {word: i+1 for i, (word, _) in enumerate(counts)}
    word_to_id['UNK'] = 0
    id_to_word = {i: w for w, i in word_to_id.items()}
    
    data = [word_to_id.get(w, 0) for w in words]
    
    # Subsampling: Discard frequent words probabilistically to improve context quality
    word_counts = collections.Counter(data)
    total = len(data)
    t = 1e-3
    probs = {i: (np.sqrt((c/total)/t) + 1) * (t/(c/total)) for i, c in word_counts.items()}
    train_data = [w for w in data if np.random.random() < probs.get(w, 0)]
    
    # Negative Sampling Table: Distribution raised to 3/4 power to boost rare word sampling
    f_pow = np.array([word_counts.get(i, 0) for i in range(vocab_size)])**0.75
    neg_probs = f_pow / np.sum(f_pow)
    unigram_table = np.random.choice(range(vocab_size), size=1000000, p=neg_probs)
    
    return train_data, unigram_table, word_to_id, id_to_word

# --- EXECUTION ---

if __name__ == "__main__":
    MODEL_PATH = "word2vec_numpy.pkl"
    V_SIZE = 10000
    
    print("--- Word2Vec NumPy System ---")
    choice = input("Enter 'T' to Train a new model or 'L' to Load an existing model: ").strip().upper()

    if choice == 'T':
        # Training Mode
        train_data, unigram_table, w2i, i2w = prepare_dataset(V_SIZE)
        model = Word2VecNumPy(vocab_size=V_SIZE, embed_dim=100)
        
        print("Starting training (this may take several minutes)...")
        model.train(train_data, unigram_table, epochs=1)
        
        # Save after training
        model.save_model(MODEL_PATH, w2i, i2w)
        
    elif choice == 'L':
        # Load Mode
        if not os.path.exists(MODEL_PATH):
            print(f"Error: No saved model found at {MODEL_PATH}. Please train one first.")
            exit()
        model, w2i, i2w = Word2VecNumPy.load_model(MODEL_PATH)
        
    else:
        print("Invalid choice. Exiting.")
        exit()

    # Final Test / Inference
    print("\n--- Model Inference Test ---")
    test_words = ['king', 'queen', 'apple', 'france', 'computer']
    for tw in test_words:
        results = model.get_similar(tw, w2i, i2w)
        print(f"Words most similar to '{tw}': {results}")
