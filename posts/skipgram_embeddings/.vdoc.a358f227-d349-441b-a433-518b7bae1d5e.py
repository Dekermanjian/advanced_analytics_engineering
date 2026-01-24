# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
import jax
import jax.numpy as jnp
from jax import random
from jax.nn import sigmoid
from functools import partial


class SGNSLoss:
    BETA = 0.75
    NUM_SAMPLES = 15

    def __init__(self, vocabulary, token_counts, key):
        self.vocabulary = vocabulary
        self.vocabulary_len = len(dataset.dictionary)
        self.key = key

        # Precompute unigram distribution
        freqs = jnp.array([dataset.dictionary.dfs[i] for i in range(self.vocab_len)], dtype=jnp.float32)
        transformed = freqs ** self.BETA
        self.unigram_probs = transformed / transformed.sum()

    def forward(self, params, batch):
        center_ids, context_ids = batch
        center = params[center_ids]  # (batch_size, embed_dim)
        context = params[context_ids]  # (batch_size, embed_dim)

        # Positive logits
        true_logits = jnp.sum(center * context, axis=-1)
        loss = self._bce_loss_with_logits(true_logits, jnp.ones_like(true_logits))

        # Negative sampling
        key, subkey = random.split(self.key)
        neg_ids = random.categorical(subkey, jnp.log(self.unigram_probs), shape=(self.NUM_SAMPLES, center.shape[0]))
        neg_embeds = params[neg_ids]  # (num_samples, batch_size, embed_dim)

        # Compute loss for each negative sample
        def neg_loss_fn(neg):
            logits = jnp.sum(center * neg, axis=-1)
            return self._bce_loss_with_logits(logits, jnp.zeros_like(logits))

        neg_losses = jax.vmap(neg_loss_fn)(neg_embeds)
        total_neg_loss = jnp.sum(neg_losses, axis=0)  # sum over negative samples

        return loss + total_neg_loss.mean()

    @staticmethod
    def _bce_loss_with_logits(logits, labels):
        return jnp.mean(jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits))))

#
#
#
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, List


class SkipGramEmbeddings(nn.Module):
    vocab_size: int
    embed_len: int

    @nn.compact
    def __call__(self, center: jnp.ndarray, context: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_len)
        word_embeds = embedding.embedding  # Retrieve embedding matrix
        return word_embeds[center], word_embeds[context]

    @staticmethod
    def nearest_neighbors(word: str, dictionary, vectors: jnp.ndarray) -> List[str]:
        """
        Finds top-10 nearest neighbors to the given word using cosine similarity.

        :param word: Target word as a string.
        :param dictionary: Gensim dictionary-like object with token2id and id2token.
        :param vectors: jnp.ndarray of shape (vocab_size, embed_len)
        :return: List of top 10 similar words.
        """
        index = dictionary.token2id[word]
        query = vectors[index]

        # Normalize vectors and query
        vectors_norm = vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)
        query_norm = query / jnp.linalg.norm(query)

        # Compute cosine similarities
        similarities = jnp.dot(vectors_norm, query_norm)

        # Get top 10 indices (excluding the word itself)
        top_indices = jnp.argsort(similarities)[::-1][1:11]

        # Map back to words
        return [dictionary[idx] for idx in top_indices]

#
#
#
#
vocab_size = 10_000
embed_len = 300
model = SkipGramEmbeddings(vocab_size=vocab_size, embed_len=embed_len)
rng = random.PRNGKey(0)
params = model.init(rng, jnp.array([1, 2]), jnp.array([3, 4]))
center_embeds, context_embeds = model.apply(params, jnp.array([1, 2]), jnp.array([3, 4]))
#
#
#
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
def tokenize_and_filter(text: str, min_occurrence: int) -> list[str]:
    """
    Tokenizes input text and filters out tokens that occur less than or equal to `min_occurrence` times.

    Args:
        text (str): Input text to be tokenized.
        min_occurrence (int): Minimum number of times a token must appear in the text to be included.

    Returns:
        List[str]: A list of tokens that occur more than `min_occurrence` times.
    """
    # Convert text to lowercase and tokenize
    tokens = word_tokenize(text.lower().strip())

    # Count occurrences of each token
    token_counts = Counter(tokens)

    # Filter tokens based on minimum occurrence threshold
    filtered_tokens = [token for token in tokens if token_counts[token] > min_occurrence]

    return filtered_tokens
#
#
#
def build_word_index(text: str) -> tuple[dict[str, int], list[str]]:
    """
    Builds a word-to-index mapping (vocabulary) from input text, including a special <unk> token.

    Args:
        text (str): Input text to extract vocabulary from.

    Returns:
        Tuple[Dict[str, int], List[str]]: 
            - word_to_index: A dictionary mapping each word to a unique integer ID.
            - vocabulary: A sorted list of unique vocabulary words including "<unk>".
    """
    # Tokenize and get unique words
    unique_tokens = set(tokenize_and_filter(text, min_occurrence=0))

    # Add special token for unknown words
    vocabulary = sorted(unique_tokens) + ["<unk>"]

    # Create word-to-index mapping
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}

    return word_to_index, vocabulary

#
#
#
def tokens_to_ids(text: str, word_to_index: dict[str, int]) -> list[int]:
    """
    Converts a text string into a list of integer token IDs using a provided word-to-index mapping.
    Unknown tokens are mapped to the ID of the "<unk>" token.

    Args:
        text (str): Input text to convert.
        word_to_index (Dict[str, int]): A dictionary mapping words to their corresponding integer IDs.

    Returns:
        List[int]: A list of integer IDs representing the tokenized input text.
    """
    tokens = tokenize_and_filter(text, min_occurrence=0)
    token_ids = [word_to_index.get(token, word_to_index["<unk>"]) for token in tokens]
    return token_ids

#
#
#
#
with open('./world_order_kissinger.txt', 'r', encoding='utf-8') as file:
    text = file.read()
#
#
#
word_to_index, vocabulary = build_word_index(text=text)
#
#
#
#
def generate_skipgram_pairs(batch_texts: list[str], word_to_index: dict[str, int], max_sequence_length: int, context_window_size: int):
    """
    Generate training pairs for the Skip-Gram model from a batch of input texts.

    Args:
        batch_texts (List[str]): A list of input text strings.
        word_to_index (Dict[str, int]): A mapping from tokens (words) to their corresponding integer IDs.
        max_sequence_length (int): The maximum number of tokens to use per input text.
        context_window_size (int): The number of words to use on each side of the center word as context.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing two JAX arrays:
            - center_words: 1D array of center word IDs (inputs).
            - context_words: 1D array of corresponding context word IDs (targets).
    """
    center_words, context_words = [], []

    for text in batch_texts:
        text = text.strip()
        token_ids = token_to_number(text, word_to_index)

        # Skip sequences too short for the context window
        if len(token_ids) < context_window_size * 2 + 1:
            continue

        # Truncate long sequences
        if len(token_ids) > max_sequence_length:
            token_ids = token_ids[:max_sequence_length]

        # Slide a window over the token sequence
        for i in range(len(token_ids) - context_window_size * 2):
            window = token_ids[i: i + context_window_size * 2 + 1]
            center = window[context_window_size]
            context = window[:context_window_size] + window[context_window_size + 1:]

            for context_word in context:
                center_words.append(center)
                context_words.append(context_word)

    center_tensor = jnp.array(center_words, dtype=jnp.int32)
    context_tensor = jnp.array(context_words, dtype=jnp.int32)

    return center_tensor, context_tensor

#
#
#
batch_texts = sent_tokenize(text.lower().strip())
#
#
#
dataset = generate_skipgram_pairs(batch_texts=batch_texts, word_to_index=word_to_index, max_sequence_length=200, context_window_size=2)
#
#
#
# Load data
vocab_size = len(vocabulary)
embed_len = 300
batch_size = 16

dataloader = NumpyLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers)

model = SkipGramEmbeddings(vocab_size, embed_len)
sgns = SGNSLoss(dataset)

# Set up optimizer - rmsprop seems to work the best
optimizer = optim.adam(1e-3)
opt_init = optimizer.init
opt_update = optimizer.update
apply_updates = optim.apply_updates

@partial(jit, static_argnums=(0,))
def update(params, opt_state, batch):
    g = grad(sgns.forward)(params, batch)
    updates, opt_state = opt_update(g, opt_state)
    params = apply_updates(params, updates)
    return opt_state, params, g

def train():
    # Initialize optimizer state!
    params = model.word_embeds
    opt_state = opt_init(params)
    for epoch in range(epochs):
        print(f'Beginning epoch: {epoch + 1}/{epochs}')
        for i, batch in enumerate(tqdm(dataloader)):
            opt_state, params, g = update(params, opt_state, batch)
        log_step(epoch, params, g)

def log_step(self, epoch, params, g):
    print(f'EPOCH: {epoch} | GRAD MAGNITUDE: {np.sum(g)}')
    # Log embeddings!
    print('\nLearned embeddings:')
    for word in dataset.queries:
        print(f'word: {word} neighbors: {model.nearest_neighbors(word, dataset.dictionary, params)}')

#
#
#
