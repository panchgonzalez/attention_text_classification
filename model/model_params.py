"""Defines default model parameters."""

from collections import defaultdict

BASE_PARAMS = defaultdict(
    lambda: None,  # set default value to None.
    learning_rate=0.0001,  # Default learning rate
    batch_size=512,  # Default batch size
    vocab_size=20000,  # Number of tokens defined in vocab
    embedding_size=50,  # Embedding dimension
    hidden_size=128,  # Number of RNN hidden units
    attention_size=50,  # Size of attention mechanism
    train_steps=100000,  # number of steps to train
    model_dir="ckpt/",  # Model directory
    data_dir="tmp/",  # Data directory
)
