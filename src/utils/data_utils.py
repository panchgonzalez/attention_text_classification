"""Data utility functions"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
from tqdm import tqdm

tqdm.pandas()

import pickle
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split

import tensorflow as tf

# Training data
DATA_URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
TRAIN_FILE = "training.1600000.processed.noemotion.csv"
TEST_FILE = "testdata.manual.2009.06.14.csv"
DATASET_ENCODING = "ISO-8859-1"
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]

EMBEDDING_URL = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
EMBEDDING_FILE = "glove.twitter.27B.50d.txt"
VOCAB_SIZE = 20000
EMBEDDING_SIZE = 50


def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_FILE, TRAIN_URL, extract=True)
    test_path = tf.keras.utils.get_file(TEST_FILE, TRAIN_URL, extract=True)
    return train_path, test_path


def maybe_download_embedding():
    """Downloads pretrained glove embeddings"""
    # Defaulting to glove twitter embedding with dimension 50

    # HACK: wrap in try-except block since it fails once file is extracted and found
    try:
        path = tf.keras.utils.get_file(EMBEDDING_FILE, EMBEDDING_URL, extract=True)
    except:
        path = tf.keras.utils.get_file(EMBEDDING_FILE, EMBEDDING_URL, extract=True)

    return path


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):

    if type(text) is not str:
        text = "unk"

    text = text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"@\S+|https?:\S+|http?:\S", " ", text)
    text = re.sub(r"[0-9]", "", text)
    text = re.sub(r"[^A-Za-z^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer("english")
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return text.strip()


def load_data():
    """Return train, test, and inference data."""

    # Maybe download training data
    train_path, test_path = maybe_download()

    # Read data into dataframe
    df = pd.read_csv(train_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
    inf_df = pd.read_csv(test_path, encoding=DATASET_ENCODING)

    # Clean dataset
    print("  Cleaning training data")
    df.text = df.text.progress_apply(lambda x: text_to_wordlist(x))
    print("  Cleaning inference data")
    inf_df.transcript = inf_df.transcript.progress_apply(lambda x: text_to_wordlist(x))

    # Replace 4 with 1
    df.target.replace(4, 1, inplace=True)

    return df, inf_df


def get_tokenizer(text):
    """Init tokenizer and git on texts"""
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(text)
    return tokenizer


def get_padded_sequences(tokenizer, text):
    """Return text as padded sequences"""
    sequences = tokenizer.texts_to_sequences(text)
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)
    return padded_seq


def get_data(params):
    """Load, clean, tokenize, and pad data fixedlen sequences."""

    # Load data if saved previously
    if os.path.exists(params.data_dir):
        print(f"Loading previously cleaned data from {params.data_dir}")

        with open(os.path.join(params.data_dir, "data.pkl"), "rb") as f:
            train_X, test_X, train_y, test_y, inf_text, tokenizer = pickle.load(f)

        return train_X, train_y, test_X, test_y, inf_text, tokenizer

    # Clean data and save
    else:

        # Get training and inference data
        print("Getting data ready")
        df, inf_df = load_data()

        # Initilize and fit a tokenizer
        print("Fitting tokenizer")
        tokenizer = get_tokenizer(df.text)

        # Pad text sequences
        print("Padding text sequences")
        text = get_padded_sequences(tokenizer, df.text)
        inf_text = get_padded_sequences(tokenizer, inf_df.transcript)

        # Split data into train/test sets
        train_X, test_X, train_y, test_y = train_test_split(
            text, df.target, test_size=0.25
        )

        # Save cleaned data for later
        print(f"Saving cleaned data to {params.data_dir}")
        os.makedirs(params.data_dir)
        with open(os.path.join(params.data_dir, "data.pkl"), "wb") as f:
            pickle.dump((train_X, test_X, train_y, test_y, inf_text, tokenizer), f)

        return train_X, train_y, test_X, test_y, inf_text, tokenizer


def get_embedding_matrix(tokenizer):
    """Construct embeding matrix from pretrained embedding."""

    embedding_path = maybe_download_embedding()

    embeddings_index = dict()
    with open(embedding_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    embedding = np.empty((VOCAB_SIZE, 50))
    for i in range(1, VOCAB_SIZE + 1):
        word = tokenizer.index_word[i]
        try:
            embedding[i - 1] = embeddings_index[word]
        except:
            embedding[i - 1] = np.random.uniform(-1, 1, (EMBEDDING_SIZE))

    return embedding


def train_input_fn(features, labels, batch_size, buffer_size=None):
    """Load and return batched examples."""
    assert buffer_size is not None

    # Convert inputs into tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle dataset, repeat, and make batches
    dataset = dataset.shuffle(buffer_size).repeat()
    dataset = dataset.batch(batch_size)

    return dataset


def eval_input_fn(features, labels, batch_size, shuffle=False, buffer_size=None):
    """Load and return batched examples."""
    assert buffer_size is not None

    if labels is None:
        # No labels for predicting, use only features
        inputs = features
    else:
        inputs = (features, labels)

    # Convert inputs into tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Shuffle dataset, repeat, and make batches
    if shuffle:
        dataset = dataset.shuffle(buffer_size).repeat()
    dataset = dataset.batch(batch_size)

    return dataset
