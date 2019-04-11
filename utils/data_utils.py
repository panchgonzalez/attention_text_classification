"""Data utility functions"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

import tensorflow as tf

TRAIN_URL = "https://storage.googleapis.com/gk-data/interview-problem/corpus.csv"
DATASET_ENCODING = "ISO-8859-1"
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
INFERENCE_PATH = "../data/calls-with-transcripts.csv"
STOPWORDS = set(stopwords.words("english"))
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"


def maybe_download():
    return tf.keras.utils.get_file(TRAIN_URL.split("/")[-1], TRAIN_URL)


def preprocess(text):
    """Remove links and input handles"""
    text = re.sub(TEXT_CLEANING_RE, " ", str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in STOPWORDS:
            tokens.append(token)
    return " ".join(tokens)


def _index_tokens(X, indexer):
    return [indexer.word_index[x] if x in indexer.word_index else 0 for x in X.split()]


def load_data():
    """Return train, test, and inference data."""

    # Maybe download training data
    train_test_path = maybe_download()

    # Read data into dataframe
    train_test_df = pd.read_csv(
        train_test_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS
    )
    infer_df = pd.read_csv(INFERENCE_PATH, encoding=DATASET_ENCODING)

    # Clean and tokenize dataset
    train_test_df.text = train_test_df.text.apply(lambda x: preprocess(x))
    infer_df.trainscript = infer_df.transcript.apply(lambda x: preprocess(x))

    # Replace 4 with 1
    train_test_df.target.replace(4, 1, inplace=True)

    # Split into train and test sets
    train_X, test_X, train_y, test_y = train_test_split(
        train_test_df.text, train_test_df.target, test_size=0.25
    )

    # Use keras tokenizer to index tokens
    indexer = tf.keras.preprocess.text.Tokenizer()
    indexer.fit_on_text(train_X.tolist())

    # TODO: find a way to match tokens to a pretrained word emebdding.
    train_X = None
    train_y = (None,)
    test_X = (None,)
    test_y = None

    return (train_X, train_y), (test_X, test_y), infer_X
