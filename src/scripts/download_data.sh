#!/bin/bash

# This script downloads and stores the necessary datasets and embedding files

# Download Sentiment140 twitter NLP dataset. This creates a directory containing
# both the training and test datasets. These will be moved to the more keras datasets
# cache directory in ~/.keras/datasets
wget https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip trainingandtestdata.zip -d ~/.keras/datasets
rm trainingandtestdata.zip

# Download a GloVe embedding pretrained using twitter data. We will also store this
# in the keras cache
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip -d ~/.keras/datasets
rm glove.twitter.27B.zip
