# Data Science Problem

## Overview

Your company has heard that certain users making calls on their platform are making statements with very negative sentiment, and you've been tasked with finding out who they are.

The file `data/calls-with-transcripts.csv` lists a set of calls pulled from the company's database. It contains information about the users who were on the calls and a transcript of what they said. Each username is unique.

For this problem, you will be training a text classifier to classify the sentiment of users transcriptions to find out who the most negative users are.

You can use any library or techinque you wish to train your classifier. We have provided a training data corpus [here](https://storage.googleapis.com/gk-data/interview-problem/corpus.csv) based on human-labeled sentiment of tweets. The the first column of the training data is the sentiment label (0 for negative and 4 for positive), and the last column has the body of text in the tweet.

As your deliverable, please provide the following items either in Python scripts or in ANSWERS.md as appropriate.

1) Python script to train your sentiment text classifier. Document any and all package dependencies.

2) A confusion matrix and a report showing the precision, recall, and F1 score of your classifier.

3) Average sentiment for each of the users in the calls provided.

4) The user with the most negative sentiment.
