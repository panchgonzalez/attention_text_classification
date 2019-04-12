"""Analyze results."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix

DATASET_ENCODING = "ISO-8859-1"


def recall_and_f1(
    y_true, y_prob, p, r, t, precisions=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
):
    """Compute recall and f1 at list of precisions"""

    recall_scores = []
    f1_scores = []

    for precision in precisions:
        t_at_p = min([t[j] for j in range(len(t)) if p[j + 1] > precision])

        # Compute predictions at precision
        y_pred = [1 if prob > t_at_p else 0 for prob in y_prob]

        # Compute precision and recall
        recall = recall_score(y_true, y_pred)
        true_precision = precision_score(y_true, y_pred)
        f1 = 2 * true_precision * recall / (true_precision + recall)

        # Append to lists
        recall_scores.append(recall)
        f1_scores.append(f1)

    return recall_scores, f1_scores, precisions


def compute_precision_recall(y_true, y_prob):
    """Compute precision and recall and save figure"""

    # Compute precision, recall, and thresholds
    p, r, t = precision_recall_curve(y_true, y_prob)

    # Compute specific recall and f1 scores at discrete precisions
    recall_scores, f1_scores, precisions = recall_and_f1(y_true, y_prob, p, r, t)

    print("Recall and f1 scores at specific precisions")
    for recall, f1, precision in zip(recall_scores, f1_scores, precisions):
        print(f"p = {precision:0.3f}\tr = {recall:0.3f}\tf1 = {f1:0.3f}")

    # Save figure precison-recall curve
    fig, ax = plt.subplots(1, 1)
    ax.plot(r, p, lw=2)
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.axis("square")
    ax.axis([0, 1, 0, 1])
    plt.tight_layout()
    plt.savefig("../tmp/pr_curve.png", bbox_inches="tight")


def compute_confusion_matrix(y_true, y_pred):
    """Compute and save confusion matrix."""

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    print(f"\nConfusion matrix\n{cm}")

    # Plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=["0", "4"],
        yticklabels=["0", "4"],
        title="Confusion matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig("../tmp/cm.png", bbox_inches="tight")


def get_lowest_score(y_pred):
    """Get lowest score from predictions."""

    # Load base dataframe
    df = pd.read_csv("../data/calls-with-transcripts.csv", encoding=DATASET_ENCODING)

    # Append predicted score
    df["y_pred"] = y_pred

    # Keep only user and score
    df = df[["user", "y_pred"]]

    # Group by user and find average
    user_avg_score_df = df.groupby(["user"]).mean().y_pred.sort_values()

    print(f"\nLowest average scores {user_avg_score_df.head()}")
    return user_avg_score_df


if __name__ == "__main__":

    # Load predictions and saved clean data
    with open("../tmp/data.pkl", "rb") as f:
        _, _, _, test_y, inf_text, tokenizer = pickle.load(f)

    # Load eval predictions
    with open("../pred/eval_pred.pkl", "rb") as f:
        # (class_ids, probabilities, alphas)
        eval_results = pickle.load(f)
    eval_pred = np.squeeze(np.stack(eval_results[0]))
    eval_prob = np.stack(eval_results[1])

    print("Analyzing results on test set")
    compute_precision_recall(test_y, eval_prob)
    compute_confusion_matrix(test_y, eval_pred)

    # Load infer predictions
    with open("../pred/inf_pred.pkl", "rb") as f:
        # (class_ids, probabilities, alphas)
        inf_results = pickle.load(f)
    inf_pred = np.squeeze(np.stack(inf_results[0]))
    inf_prob = np.stack(inf_results[1])

    print("\nAnalyzing results on inference set")

    # Replace 1 with 4 in predictions
    inf_pred[np.where(inf_pred == 1)] = 4
    print(f"\nAverage sentiment is {np.mean(inf_pred)}")

    # Find
    user_avg_score_df = get_lowest_score(inf_pred)
