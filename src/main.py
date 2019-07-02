"""Train and evaluate sentiment classifier."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pickle
import numpy as np
import tensorflow as tf

from .model import model as classifier
from .utils import data_utils
from .utils import utils


def model_fn(features, labels, mode, params):
    """Defines EstimatorSpec passed to Estimator."""

    if isinstance(features, dict):  # For serving
        features = features["feature"]

    # Create model and get output logits and attention weights
    model = classifier.Model(params, mode == tf.estimator.ModeKeys.TRAIN)
    logits, alphas = model(inputs=features)

    ## Predict
    probabilities = tf.nn.sigmoid(logits)
    predicted_classes = tf.round(probabilities)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "class_ids": predicted_classes,
            "probabilities": probabilities,
            "logits": logits,
            "alphas": alphas,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss
    loss = tf.losses.sigmoid_cross_entropy(tf.expand_dims(labels, -1), logits=logits)

    # Compute evaluation metrics (optionally add summary)
    # NOTE: add another useful metric like recall
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predicted_classes, name="acc_op"
    )
    metrics = {"accuracy": accuracy}
    tf.summary.scalar("accuracy", accuracy[1])

    ## Evaluate
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    ## Train
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    # Define train op
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train_and_eval(estimator, params):
    """Train and evaluate estimator."""

    # Fetch the data
    train_X, train_y, test_X, test_y, _, _ = data_utils.get_data(params)

    # Define train and eval spec
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: data_utils.input_fn(
            features=train_X,
            labels=train_y,
            batch_size=params.batch_size,
            buffer_size=1_200_000,
        ),
        max_steps=params.train.steps,
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: data_utils.input_fn(
            features=test_X, labels=test_y, batch_size=1024, buffer_size=400_000
        ),
        steps=params.eval.steps,
        start_delay_secs=params.eval.start_delay_secs,
        throttle_secs=params.eval.throttle_secs,
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def predict(estimator, params):
    """Make predictions."""

    # Fetch the data
    _, _, _, _, inf_text, _ = data_utils.get_data(params)

    # Make predictions on call transcripts
    print("\nMaking predictions on inference set")
    predictions = estimator.predict(
        input_fn=lambda: data_utils.input_fn(
            features=inf_text, labels=None, batch_size=1024, buffer_size=5000
        )
    )

    # Save after this
    print("\nSaving infer predictions")
    class_ids, probabilities, alphas = [], [], []
    for pred_dict in predictions:
        class_ids.append(pred_dict["class_ids"])
        probabilities.append(pred_dict["probabilities"])
        alphas.append(pred_dict["alphas"])

    with open(os.path.join(params.pred_dir, "inf_pred.pkl"), "wb") as f:
        pickle.dump((class_ids, probabilities, alphas), f)


def main(argv):
    """Main entry point."""

    # parse arguments
    args = parser.parse_args(argv[1:])

    # Get the base parameters
    params = utils.load_config("src/config.yml")

    # Fetch the data
    _, _, _, _, _, tokenizer = data_utils.get_data(params)

    # Add tokenizer to params
    params.tokenizer = tokenizer

    # Create classifier
    print("\nBuilding model")

    # Make model directory path
    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)

    # Build model
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, params=params, model_dir=params.model_dir
    )

    # Train and eval, or predict
    if args.train_and_eval:
        train_and_eval(estimator, params)

    if args.predict:
        predict(estimator, params)


if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_and_eval", dest="train_and_eval", action="store_true")
    parser.add_argument("--predict", dest="predict", action="store_true")

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
