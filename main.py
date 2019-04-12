"""Train and evaluate sentiment classifier."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pickle
import numpy as np
import tensorflow as tf

from .model import model as classifier
from .model import model_params
from .utils import data_utils


def model_fn(features, labels, mode, params):
    """Defines EstimatorSpec passed to Estimator."""

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
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

    # Define train op
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    """Main entry point."""

    # Get the base parameters
    params = model_params.BASE_PARAMS

    # Fetch the data
    train_X, train_y, test_X, test_y, inf_text, tokenizer = data_utils.get_data(params)

    # Add tokenizer to params
    params["tokenizer"] = tokenizer

    # Create classifier
    print("\nBuilding model")

    # Make model directory path
    if not os.path.exists(params["model_dir"]):
        os.makedirs(params["model_dir"])

    # Build model
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, params=params, model_dir=params["model_dir"]
    )

    # Train the model
    print("\nTraining model")
    estimator.train(
        input_fn=lambda: data_utils.train_input_fn(
            features=train_X,
            labels=train_y,
            batch_size=params["batch_size"],
            buffer_size=1_200_000,
        ),
        steps=params["train_steps"],
    )

    # Evaluate the model.
    print("\nEvaluating model on test set")
    eval_result = estimator.evaluate(
        input_fn=lambda: data_utils.eval_input_fn(
            features=test_X,
            labels=test_y,
            batch_size=params["batch_size"],
            shuffle=False,
            buffer_size=400_000,
        )
    )

    print("\nTest set accuracy: {accuracy:0.3f}\n".format(**eval_result))

    # Make predictions on evaluation set
    print("\nMaking predictions on evaluation set")
    eval_predictions = estimator.predict(
        input_fn=lambda: data_utils.eval_input_fn(
            features=test_X,
            labels=None,
            batch_size=1024,
            shuffle=False,
            buffer_size=400_000,
        )
    )

    print("\nSaving eval predictions")

    # Check or create directory for predictions
    if not os.path.exists(params["pred_dir"]):
        os.makedirs(params["pred_dir"])

    class_ids, probabilities, alphas = [], [], []
    for pred_dict in eval_predictions:
        class_ids.append(pred_dict["class_ids"])
        probabilities.append(pred_dict["probabilities"])
        alphas.append(pred_dict["alphas"])

    with open(os.path.join(params["pred_dir"], "eval_pred.pkl"), "wb") as f:
        pickle.dump((class_ids, probabilities, alphas), f)

    # Make predictions on call transcripts
    print("\nMaking predictions on inference set")
    predictions = estimator.predict(
        input_fn=lambda: data_utils.eval_input_fn(
            features=inf_text,
            labels=None,
            batch_size=1024,
            shuffle=False,
            buffer_size=5000,
        )
    )

    # Save after this
    print("\nSaving infer predictions")
    class_ids, probabilities, alphas = [], [], []
    for pred_dict in predictions:
        class_ids.append(pred_dict["class_ids"])
        probabilities.append(pred_dict["probabilities"])
        alphas.append(pred_dict["alphas"])

    with open(os.path.join(params["pred_dir"], "inf_pred.pkl"), "wb") as f:
        pickle.dump((class_ids, probabilities, alphas), f)


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
