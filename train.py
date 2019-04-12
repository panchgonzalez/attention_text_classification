"""Train sentiment classifier."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import argparse

from .model import model
from .model import model_params
from .utils import data_utils


def main(argv):
    """Main entry point."""

    # Parse arguments
    args = parser.parse_args(argv[1:])

    # Get the base parameters
    params = model_params.BASE_PARAMS

    # Get the data
    train_X, train_y, test_X, test_y, inf_text, tokenizer = data_utils.get_data()


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
