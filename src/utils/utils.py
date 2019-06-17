"""Generally useful utility functions."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import yaml
import pickle
from munch import munchify


def load_config(config_file):
    """Load YAML config file and convert to Munch object for attribute-style access

    Args:
        config_file: path to config.yml file

    Returns:
        Munch object containing configurations
    """
    # Load yaml file
    with open(config_file, "r") as f:
        cfg = yaml.load(f)

    # Convert cfg dictionary to Munch dictionary
    cfg = munchify(cfg)

    return cfg
