"""Define sentiment classifier model."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from ..utils import data_utils


class Model(object):
    """Sentiment classifier model."""

    def __init__(self, params, train):
        """Initialize layers to build model.

        Args:
            params: hyperparameter object defining layer sizes
            train: boolean indicating whether the model is in training mode.
        """
        # NOTE: train bool for adding dropout
        self.train = train
        self.params = params

        # Initialize word embedding
        self.embedding = self.init_embedding()

    def __call__(self, inputs):
        """Calculate target logits or inferred target logits.

        Args:
            inputs: int tensor with shape [batch_size, input_length]
        """
        # Define initializer
        initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=0)

        with tf.variable_scope("Model", initializer=initializer, reuse=tf.AUTO_REUSE):
            # Get embedded inputs
            embedded_inputs = tf.nn.embedding_lookup(self.embedding, inputs)

            # Run dynamic bidirectional rnn
            rnn_states = self.birnn(embedded_inputs)

            # Run attention mechanism on rnn_states
            attn_outputs, alphas = self.attention(rnn_states)

            # Run attention output through feed-forward network
            dense_1 = tf.layers.dense(attn_outputs, units=128, activation=tf.nn.relu)
            dense_2 = tf.layers.dense(dense_1, units=32, activation=tf.nn.relu)
            dense_3 = tf.layers.dense(dense_2, units=8, activation=tf.nn.relu)
            logits = tf.layers.dense(dense_3, units=1, activation=None)

        return logits, alphas

    def init_embedding(self):
        """Initialize word embedding matrix using pretrained embedding."""

        embedding = data_utils.get_embedding_matrix(self.params.tokenizer)

        with tf.name_scope("embedding"):
            embedding = tf.get_variable(
                name="embedding_matrix",
                shape=[self.params.vocab_size, self.params.embedding_size],
                initializer=tf.constant_initializer(embedding),
                dtype=tf.float32,
                trainable=False,
            )
        return embedding

    def birnn(self, inputs):
        """Wrapper for bidirectional rnn.

        Args:
            inputs: tensor of shape [batch_size, input_length]

        Returns:
            tensor of shape [batch_size, input_length, 2*hidden_size]
        """

        sequence_length = tf.tile([50], [tf.shape(inputs)[0]])
        with tf.name_scope("rnn"):
            # Construct cell with cell size num_rnn_units
            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params.hidden_size)

            # NOTE: bi_outputs would be used by attention mechanism
            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                cell, cell, inputs, dtype=tf.float32, sequence_length=sequence_length
            )

            bi_outputs = tf.concat((bi_outputs[0], bi_outputs[1]), -1)

        return bi_outputs

    def attention(self, inputs):
        """Attention mechanism

        Args:
            inputs: tensor of shape [batch_size, input_length, 2*hidden_size]

        Returns:
            attention_output: tensor of shape [batch_size, 2*hidden_size]
        """
        hidden_size = 2 * self.params.hidden_size
        attention_size = self.params.attention_size

        with tf.name_scope("attention"):

            # Trainable variables
            w_omega = tf.get_variable("w_omega", [hidden_size, attention_size])
            b_omega = tf.get_variable("b_omega", [attention_size])
            u_omega = tf.get_variable("u_omega", [attention_size])

            with tf.name_scope("v"):
                v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

            vu = tf.tensordot(v, u_omega, axes=1, name="vu")
            alphas = tf.nn.softmax(vu, name="alphas")

            output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        return output, alphas
