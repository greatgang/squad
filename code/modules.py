# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


class SelfAttn(object):
    """
    Module for self attention.
    """

    def __init__(self, keep_prob, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("SelfAttn"):

            # source_sequence_length = tf.reduce_sum(values_mask, reduction_indices = 1)
            v1 = tf.layers.dense(values, self.value_vec_size, use_bias=False)
            v2 = tf.layers.dense(values, self.value_vec_size, use_bias=False)

            v = tf.get_variable("v_attention", shape=[self.value_vec_size], 
                initializer=tf.contrib.layers.xavier_initializer())

            reshaped_v1 = tf.expand_dims(v1, 1)
            reshaped_v2 = tf.expand_dims(v2, 2)

            self_attn_logits = tf.reduce_sum(v * tf.tanh(reshaped_v1 + reshaped_v2), 3)

            self_attn_logits_mask = tf.expand_dims(values_mask, 1) # (batch_size, 1, num_values)
            _, self_attn_dist = masked_softmax(self_attn_logits, self_attn_logits_mask, 2) # (batch_size, num_values, num_values)

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(self_attn_dist, values) # shape (batch_size, num_values, value_vec_size)

            # Apply dropout
            # output = tf.nn.dropout(output, self.keep_prob)

            return self_attn_dist, output


class DotAttn(object):
    """
    Module for self attention.
    """

    def __init__(self, keep_prob, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("DotAttn"):

            #v1 = tf.layers.dense(values, self.value_vec_size, activation=tf.nn.relu, use_bias=False, name="W1")
            #v2 = tf.layers.dense(values, self.value_vec_size, activation=tf.nn.relu, use_bias=False, name="W2")
            v1 = tf.layers.dense(values, self.value_vec_size, use_bias=False, name="W1")
            v2 = tf.layers.dense(values, self.value_vec_size, use_bias=False, name="W2")

            #self_attn_logits = tf.matmul(v1, 
            #                   tf.transpose(v2, [0, 2, 1]) / np.sqrt(self.value_vec_size))
            self_attn_logits = tf.matmul(v1, tf.transpose(v2, [0, 2, 1]))

            self_attn_logits_mask = tf.expand_dims(values_mask, 1) # (batch_size, 1, num_values)
            _, self_attn_dist = masked_softmax(self_attn_logits, self_attn_logits_mask, 2) # (batch_size, num_values, num_values)

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(self_attn_dist, values) # shape (batch_size, num_values, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return self_attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist

def test_self_attn_layer():
    with tf.Graph().as_default():
        with tf.variable_scope("test_self_attn_layer"):
            # key_placeholder is shape (batch_size, context_len, hidden_size*2)
            value_placeholder = tf.placeholder(tf.float32, shape=[1, 3, 2])
            value_mask_placeholder = tf.placeholder(tf.float32, shape=[1, 3])

            """
            with tf.variable_scope("rnn"):
                tf.get_variable("W_x", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("W_h", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b",  initializer=np.array(np.ones(2), dtype=np.float32))
            """

            #tf.get_variable_scope().reuse_variables()
            self_attn_layer = SelfAttn(1, 2)
            dist, self_attn_output = self_attn_layer.build_graph(value_placeholder, value_mask_placeholder) 
            #print self_attn_output.get_shape()
            print "self attn distribution shape = " + str(np.shape(dist))
            print "self attn output shape = " + str(np.shape(self_attn_output))

            """
            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                x = np.array([
                    [0.4, 0.5, 0.6],
                    [0.3, -0.2, -0.1]], dtype=np.float32)
                h = np.array([
                    [0.2, 0.5],
                    [-0.3, -0.3]], dtype=np.float32)
                y = np.array([
                    [0.832, 0.881],
                    [0.731, 0.622]], dtype=np.float32)
                ht = y

                y_, ht_ = session.run([y_var, ht_var], feed_dict={x_placeholder: x, h_placeholder: h})
                print("y_ = " + str(y_))
                print("ht_ = " + str(ht_))

                assert np.allclose(y_, ht_), "output and state should be equal."
                assert np.allclose(ht, ht_, atol=1e-2), "new state vector does not seem to be correct."
            """

def test_dot_attn_layer():
    print "Test dot attention layer:"
    with tf.Graph().as_default():
        with tf.variable_scope("test_dot_attn_layer"):
            value_placeholder = tf.placeholder(tf.float32, shape=[None, 3, 2]) # (batch_size, context_len, hidden_size*2)
            value_mask_placeholder = tf.placeholder(tf.float32, shape=[None, 3])

            with tf.variable_scope("DotAttn"):
                tf.get_variable("W1/kernel", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("W2/kernel", initializer=np.array(np.eye(2,2), dtype=np.float32))

            tf.get_variable_scope().reuse_variables()
            dot_attn_layer = DotAttn(1, 2) # (keep_prob, context_len)

            dist, dot_attn_output = dot_attn_layer.build_graph(value_placeholder, value_mask_placeholder) 
            print "Trainable variables: "
            print tf.trainable_variables()
            print "dot attn distribution shape = " + str(np.shape(dist))
            print "dot attn output shape = " + str(np.shape(dot_attn_output))

            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                v = np.array([
                    [[0.4,  0.5], # batch 0
                     [0.3, -0.2],
                     [0.6, -0.1]],
                    [[4, -5],     # batch 1
                     [8,  2],
                     [9, -1]]
                    ], dtype=np.float32)
                m = np.array([[1, 1, 0], # batch 0
                              [1, 0, 0]  # batch 1
                             ], dtype=np.float32)
                dist_, attn_ = session.run([dist, dot_attn_output], 
                               feed_dict={value_placeholder: v, value_mask_placeholder: m})
                print("\ndist_ = ")
                print dist_
                print("\nattn_ = ")
                print attn_
                expected_attn_ = np.array([
                    [[0.35962827, 0.21739789],  # batch 0
                     [0.34725277, 0.13076939],
                     [0.34975   , 0.14825001]],
                    [[4, -5],                   # batch 1
                     [4, -5],
                     [4, -5]]
                    ], dtype=np.float32)
                assert np.allclose(attn_, expected_attn_, atol=1e-2), "attention not correct"


def do_test(_):
    print "Testing starts:"
    test_self_attn_layer()
    test_dot_attn_layer()
    print "Passed!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests modules')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

