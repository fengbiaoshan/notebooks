
# coding: utf-8

# # 自定义的RNNCell


import tensorflow as tf


# ## 模仿基本的LSTM  
# Generating sequences with recurrent neural networks. Grave. 2013


class diyLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, activation=None, reuse=None):
        super(diyLSTMCell, self).__init__(_reuse=reuse)
        self._activation = activation or tf.tanh
        self._num_units = num_units
    
    @property
    def state_size(self):
        return self._num_units*2
    
    @property
    def output_size(self):
        return self._num_units
    
    def call(self, inputs, state):
        prec, preh = tf.split(state, 2, axis=1)
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope) as outer_scope:
            weights = tf.get_variable(
                "weights", [inputs.shape[1]+preh.shape[1], 4*self._num_units],
                dtype=inputs.dtype,
                initializer=tf.truncated_normal_initializer(-0.1,0.1))
            with tf.variable_scope(outer_scope) as inner_scope:
                inner_scope.set_partitioner(None)
                biases = tf.get_variable(
                    "bias", [4*self._num_units],
                    dtype=inputs.dtype,
                    initializer=tf.constant_initializer(0.0, dtype=inputs.dtype))
        m = tf.matmul(tf.concat((inputs,preh),1), weights) + biases
        i,f,o,g = tf.split(m,4,axis=1)
        newc = tf.sigmoid(f)*prec + tf.sigmoid(i)*self._activation(g)
        newh = tf.sigmoid(o)*self._activation(newc)
        new_state = tf.concat((newc, newh), 1)
        return newh, new_state
        


