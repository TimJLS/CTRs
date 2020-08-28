#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[ ]:


class PredictionLayer(tf.keras.layers.Layer):
    def __init__(self, task='binary', use_bias=False, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(name="global_bias",
                                               initializer=initializers.Zeros(),
                                               shape=(1,))

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)
        output = tf.reshape(x, (-1, 1))
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[ ]:


class LinearLayer(tf.keras.layers.Layer):
    """The Linear Layer
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``.
          The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_units[-1])``.
          For instance, for a 2D input with shape ``(batch_size, input_dim)``,
          the output would have shape ``(batch_size, hidden_units[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **use_bias**: bool. Whether to add bias into layer. Default `True`.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength. Default 0.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout. Default 0., meaning no dropout.
        - **seed**: A Python integer to use as random seed.
    """
    
    def __init__(self, hidden_units, use_bias=True, dropout_rate=0., l2_reg=0., seed=27, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.seed = seed

    def build(self, input_shape, **kwargs):  # TensorShape of input when run call(), inference from inputs
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.kernel = self.add_weight(
            name="linear_kernel",
            shape=[input_shape[-1], self.hidden_units],
            initializer=tf.keras.initializers.glorot_normal(self.seed),
            regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="linear_bias",
                shape=[self.hidden_units],
                initializer=tf.keras.initializers.Zeros(),
            )
        # Be sure to call this somewhere!
        super(LinearLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        feats = self.dropout(inputs, training=training)
        return tf.matmul(feats, self.kernel) + self.bias


# In[ ]:


class FMLayer(tf.keras.layers.Layer):
    """The Factorization Machine Layer
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``.
          The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_units[-1])``.
          For instance, for a 2D input with shape ``(batch_size, input_dim)``,
          the output would have shape ``(batch_size, hidden_units[-1])``.
      Arguments
        - **l2_reg**: float between 0 and 1. L2 regularizer strength. Default 0.
        - **v**: interaction units. Default 10.
        - **seed**: A Python integer to use as random seed.
    """
    
    def __init__(self, l2_reg=0.0001, v=10, seed=27, **kwargs):
        self.v = v
        self.l2_reg = l2_reg
        self.seed = seed
        super(FM, self).__init__(**kwargs)
    
    def build(self, input_shape, **kwargs):
        if isinstance(input_shape, list):
            shape = [int(input_shape[1][-1]), self.v]
        else:
            shape = [int(input_shape[-1]), self.v]
        self.kernel = self.add_weight(name='fm_kernel',
                                      shape=shape,
                                      initializer=tf.keras.initializers.glorot_normal(self.seed),
                                      regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                      trainable=True)
        # Be sure to call this somewhere!
        super(FM, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        # inferences
        embedded_inputs = tf.matmul(inputs, self.kernel)
        self.embedded_inputs = inputs
        square_of_sum = tf.square(tf.reduce_sum(embedded_inputs, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(tf.pow(embedded_inputs, 2), axis=1, keepdims=True)

        self.square_of_sum = square_of_sum
        self.sum_of_square = sum_of_square
        
        cross_term = tf.reduce_sum((square_of_sum - sum_of_square), axis=1, keepdims=False)
        
        interaction_term = tf.multiply(0.5, cross_term)
        return interaction_term


# In[ ]:


class SimpleDNN(tf.keras.layers.Layer):
    """The Multi Layer Perceptron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``.
          The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_units[-1])``.
          For instance, for a 2D input with shape ``(batch_size, input_dim)``,
          the output would have shape ``(batch_size, hidden_units[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function. Default `relu`.
        - **kernel_initializer**: Initializer for the kernel weights matrix.
        - **bias_initializer**: Initializer for the bias vector.
        - **kernel_regularizer**: Regularizer function applied to the kernel weights matrix.
        - **bias_regularizer**: Regularizer function applied to the bias vector.
        - **activity_regularizer**: Regularizer function applied to the output of the layer (its "activation").
        - **l2_reg**: float between 0 and 1. L2 regularizer strength.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout. Default `None`, meaning no dropout.
        - **use_bn**: bool. Whether to use BatchNormalization before activation.
        - **seed**: A Python integer to use as random seed.
        - **use_bias**: bool. Whether to add bias into layer. Default `True`.
    """

    def __init__(
        hidden_units, activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, l2_reg=None, dropout_rate=None, seed=27, **kwargs
    ):
        if l2_reg:
            self.l2_reg = l2_reg
            kernel_regularizer = tf.keras.regularizers.l2(l2_reg)
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        super(SimpleDNN, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.denses = [
            tf.keras.layers.Dense(
                units, activation, use_bias, kernel_initializer, bias_initializer,
                kernel_regularizer, bias_regularizer, activity_regularizer)
            for units in hidden_units
        ]
        if self.use_bn:
            self.bn_layers = [
                tf.keras.layers.BatchNormalization()
                for _ in range(len(self.hidden_units))
            ]
        if self.dropout_rate:
            self.dropout_layers = [
                tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i)
                for i in range(len(self.hidden_units))
            ]
        super(SimpleDNN, self).build(input_shape)
        
    def call(self, inputs, training=None, **kwarg):
        input_tensors = inputs
        for idx, layer in enumerate(self.denses):
            out = layer(input_tensors)
            if self.use_bn:
                out = self.bn_layers[idx](out, training=training)
            if self.dropout_rate:
                out = self.dropout_layers[idx](out, training=training)
            deep_input = out
        return deep_input
    
    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape
        return tuple(shape)

    def get_config(self):
        config = {
            'activation': self.activation,
            'hidden_units': self.hidden_units, 
            'l2_reg': self.l2_reg,
            'use_bn': self.use_bn,
            'dropout_rate': self.dropout_rate,
            'seed': self.seed
        }
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

