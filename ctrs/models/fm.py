#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('..')
import tensorflow as tf
from ..layers import layer


# In[ ]:


class FM(tf.keras.models.Model):
    def __init__(self, sparse_feature_columns, dense_feature_columns, task='binary', **kwarg):
        super(FM, self).__init__(**kwarg)
        self.linear1 = layer.LinearLayer()
        self.fm_layers = [
            layer.FMLayer(seed=idx, **kwarg)
            for idx, feat_col in enumerate(sparse_feature_columns+dense_feature_columns)
        ]
        self.sp_feat_layers = [
            tf.keras.layers.DenseFeatures(feat_col)
            for feat_col in sparse_feature_columns+dense_feature_columns
        ]
        self.dense_feature_layer = tf.keras.layers.DenseFeatures(dense_feature_columns)
        self.out_layer = layer.PredictionLayer(task)

    def call(self, inputs, **kwarg):
        fm_logits = []
        sparse_inputs = inputs[0]
        dense_inputs = inputs[1]
        total_inputs = {**sparse_inputs, **dense_inputs}

        for idx, (key, val) in enumerate(total_inputs.items()):
            sp_out = self.sp_feat_layers[idx]({key: val})
            fm_logit = self.fm_layers[idx](sp_out)
            fm_logits.append(fm_logit)
        fm_tensors = tf.keras.layers.add(fm_logits)
        dense_out = self.dense_feature_layer(total_inputs)
        linear_logits = self.linear1(dense_out)
        final_logits = tf.keras.layers.add([linear_logits, fm_tensors])
        return self.out_layer(final_logits)

    def _callbacks(self):
        self.callbacks = []
        if tensorboard:
            self.log_dir = "jupyter/logs/{}_".format(
                self._name, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                                       histogram_freq=1,
                                                                       embeddings_freq=1)
            self.callbacks.append(self.tensorboard_callback)
        if early_stopping:
            self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='binary_crossentropy',
                                                                   min_delta=0, patience=10,
                                                                   verbose=0, mode='auto', baseline=None,
                                                                   restore_best_weights=False)
            self.callbacks.append(self.early_stopping)

    def fit(self, **kwargs):
        self._callbacks()
        super(FM, self).fit(callbacks=self.callbacks, **kwargs)

