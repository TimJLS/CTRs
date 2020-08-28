#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('..')
import tensorflow as tf
from layers import layer


# In[ ]:


class DeepFM(tf.keras.models.Model):
    def __init__(self, sparse_feature_columns, dense_feature_columns, task='binary', training=True, **kwarg):
        super(DeepFM, self).__init__(**kwarg)
        self.training = training
        self.dnn = layer.SimpleDNN(**kwarg)
        self.dense = tf.keras.layers.Dense(1)
        self.fm_layers = {
            feat_col.name.split('_embedding')[0]: layer.FMLayer(seed=idx,
                                                          name=feat_col.name.split('_embedding')[0],
                                                          **kwarg)
            for idx, feat_col in enumerate(sparse_feature_columns+dense_feature_columns)
        }
        self.sp_feat_layers = {
            feat_col.name.split('_embedding')[0]: tf.keras.layers.DenseFeatures(
                feat_col, name=feat_col.name.split('_embedding')[0])
            for feat_col in sparse_feature_columns+dense_feature_columns
        }
        self.dense_feature_layer = tf.keras.layers.DenseFeatures(sparse_feature_columns+dense_feature_columns)
        self.out_layer = PredictionLayer(task)
        
    def call(self, inputs, **kwarg):
        fm_logits = []
        total_inputs = inputs
        for idx, (key, val) in enumerate(total_inputs.items()):
            sp_out = self.sp_feat_layers[key]({key: val})
            fm_logit = self.fm_layers[key](sp_out)
            fm_logits.append(fm_logit)
        fm_tensors = tf.keras.layers.add(fm_logits)
        dense_out = self.dense_feature_layer(total_inputs)
        dnn_logits = self.dnn(dense_out)
        dnn_logits = self.dense(dnn_logits)
        final_logits = tf.keras.layers.add([dnn_logits, fm_tensors])
        return self.out_layer(final_logits)
    
    def _callbacks(self, early_stopping=True, tensorboard=False, **kwargs):
        self.callbacks = []
        if tensorboard:
            self.log_dir = "jupyter/logs/{}_".format(self._name)
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
        self._callbacks(**kwargs)
        super(DeepFM, self).fit(callbacks=self.callbacks, **kwargs)

