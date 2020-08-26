import numpy as np
import pandas as pd
import tensorflow as tf

def df_to_dataset(dataframe, sparse_feature, dense_feature, labels=None, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    df_sparse = dataframe[sparse_feature]
    df_dense = dataframe[dense_feature]

    if labels is None and not 'target' in dataframe:
        ds = tf.data.Dataset.from_tensor_slices(((dict(df_sparse), dict(df_dense)),))
    elif 'target' in dataframe:
        labels = dataframe.pop('target')
        ds = tf.data.Dataset.from_tensor_slices(((dict(df_sparse), dict(df_dense)), labels))
    elif labels is not None:
        ds = tf.data.Dataset.from_tensor_slices(((dict(df_sparse), dict(df_dense)), labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    if batch_size:
        ds = ds.batch(batch_size)
    return ds

def make_feature_column(dataframe, sparse_feature, dense_feature, use_emb=True, emb_dim=4, l2_reg_emb=0.0001):
    sparse_feature_columns = []
    dense_feature_columns = []
    sparse_feature_emb_columns = []
    for feat in sparse_feature:
        fc = tf.feature_column.categorical_column_with_vocabulary_list(feat, list(dataframe[feat].unique()))
        sparse_feature_columns.append(fc)

    for feat in dense_feature:
        fc = tf.feature_column.numeric_column(feat)
        dense_feature_columns.append(fc)

    if use_emb:
        for fc in sparse_feature_columns:
            emb_fc = tf.feature_column.embedding_column(fc, dimension=emb_dim, max_norm=l2_reg_emb)
            sparse_feature_emb_columns.append(emb_fc)
        return sparse_feature_emb_columns, dense_feature_columns

    return sparse_feature_columns, dense_feature_columns