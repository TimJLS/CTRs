#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath('')))

import json
import joblib
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy import sql
from sqlalchemy import Table, MetaData
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# In[ ]:


def dump_scaler(obj, name, file_path=None):
    if not str(name).endswith(".scaler"):
        name = str(name) + ".scaler"
    if file_path is None:
        file_path = os.path.abspath('')
    full_path = file_path + '/'+ str(name)
    joblib.dump(obj, full_path)

def dump_encoder(obj, name, file_path=None):
    if not str(name).endswith(".encoder"):
        name = str(name) + ".encoder"
    if file_path is None:
        file_path = os.path.abspath('')
    full_path = file_path + '/'+ str(name)
    joblib.dump(obj, full_path)


# In[ ]:


class Dataset(object):
    def __init__(self, data: pd.DataFrame, label=None, params=None,
                 categorical_feature=[], scaler_path=None, encoder_path=None):
        self.handle = None
        self._data = None
        self._label = None
        self.data = data.dropna()
        self.label = label
        self.categorical_feature = categorical_feature
        self.params = params
        if params is None:
            raise ValueError("Specify params")
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        self.is_construct = False

    def __repr__(self):
        return "<{0}> {1}".format(
            self.__class__.__name__,
            json.dumps(
                self.export_value(self.params),
                sort_keys=True,
                indent=4,
                separators=(',', ': '),
            ),
        )

    def export_value(self, data):
        if isinstance(data, dict):
            data = dict((k, self.export_value(v))
                        for k, v in data.items()
                        if 'data' not in k)
        elif isinstance(data, list):
            data = [self.export_value(v) for v in data]
        elif isinstance(data, pd.DataFrame):
            data = data.to_dict()
        elif isinstance(data, pd.Series):
            data = data.to_dict()
        elif isinstance(data, MinMaxScaler):
            data = data.scale_.tolist()
        elif isinstance(data, LabelEncoder):
            data = data.classes_.tolist()
        return data

    def get_numerical_cols(self):
        numerical_subset = []
        if self.params is not None:
            if 'numerical_cols' in self.params:
                numerical_subset = self.params['numerical_cols']
                assert type(numerical_subset)==list, "numerical_cols should be list."
            elif 'scale_cols' in self.params:
                numerical_subset = self.params['scale_cols']
                assert type(numerical_subset)==list, "numerical_cols should be list."
        return numerical_subset

    def get_scale_cols(self):
        scale_subset = []
        if self.params is not None:
            if 'scale_cols' in self.params:
                scale_subset = self.params['scale_cols']
                assert type(scale_subset)==list, "scale_cols should be list."

            elif 'numerical_cols' in self.params:
                scale_subset = self.params['numerical_cols']
                assert type(scale_subset)==list, "scale_cols should be list."
        return scale_subset

    def get_label_encode_cols(self):
        label_encode_subset = []
        if self.params is not None:
            if 'label_encode_cols' in self.params:
                label_encode_subset = self.params['label_encode_cols']
                assert type(label_encode_subset)==list, "label_encode_cols should be list."
        return label_encode_subset

    def get_field(self, field_name):
        if field_name == 'label':
            if field_name in self.data.columns:
                self.label = self.data.pop(field_name)
            return self.label
        elif field_name in self.data:
            return self.data[field_name]
        else:
            raise ValueError("Unknown field_name")

    def get_label(self):
        if self.label is None:
            self.label = self.get_field('label')
        return self.label

    def set_field(self, field_name, data):
        if field_name == 'label':
            self.label = data
        else:
            self.data[field_name] = data
        return self

    def set_label(self, label):
        if label is not None:
            self.set_field('label', label)
            self.label = self.get_field('label')
            self._label = self.label
        return self

    def transform_fields(self, trans_type=None):
        if trans_type == 'numerical_cols':
            numerical_cols = self.get_numerical_cols()
            self._data[numerical_cols] = self.data[numerical_cols].apply(pd.to_numeric,
                                                                         errors='ignore')

        elif trans_type == 'scale_cols':
            scale_cols = self.get_scale_cols()
            if self.scaler_path is not None:
                self.scaler = joblib.load(self.scaler_path)
                self._data[scale_cols] = pd.DataFrame(
                    self.scaler.inverse_transform(self.data[scale_cols]), columns=scale_cols)
            else:
                self.scaler = MinMaxScaler()
                self._data[scale_cols] = pd.DataFrame(
                    self.scaler.fit_transform(self.data[scale_cols]), columns=scale_cols)

        elif trans_type == 'categorical_cols':
            categorical_cols = self.categorical_feature
            dummies = pd.get_dummies(self.data[categorical_cols])
            self._data = pd.concat([self._data, dummies], axis=1)

        elif trans_type == 'label_encode_cols':
            label_encode_cols = self.get_label_encode_cols()
            if self.encoder_path is not None:
                self.label_encoder = joblib.load(self.encoder_path)
                self._data[label_encode_cols] = pd.DataFrame(
                    self.label_encoder.inverse_transform(self.data[label_encode_cols]))
            else:
                for col in label_encode_cols:
                    self.label_encoder = LabelEncoder()
                    self._data[col] = pd.DataFrame(
                        self.label_encoder.fit_transform(self.data[col]))
        return self

    def full_transform(self):
        if self._data is None:
            self._data = pd.DataFrame()

        if self.params is not None:
            if 'numerical_cols' in self.params:
                self.transform_fields(trans_type='numerical_cols')
                
            if 'scale_cols' in self.params:
                self.transform_fields(trans_type='scale_cols')

            if 'categorical_cols' in self.params:
                self.transform_fields(trans_type='categorical_cols')

            if 'label_encode_cols' in self.params:
                self.transform_fields(trans_type='label_encode_cols')
        return self

    def _lazy_init(self, data, label=None, params=None,
                   categorical_feature=None):
        if data is None:
            return self
        params = {} if params is None else params
        if isinstance(categorical_feature, list) and categorical_feature:
            categorical_feature = pd.Index(categorical_feature)
            if categorical_feature.isin(self.data.columns).all():
                self.params['categorical_cols'] = list(categorical_feature)

        if label is not None:
            self.set_label(label)
        self.get_label()
        self.full_transform()

    def construct(self):
        if self._data is None:
            self._lazy_init(self.data, label=self.label, params=self.params,
                            categorical_feature=self.categorical_feature)
        self.is_construct = True
        return self

    def train_eval_test_split(self, eval_size, test_size, random_state=27):
        if not self.is_construct:
            self.construct()
        n = len(self.data)
        e = round(n * eval_size)
        t = round(n * test_size)
        assert eval_size+test_size < 1, "size bigger than 1"
        if n == 0:
            raise ValueError
        elif eval_size == 0 and test_size == 0:
            raise ValueError

        np.random.seed(random_state)

        full_idxs = range(n)
        eval_idx = np.random.choice(full_idxs, size=e, 
                                    replace=False)
        full_idxs = list(set(full_idxs) - set(eval_idx))
        test_idx = np.random.choice(full_idxs, size=t,
                                    replace=False)
        full_idxs = list(set(full_idxs) - set(test_idx))

        tpl = ()

        for idxs in [full_idxs, eval_idx, test_idx]:
            dataset = Dataset(pd.DataFrame(), params=self.params,
                              categorical_feature=self.categorical_feature)
            for k, v in dataset.__dict__.items():
                attr = self.__dict__[k]
                if k in ['data', '_data', 'label', '_label']:
                    dataset.__dict__[k] = attr[attr.index.isin(idxs)]
            tpl += (dataset,)
        return tpl


# In[ ]:


# !jupyter nbconvert --to script dataset.ipynb


# In[ ]:




