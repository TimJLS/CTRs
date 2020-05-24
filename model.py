#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath('')))

import json
import copy
import joblib
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import lightgbm as lgb
import hyperopt as hpt
import dataset


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


class Model():
    def __init__(self):
        pass

    def train(self):
        pass

    def cv(self):
        pass

    def re_fit(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass

    def load(self, model_file=None):
        if model_file is None:
            if hasattr(self, 'model_file'):
                model_file = self.model_file
            else:
                raise ValueError
        self.model = joblib.load(model_file)

    def dump(self, name, model_file=None):
        if not str(name).endswith(".model"):
            name = str(name) + ".model"
        if model_file is None:
            if hasattr(self, 'model_file'):
                model_file = self.model_file
            else:
                raise ValueError
        model_file = model_file + '/'+ str(name)
        joblib.dump(self.model, model_file)

    def history(self):
        pass


# In[ ]:


class Tunner(dict):
    def __init__(self, train_set, objective, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.params = params

    def __repr__(self):
        return "<{0}> {1}".format(
            self.__class__.__name__,
            json.dumps(
                self.export_value(self.__dict__),
                sort_keys=True,
                indent=4,
                separators=(',', ': '),
            ),
        )

    def export_value(self, data):
        if isinstance(data, dict):
            data = dict((k, self.export_value(v))
                        for k, v in data.items()
                        if not k.startswith('_'))
        elif isinstance(data, list):
            data = [self.export_value(v) for v in data]
        return data


# In[ ]:


class LGBMTunner(dict):
    def __init__(self, objective, train_set, params, label=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        if isinstance(train_set, dataset.Dataset):
            if label is not None:
                self.train_set = lgb.Dataset(train_set._data, label)
            elif train_set._label is not None:
                self.train_set = lgb.Dataset(train_set._data, train_set._label)
            else:
                self.train_set = lgb.Dataset(train_set._data)

        elif isinstance(train_set, lgb.Dataset):
            self.train_set = train_set
        self.params = params
        self.objective = objective
        self.trials = hpt.Trials()

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
                        if not k.startswith('_'))
        elif isinstance(data, list):
            data = [self.export_value(v) for v in data]
        return data

    def full_objective(self, params):
        if 'nfold' not in params:
            result = self.objective(params, self.train_set, nfold=10)
        else:
            result = self.objective(params, self.train_set)
        best_score = np.max(result['auc-mean'])
        loss = 1 - best_score
        n_estimators = int(np.argmax(result['auc-mean']) + 1)
        return {'loss': loss, 'params': params,
                'estimators': n_estimators, 
                'status': hpt.STATUS_OK}

    def start(self, max_evals=100):
        best = hpt.fmin(fn=self.full_objective,
                        space=self.params,
                        algo=hpt.tpe.suggest,                      
                        max_evals=max_evals, 
                        trials=self.trials,
                        rstate=np.random.RandomState(27))
        return best

    def show_best(self):
        return pd.DataFrame(self.trials.best_trial).result.params


# In[ ]:


class LGBModel(Model):
    def __init__(self, params=None, model_file=None, silent=False):
        """Initialize the LightGBM Booster.

        Parameters
        ----------
        params : dict or None, optional (default=None)
            Parameters for Booster.
        train_set : Dataset or None, optional (default=None)
            Training dataset.
        model_file : string or None, optional (default=None)
            Path to the model file.
        model_str : string or None, optional (default=None)
            Model will be loaded from this string.
        """
        self.params = params
        self.trials = hpt.Trials()
        self.model_file = model_file

    def train(self, train_set, params=None, num_boost_round=100,
              valid_sets=None, valid_names=None, *arg, **kwargs):
        if params is None and self.params is None:
            raise ValueError
        elif params:
            self.params = copy.deepcopy(params)

        if valid_sets is not None:
            if isinstance(valid_sets, dataset.Dataset):
                valid_sets = [lgb.Dataset(valid_set._data, valid_set._label)]
            elif isinstance(valid_sets, lgb.Dataset):
                valid_sets = [valid_sets]
            elif not isinstance(valid_sets, list):
                raise TypeError
            for i, valid_set in enumerate(valid_sets):
                if isinstance(valid_set, dataset.Dataset):
                    valid_sets[i] = lgb.Dataset(valid_set._data, valid_set._label)
                if isinstance(valid_set, lgb.Dataset):
                    valid_sets[i] = valid_set

        if 'nfold' in self.params:
            self.params.pop('nfold', None)
        if 'num_boost_round' in self.params:
            self.params.pop('num_boost_round', None)

        if isinstance(train_set, dataset.Dataset):
            train_set.construct()
            self.train_set = lgb.Dataset(train_set._data, train_set._label)
        elif isinstance(train_set, lgb.Dataset):
            self.train_set = train_set
        else:
            raise TypeError
        self.model = lgb.train(params=self.params,
                               train_set=self.train_set,
                               valid_sets=valid_sets,
                               valid_names=valid_names,
                               *arg, **kwargs)
        return self.model

    def predict(self, data):
        if isinstance(data, lgb.Dataset):
            raise TypeError
        elif isinstance(data, dataset.Dataset):
            data = data._data
        predicts = self.model.predict(data)
        return predicts


# In[ ]:


def pipeline_test():
    ## construct dataset
    df = dataset.get_train_lookup()
    label = df.pop('is_over_kpi')
    categorical_feature = ['industry', 'type', 'is_avg_over_kpi']
    params = {
        'numerical_cols': ['kpi_value', 'campaign_period', 'avg_spend_cap', 'avg_cpm', 'avg_ctr',
                           'day_spend', 'audience_size'],
        'scale_cols': ['kpi_value', 'campaign_period', 'avg_spend_cap', 'avg_cpm', 'avg_ctr',
                       'day_spend', 'audience_size'],
        'label_encode_cols': ['interest_id']
    }
    trainset = dataset.Dataset(df, label=label, categorical_feature=categorical_feature, params=params)
    trainset.construct()
    # split into train, eval, test
    trainset, validset, testset = trainset.train_eval_test_split(eval_size=0.3,
                                                                 test_size=0.3,
                                                                 random_state=27)
    trainset.construct(), validset.construct(), testset.construct()

    # parameter tuning
    space = {
        'boosting_type': hpt.hp.choice('boosting_type', ['gbdt', 'goss']),
        'num_leaves': hpt.hp.choice('num_leaves', np.arange(2, 15+1, dtype=int)),
        'max_depth': hpt.hp.choice('max_depth', np.arange(2, 10+1, dtype=int)),
        'learning_rate': hpt.hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'seed': 27,
        'nfold': 10,
        'metrics': 'auc',
        'num_boost_round': 1000,
        'early_stopping_rounds': 100,
    }

    tuner = LGBMTunner(objective=lgb.cv,
                       train_set=trainset,
                       params=space)
    tuner.start(max_evals=10)

    ## use best params of tunner to train a lgb booster
    best_hp = tuner.show_best()
    best_hp['objective'] = 'binary'
    best_m = LGBModel(params=best_hp)

    best_m.train(trainset,
                 params=best_hp,
                 valid_sets=[trainset, validset],
                 valid_names=['train', 'eval'],
                 num_boost_round=100)
    best_m.dump('test', model_file='ind_lgb_3/')
    # predict using testset
    predicts = best_m.predict(testset)
    predictions = pd.DataFrame({'class': predicts})
    predictions['class'] = predictions['class'].apply(lambda x : 1 if x >= 0.5 else 0)
#     f1_score(test_label.values, predictions)
    return predictions, best_m, testset


# In[ ]:


# !jupyter nbconvert --to script model.ipynb

