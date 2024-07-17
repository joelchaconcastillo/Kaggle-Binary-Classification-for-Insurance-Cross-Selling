##!pip install -U lightautoml[all]
import pandas as pd

train = pd.read_csv('../../input/train.csv')
train.head()

test = pd.read_csv('../../input/test.csv')
test.head()

train.info()

train.nunique().sort_values(ascending=False)

round(train['Response'].value_counts()*100/len(train), 2)

train.isnull().sum()


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('object')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

import numpy as np

train = reduce_mem_usage(train)
train.info()

test = reduce_mem_usage(test)
test.info()


import numpy as np
import pandas as pd
from sklearn.metrics import median_absolute_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim.lr.scheduler import ReduceLROnPlateau

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import task



def map_class(x, task, reader):
    if task.name == 'multiclass':
        return reader[x]
    else:
        return x

mapped = np.vectorize(map_class)

def score(task, y_true, y_pred):
    if task.name == 'binary':
        return roc_auc_score(y_true, y_pred)
    elif task.name == 'multiclass':
        return accuracy_score(y_true, np.argmax(y_pred, 1))
    elif task.name == 'reg' or task.name == 'multi:reg': ##regression
        return median_absolute_error(y_true, y_pred)
    else:
        raise 'Task is not correct.'
        
def take_pred_from_task(pred, task):
    if task.name == 'binary' or task.name == 'reg':
        return pred[:, 0]
    elif task.name == 'multiclass' or task.name == 'multi:reg':
        return pred
    else:
        raise 'Task is not correct.'
        
def use_plr(USE_PLR):
    if USE_PLR:
        return "plr"
    else:
        return "cont"

import os
RANDOM_STATE = 42
N_THREADS = os.cpu_count()

np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)

task = Task('binary')
automl = TabularAutoML(
    task = task,
    timeout = 9 * 3600,
    cpu_limit = os.cpu_count(),
    general_params = {"use_algos": [['fttransformer']]}, # ['nn', 'mlp', 'dense', 'denselight', 'resnet', 'snn', 'node', 'autoint', 'fttransformer'] or custom torch model
    nn_params = {
        "n_epochs": 10,
        "bs": 1024,
        "num_workers": 0,
        "path_to_save": None,
        "freeze_defaults": True,
        "cont_embedder": 'plr',
        'cat_embedder': 'weighted',
        "hidden_size": 32,
        'hid_factor': [3, 3],
        'block_config': [3, 3],
        'embedding_size': 32,
        'stop_by_metric': True,
        'verbose_bar': True,
        "snap_params": { 'k': 2, 'early_stopping': True, 'patience': 2, 'swa': True }
    },
    nn_pipeline_params = {"use_qnt": False, "use_te": False},
    reader_params = {'n_jobs': os.cpu_count(), 'cv': 5, 'random_state': 42, 'advanced_roles': True}
)

out_of_fold_predictions = automl.fit_predict(
    train,
    roles = {
        'target': 'Response',
    },
    verbose = 4
)

roc_auc_score(train.Response, out_of_fold_predictions.data)

y_pred = automl.predict(test).data[:, 0]
y_pred[:5]

df = pd.DataFrame(y_pred,columns=['Response'])
df.head()

#sol=pd.read_csv('/kaggle/input/playground-series-s4e7/sample_submission.csv')
sol=pd.read_csv('../../input/sample_submission.csv')
sol.head()

sol['Response']=df['Response']
sol.head()

sol.to_csv('./FTTransformer.csv',index=False)
