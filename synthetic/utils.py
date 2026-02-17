import numpy as np
import random
import os

def empty_metrics(T, n_x, n_y):
    """Allocate an empty metric dict for one method."""
    zeros_x = np.zeros((T, n_x))
    zeros_y = np.zeros((T, n_y))
    return dict(
        match_x=zeros_x.copy(),  match_y=zeros_y.copy(),
        exposure_x=zeros_x.copy(), exposure_y=zeros_y.copy(),
        fair_x=zeros_x.copy(),    fair_y=zeros_y.copy(),
        active_users_x=zeros_x.copy(), active_users_y=zeros_y.copy(),
        user_retain_x=zeros_x.copy(), user_retain_y=zeros_y.copy(),
        true_user_retain_x=zeros_x.copy(), true_user_retain_y=zeros_y.copy(),
    )


def exam_func(n_items: int, K: int, shape: str = "exp") -> np.ndarray:
    assert shape in ["inv", "exp", "log", "uniform", "linear"]
    if shape == "inv":
        e = 1.0 / np.arange(1, n_items + 1)
    elif shape == "exp":
        e = 1.0 / np.exp(np.arange(n_items))
    elif shape == "log":
        e = 1.0 / np.log2(np.arange(n_items) + 2)
    elif shape == "uniform":
        e = np.ones(n_items)
    elif shape == "linear":
        e = np.linspace(1.0, 0.1, n_items)
    e[K:] = 0
    return e

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fix_seed(seed=12345):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def relative_by_policy(data, x, relative):
    if relative == None:
        return data
    logging_values = data[data['method'] == relative].set_index(['seed', x])

    numeric_cols = ['match_x', 'match_y','active_users_x', 'active_users_y','true_user_retain_x','true_user_retain_y']

    def relative_row(row):
        key = (row['seed'], row[x])        
        logging_row = logging_values.loc[key, numeric_cols]
        return row[numeric_cols] / logging_row.replace(0, float('inf')) 

    relative_data = data.copy()
    relative_data[numeric_cols] = data.apply(relative_row, axis=1)

    return relative_data
