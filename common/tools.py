import os
import pandas as pd
import numpy as np
import logging
FILLNA_VALUE = -99

def split_dataset(df, label):
    logging.info('splitting dataset ...')
    logging.info(f'fillna={FILLNA_VALUE}')
    df = df.fillna(FILLNA_VALUE)
    df_train = df.sample(frac=0.8, random_state=23)
    df_test = df.drop(df_train.index)

    X_train, y_train = cut_columns(df_train, label)
    X_test, y_test = cut_columns(df_test, label)

    return (X_train, y_train), (X_test, y_test)


def cut_columns(df, label):
    X = df[[i for i in df.columns.tolist() if i not in [label]]] #训练集
    y = df[label]
    return X, y

def data_split(data_source, num_processes):
    '''数据分块'''
    sec = int(num_processes)
    split_dfs = np.array_split(data_source, sec)
    return split_dfs


def convert2int(lst: list):
    return [int(i) for i in lst if i != '']


def dict_get_key(dt: dict, key: str):
    value = dt.get(key)
    if value is None and key != 'default':
        logging.warning(f'cannot find key:{key}, set default value')
        return dict_get_key(dt, 'default')
    logging.debug(f'dict_get_key: {key} -> {value}')
    return value


def create_pd_dict(df, key, value):
    '''通过dataframe中的两列生成字典'''
    return df.set_index(key)[value].to_dict()


def load_cache(path):
    logging.info('loading cache ...')
    if path is None:
        logging.error('path is None')
        return None
    if os.path.exists(path):
        logging.info(f'loading cache from: {path}')
        return pd.read_csv(path)
    logging.info('cache not exists')
    return None


def save_data(df, path):
    logging.info(f'saving data to {path}')
    logging.info(f'tot: {len(df)}')
    if path is None:
        logging.warning('file not saved!')
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        logging.error(e)