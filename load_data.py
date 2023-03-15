import yaml
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def load_and_transform_data(config_path):
    config = read_params(config_path)
    raw_data = config['data_config']['raw_csv']
    num_cols = config['data_config']['num_cols']
    cat_cols = config['data_config']['cat_cols']
    target = config['data_config']['target']
    train_path = config['data_config']['train_csv']
    test_path = config['data_config']['test_csv']
    random_state = config['data_config']['random_state']

    raw = pd.read_csv(raw_data)
    raw = raw[num_cols+cat_cols+[target]]
    for i in num_cols:
        if raw[i].dtype != 'int64' and raw[i].dtype != 'float64':
            raw[i] = raw[i].str.replace('_', '')
        raw[i] = raw[i].astype('float')
        max_i = np.nanpercentile(raw[i], 98)
        min_i = np.nanpercentile(raw[i], 2)
        raw = raw[(raw[i].between(min_i, max_i)) & (raw[i] > 0)]
    raw[target] = raw[target].map({'Poor': 0, 'Standard': 1, 'Good': 2})
    train, test = train_test_split(raw, random_state=random_state, test_size=0.2)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_transform_data(config_path=parsed_args.config)
