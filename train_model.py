import yaml
import joblib
import argparse
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def train_model(config_path):
    config = read_params(config_path)
    num_cols = config['data_config']['num_cols']
    cat_cols = config['data_config']['cat_cols']
    target = config['data_config']['target']
    train_path = config['data_config']['train_csv']
    test_path = config['data_config']['test_csv']
    depth = config['classifier']['depth']
    learning_rate = config['classifier']['learning_rate']
    iterations = config['classifier']['iterations']
    model_dir = config['model_dir']

    train = pd.read_csv(train_path)
    x_train, y_train = train.drop(columns=target), train[target]
    test = pd.read_csv(test_path)
    x_test, y_test = test.drop(columns=target), test[target]

    ct = ColumnTransformer([('one_hot_enc', OneHotEncoder(), cat_cols),
                            ('standard_scaler', StandardScaler(), num_cols)])
    pipe = Pipeline([('preprocess', ct),
                     ('classifier', CatBoostClassifier(iterations=iterations,
                                                       depth=depth,
                                                       learning_rate=learning_rate))])
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    f1score = classification_report(y_test, y_pred)
    print(f1score)
    joblib.dump(pipe, model_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_model(config_path=parsed_args.config)
