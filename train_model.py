import yaml
import joblib
import mlflow


import argparse
import pandas as pd
from urllib.parse import urlparse
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from hyperparam_tuning import find_hyperparam


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def train_model(config_path):
    config = read_params(config_path)
    data_config = config['data_config']
    model_dir = config['model_dir']
    classifier_config = config['classifier']
    mlflow_config = config["mlflow_config"]


    target = data_config['target']
    train_path = data_config['train_csv']
    test_path = data_config['test_csv']
   
    
    train = pd.read_csv(train_path)
    x_train, y_train = train.drop(columns=target), train[target]
    test = pd.read_csv(test_path)
    x_test, y_test = test.drop(columns=target), test[target]
    if config['hyperparams_tuning']:
        find_hyperparam(config)

    depth = classifier_config['depth']
    learning_rate = classifier_config['learning_rate']
    iterations = classifier_config['iterations']

    mlflow.set_tracking_uri(mlflow_config["remote_server_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]):
        classifier = CatBoostClassifier(iterations=iterations,
                                        depth=depth,
                                        learning_rate=learning_rate,
                                        verbose=False)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        f1score = f1_score(y_test, y_pred, average='weighted')
        mlflow.log_metric('f1_weighted', f1score)
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(classifier, "model",
            registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.log_model(classifier, "model")
    joblib.dump(classifier, model_dir)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_model(config_path=parsed_args.config)
