import optuna
import mlflow
import yaml
from optuna.integration.mlflow import MLflowCallback
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import pandas as pd


def objective(trial, x_train, y_train, cat_cols, num_cols):
    iterations = trial.suggest_int("iterations", 100, 1500)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1, log=True)
    depth = trial.suggest_int("depth", 2, 10)

    classifier = CatBoostClassifier(iterations=iterations,
                                    depth=depth,
                                    learning_rate=learning_rate,
                                    verbose=False)
    ct = ColumnTransformer([('one_hot_enc', OneHotEncoder(), cat_cols),
                            ('standard_scaler', StandardScaler(), num_cols)])
    pipe = Pipeline([('preprocess', ct),
                    ('classifier', classifier)])
    f1_weighted = cross_val_score(pipe, x_train, y_train, 
                                  cv=3, scoring='f1_weighted').mean()
    return f1_weighted


def find_hyperparam(config):
    data_config = config['data_config']
    mlflow_config = config['mlflow_config']
    num_cols = data_config['num_cols']
    cat_cols = data_config['cat_cols']
    target = data_config['target']
    train_path = data_config['train_csv']
    n_trials = config['hyperparams_trials']

    train = pd.read_csv(train_path)
    x_train, y_train = train.drop(columns=target), train[target]

    mlflow.set_experiment(mlflow_config["experiment_name"])
    mlflc = MLflowCallback(metric_name="f1_weighted",
                           tracking_uri=mlflow_config["remote_server_uri"])
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="cs_model", sampler=sampler,
                                direction='maximize')
    study.optimize(lambda trial: objective(trial, x_train, y_train, cat_cols, num_cols),
                   n_trials=n_trials, callbacks=[mlflc])
    with open("params.yaml") as f:
        data = yaml.safe_load(f)
    data['classifier'] = study.best_params
    with open("params.yaml",'w') as f:
        yaml.dump(data, f)
