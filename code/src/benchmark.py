import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score

from datasets.common import ResearchDataset
from pipelines.monitoring import ExperimentSettings
from datasets.datasets import ForestCoverDataset

import argparse
import os

import wandb
from config_parsers.parsers import ConfigParser
from pipelines.common import ResearchPipeline
import catboost as cb

wandb.login(key=os.environ["wandb_key"])
print("Config path: ", os.environ["config_path"])

if __name__ == '__main__':
    print("RUNNING")
    parser = argparse.ArgumentParser(description='Running')
    parser.add_argument("--config_path", "-c", "-p", metavar="c", type=str, dest="config_path")
    config_path = parser.parse_args().config_path

    cp = ConfigParser(config_path)
    parsed = cp.parse()
    settings: ExperimentSettings = parsed["settings"]
    dataset: ResearchDataset = parsed["dataset"]
    metric = sklearn.metrics.get_scorer(settings.settings["metric"])._score_func

    optuna_time_limit = settings.settings["optuna_time_limit"]
    catboost_time_limit = settings.settings["catboost_time_limit"]

    dataset.load()


    def objective(trial):
        param = {
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 2, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.02),
            "iterations": trial.suggest_int("iterations", 100, 1200),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 5, 50),
            "used_ram_limit": "7gb",
            # "loss_function": "MultiClass",
            "eval_metric": "AUC",
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        gbm = cb.CatBoostClassifier(**param)
        # FRAC = 0.2
        # _x_train: pd.DataFrame = dataset.X_train.sample(frac=FRAC, random_state=42)
        # _y_train: pd.DataFrame = dataset.y_train.sample(frac=FRAC, random_state=42)
        # _x_test: pd.DataFrame = dataset.X_test.sample(frac=FRAC, random_state=42)
        # _y_test: pd.DataFrame = dataset.y_test.sample(frac=FRAC, random_state=42)

        _x_train: pd.DataFrame = dataset.X_train
        _y_train: pd.DataFrame = dataset.y_train
        _x_test: pd.DataFrame = dataset.X_test
        _y_test: pd.DataFrame = dataset.y_test

        gbm.fit(
            _x_train, _y_train,
            eval_set=[(_x_test, _y_test)],
            verbose=0,
            cat_features=dataset.cat_columns if hasattr(dataset, "cat_columns") else None,
        )

        preds = gbm.predict(dataset.X_test)
        pred_labels = np.rint(preds)
        metric_value = metric(dataset.y_test, pred_labels)
        return metric_value

    for attempt in range(settings.n_attempts):
        print("ATTEMPT ", attempt)
        np.random.seed(attempt)
        dataset.load(seed=attempt)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, timeout=optuna_time_limit, show_progress_bar=True)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        best_params = trial.params
        print("  Params: ")
        for key, value in best_params.items():
            print("    {}: {}".format(key, value))

        settings.wandb_monitoring.wandb.log({"best_params": best_params})
        best_model: cb.CatBoostClassifier = cb.CatBoostClassifier(**best_params,
                                                                  eval_metric='AUC:hints=skip_train~false')

        best_model.fit(
            dataset.X_train, dataset.y_train,
            eval_set=[(dataset.X_test, dataset.y_test)],
            verbose=0,
            cat_features=dataset.cat_columns if hasattr(dataset, "cat_columns") else None
        )

        settings.wandb_monitoring.log_metrics(
            {"best_metric": best_model.get_best_score()["validation"]["AUC"]},
            attempt
        )
