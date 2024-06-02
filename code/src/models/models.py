import os
import pickle
from abc import ABC

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics

ENV_TYPE = os.getenv("env_type")
if ENV_TYPE == "autosklearn":
    import autosklearn
    from autosklearn.classification import AutoSklearnClassifier
    from autosklearn.metrics import CLASSIFICATION_METRICS
elif ENV_TYPE == "autogluon":
    from autogluon.tabular import TabularDataset, TabularPredictor
elif ENV_TYPE == "oboe":
    from oboe import AutoLearner
elif ENV_TYPE == "lama":
    from lightautoml.automl.base import AutoML
    from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
    from lightautoml.tasks import Task
elif ENV_TYPE == "autopytorch":
    from autoPyTorch.api.tabular_classification import TabularClassificationTask


###########################
###########################
class AutoMLWrapper(ABC):
    def __init__(self):
        pass

    def dump_artifact(self, *args, **kwargs):
        pass

    def train(self, **kwargs):
        pass

    def get_metrics(self, **kwargs):
        pass
###########################
###########################


class EmptyCatboostWrapper(AutoMLWrapper):
    def __init__(self, params):
        super().__init__()


class AutoPytorchWrapper(AutoMLWrapper):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.optimize_metric = self.params.pop("optimize_metric")
        self.total_walltime_limit = self.params.pop("total_walltime_limit")
        self.func_eval_time_limit_secs = self.params.pop("func_eval_time_limit_secs")

    def _init_predictor(self):
        # initialise Auto-PyTorch api
        self.automl = TabularClassificationTask(
            **self.params
        )

    def train(self, **kwargs):
        self._init_predictor()
        dataset = kwargs["dataset"]
        self.automl.search(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            optimize_metric=self.optimize_metric,
            total_walltime_limit=self.total_walltime_limit,
        )


    def get_metrics(self, **kwargs):
        dataset = kwargs["dataset"]
        y_pred = self.automl.predict(dataset.X_test)
        score = self.automl.score(y_pred, dataset.y_test)
        return {
            "best_metric": score
        }




class OboeWrapper(AutoMLWrapper):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.metric = self.params.pop("metric")
        self.metric = sklearn.metrics.get_scorer(self.metric)._score_func

    def dump_artifact(self, *args, **kwargs):
        pass


    def _init_predictor(self):
        self.automl = AutoLearner(
            **self.params,
        )


    def train(self, **kwargs):
        self._init_predictor()
        dataset = kwargs["dataset"]
        self.automl.fit(dataset.X_train.to_numpy(), dataset.y_train.to_numpy()) ##.astype('>i4').astype('>i4')


    def get_metrics(self, **kwargs):
        dataset = kwargs["dataset"]
        print(self.automl.predict(dataset.X_test).reshape(-1, 1).shape)
        print(dataset.y_test.shape)
        return {
            "best_metric": self.metric(
                self.automl.predict(dataset.X_test).reshape(-1, 1),
                dataset.y_test,
            )
        }




class LamaWrapper(AutoMLWrapper):
    def __init__(self, params):
        super().__init__()
        self.task = params.pop("task")
        self.params = params
        if self.task["metric"] == "accuracy":
            self.task["metric"] = lambda y_true, y_pred: \
                sklearn.metrics.accuracy_score(
                    y_true,
                    np.argmax(
                        np.array(y_pred),
                        axis=1
                    )
                )

        self.fit_predict_params = self.params.pop("fit_predict_params")
        self.task = Task(**self.task)

    def _init_predictor(self):
        self.automl = TabularUtilizedAutoML(
            task=self.task,
            **self.params
        )

    def train(self, **kwargs):
        self._init_predictor()
        dataset = kwargs["dataset"]
        self.automl.fit_predict(
            train_data=dataset.data_train,
            valid_data=dataset.data_test,
            **self.fit_predict_params
        )
        pass

    def get_metrics(self, **kwargs):
        dataset = kwargs["dataset"]
        return {
            "best_metric": self.task.metric_func(dataset.y_test,
                                                 self.automl.predict(dataset.X_test).data),
        }


class AutoGluonWrapper(AutoMLWrapper):
    def __init__(self, params):
        # non init params
        self.params: dict = params
        self.holdout_frac = self.params.pop("holdout_frac")
        self.time_limit = self.params.pop("time_limit")

    def _init_predictor(self):
        self.automl: TabularPredictor = TabularPredictor(
            **self.params
        )

    def train(self, **kwargs):
        self._init_predictor()
        dataset = kwargs["dataset"]
        train_data = TabularDataset(dataset.data)
        predictor: TabularPredictor = self.automl.fit(
            train_data,
            holdout_frac=self.holdout_frac,
            time_limit=self.time_limit,
        )
        return predictor

    def get_metrics(self, **kwargs):
        return self.automl.leaderboard(extra_info=True)


class AutoSklearnWrapper(AutoMLWrapper):

    def get_filename(self, attempt_number, save_path):
        return os.path.join(save_path, f'autosklearn_metric_{self.params["metric"]}_attempt_{attempt_number}.pkl')

    def dump_artifact(self, *args, **kwargs):
        with open(kwargs["model_save_path"], "wb") as f:
            pickle.dump(self.automl, f)
        print(f"Saved model {self.automl} successfully")

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.params["metric"] = CLASSIFICATION_METRICS[params["metric"]]
        if "scoring_functions" in params:
            self.params["scoring_functions"] = [CLASSIFICATION_METRICS[x] for x in params["scoring_functions"]]

    def _init_predictor(self):
        self.automl: AutoSklearnClassifier = AutoSklearnClassifier(
            **self.params,
        )

    def train(self, **kwargs):
        self._init_predictor()
        dataset = kwargs['dataset']
        self.automl.fit(
            dataset.X_train,
            dataset.y_train,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            #dataset_name=kwargs['dataset_name'],
        )
        return self.automl
        #print(self.automl.sprint_statistics())
        #print(self.automl.leaderboard(detailed=True))
        #self.automl.performance_over_time_

    def get_metrics(self, **kwargs):
        dataset = kwargs["dataset"]
        #print(self.automl.leaderboard(detailed=True))
        return {
            "best_metric": self.params["metric"](dataset.y_test, self.automl.predict(dataset.X_test)),
        }