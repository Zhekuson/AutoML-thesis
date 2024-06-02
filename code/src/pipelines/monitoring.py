import pandas as pd
import wandb
import typing as tp


class WandbMonitoring:
    def __init__(
        self,
        init_config
    ):
        self.wandb: wandb = wandb.init(**init_config)

    def log_metrics(self, metrics, run_attempt):
        print(type(metrics))
        print(metrics)
        if type(metrics) is pd.DataFrame:
            table = wandb.Table(dataframe=metrics, allow_mixed_types=True)
            print("logging table:", table)
            self.wandb.log({"run_results": table})
            self.wandb.log({"best_metric": metrics.iloc[0]["score_val"]})
        else:
            print("logging metrics:", metrics)
            self.wandb.log(metrics)

class ExperimentSettings:
    def __init__(self, settings_config):
        self.settings = settings_config
        self.wandb_monitoring = WandbMonitoring(
            settings_config["wandb"]
        )
        # self.model_save_path = settings_config["model_save_path"]
        # self.dataset_name = settings_config["dataset_name"]
        self.n_attempts = settings_config["n_attempts"]