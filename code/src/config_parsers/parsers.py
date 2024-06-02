import yaml
import typing as tp

from datasets.datasets_mapping import DATASET_NAMES_MAPPING
from models.models_mapping import MODEL_NAMES_MAPPING
from datasets.common import ResearchDataset
from models.models import AutoMLWrapper
from pipelines.monitoring import WandbMonitoring, ExperimentSettings


class DatasetParser:
    def __init__(self):
        pass

    def parse(self, config) -> ResearchDataset:
        print(config)
        return DATASET_NAMES_MAPPING[config["name"]](config["params"])

class ModelParamsParser:
    def __init__(self):
        pass

    def parse(self, config) -> AutoMLWrapper:
        return MODEL_NAMES_MAPPING[config["name"]](config["params"])

class SettingsParser:
    def __init__(self):
        pass

    def parse(self, config):
        return ExperimentSettings(config)


class ConfigParser:
    def __init__(self, config_path):
        self.settings = None
        self.dataset = None
        self.model = None
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.dataset_parser = DatasetParser()
        self.model_params_parser = ModelParamsParser()
        self.settings_parser = SettingsParser()

    def parse(self):
        self.dataset = self.dataset_parser.parse(self.config["dataset"])
        self.model = self.model_params_parser.parse(self.config["model"])
        self.settings = self.settings_parser.parse(self.config["settings"])
        return {
            "dataset": self.dataset,
            "model": self.model,
            "settings": self.settings
        }