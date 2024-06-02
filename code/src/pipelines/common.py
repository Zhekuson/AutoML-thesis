from abc import ABC, abstractmethod

import numpy as np

from datasets.common import ResearchDataset
from models.models import AutoMLWrapper
from pipelines.monitoring import ExperimentSettings


class ResearchPipeline:
    def __init__(self, dataset: ResearchDataset, model: AutoMLWrapper, settings: ExperimentSettings):
        self.dataset = dataset
        self.model = model
        self.settings = settings
        self.wandb_monitoring = self.settings.wandb_monitoring

    def run(self):
        for attempt in range(self.settings.n_attempts):
            np.random.seed(attempt)
            self.dataset.load(seed=attempt)
        # try:
            self.model.train(
                dataset=self.dataset,
                #dataset_name=self.settings.dataset_name
            )

            self.wandb_monitoring.log_metrics(self.model.get_metrics(dataset=self.dataset), attempt)
        # except Exception as e:
        #     print(f"EXCEPTION: {e}")
        #     continue
