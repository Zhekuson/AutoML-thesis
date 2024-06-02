from datasets.datasets import HiggsDataset, ForestCoverDataset, CensusIncomeDataset, CDCDataset, CreditCardFraudDataset, \
    SepsisDataset, MushroomDataset, AppleQualityDataset
from datasets.syn_datasets import SynSingleModalDataset, SynMultiModalDataset

DATASET_NAMES_MAPPING = {
    "higgs": HiggsDataset,
    "forest": ForestCoverDataset,
    "census": CensusIncomeDataset,
    "diabetes": CDCDataset,
    "fraud": CreditCardFraudDataset,
    "sepsis": SepsisDataset,
    "mushroom": MushroomDataset,
    "ssd": SynSingleModalDataset,
    "smd": SynMultiModalDataset,
    "apple": AppleQualityDataset,
}