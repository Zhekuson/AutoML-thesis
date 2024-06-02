from const import BASE_SEED
from datasets.common import ResearchDataset
import pandas as pd
from sklearn.model_selection import train_test_split

class HiggsDataset(ResearchDataset):
    def __init__(self):
        pass


class ForestCoverDataset(ResearchDataset):
    def __init__(self, params):
        self.data_train = None
        self.data_test = None
        self.y_test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        print(params)
        self.path = params["load_path"]
        self.input_type: str = params["input_type"]
        self.train_test_split_args: dict = None if "train_test_split" not in params else params["train_test_split"]

    def load(self, seed=BASE_SEED):
        self.data = pd.read_csv(self.path)[:1000]
        if self.train_test_split_args:
            base_args = {
                "stratify": self.data["Cover_Type"],
                "random_state": seed,
            }
            base_args.update(self.train_test_split_args)
            self.train_test_split_args = base_args

            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.data.drop(columns=["Cover_Type"]),
                    self.data["Cover_Type"],
                    **self.train_test_split_args
                )
            del self.data
        if self.input_type == "data_split":
            self.data_train = pd.merge(self.X_train, self.y_train, left_index=True, right_index=True)
            del self.X_train, self.y_train
            self.data_test = pd.merge(self.X_test, self.y_test, left_index=True, right_index=True)




class CensusIncomeDataset(ResearchDataset):
    def __init__(self, params):
        self.cat_columns = [
            'education',
            'marital.status',
            'native.country',
            'occupation',
            'race',
            'relationship',
            'sex',
            'workclass'
        ]
        self.path = params["load_path"]
        self.input_type: str = params["input_type"]
        self.train_test_split_args: dict = None if "train_test_split" not in params else params["train_test_split"]

    def load(self, seed=BASE_SEED):
        self.data = pd.read_csv(self.path)
        self.data["income"] = self.data["income"].apply(lambda x: 1 if x == ">50K" else 0)
        if self.train_test_split_args:
            base_args = {
                "stratify": self.data["income"],
                "random_state": seed,
            }
            base_args.update(self.train_test_split_args)
            self.train_test_split_args = base_args

            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.data.drop(columns=["income"]),
                    self.data["income"],
                    **self.train_test_split_args
                )
            del self.data
        if self.input_type == "data_split":
            self.data_train = pd.merge(self.X_train, self.y_train, left_index=True, right_index=True)
            del self.X_train, self.y_train
            self.data_test = pd.merge(self.X_test, self.y_test, left_index=True, right_index=True)


class CDCDataset(ResearchDataset):
    def __init__(self, params):
        self.path = params["load_path"]
        self.input_type: str = params["input_type"]
        self.train_test_split_args: dict = None if "train_test_split" not in params else params["train_test_split"]

    def load(self, seed=BASE_SEED):
        self.data = pd.read_csv(self.path)
        if self.train_test_split_args:
            base_args = {
                "stratify": self.data["Diabetes_binary"],
                "random_state": seed,
            }
            base_args.update(self.train_test_split_args)
            self.train_test_split_args = base_args

            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.data.drop(columns=["Diabetes_binary"]),
                    self.data["Diabetes_binary"],
                    **self.train_test_split_args
                )
            del self.data
        if self.input_type == "data_split":
            self.data_train = pd.merge(self.X_train, self.y_train, left_index=True, right_index=True)
            del self.X_train, self.y_train
            self.data_test = pd.merge(self.X_test, self.y_test, left_index=True, right_index=True)



class AppleQualityDataset(ResearchDataset):
    def __init__(self, params):
        self.path = params["load_path"]
        self.input_type: str = params["input_type"]
        self.train_test_split_args: dict = None if "train_test_split" not in params else params["train_test_split"]

    def load(self, seed=BASE_SEED):
        self.data = pd.read_csv(self.path).drop(columns=["A_id"])
        self.data["Quality"] = self.data["Quality"].apply(lambda x: 1 if x == "good" else 0)
        if self.train_test_split_args:
            base_args = {
                "stratify": self.data["Quality"],
                "random_state": seed,
            }
            base_args.update(self.train_test_split_args)
            self.train_test_split_args = base_args

            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.data.drop(columns=["Quality"]),
                    self.data["Quality"],
                    **self.train_test_split_args
                )
            del self.data
        if self.input_type == "data_split":
            self.data_train = pd.merge(self.X_train, self.y_train, left_index=True, right_index=True)
            del self.X_train, self.y_train
            self.data_test = pd.merge(self.X_test, self.y_test, left_index=True, right_index=True)


class CreditCardFraudDataset(ResearchDataset):
    def __init__(self, params):
        self.path = params["load_path"]
        self.input_type: str = params["input_type"]
        self.train_test_split_args: dict = None if "train_test_split" not in params else params["train_test_split"]

    def load(self, seed=BASE_SEED):
        self.data = pd.read_csv(self.path).drop(columns=["id"])
        if self.train_test_split_args:
            base_args = {
                "stratify": self.data["Class"],
                "random_state": seed,
            }
            base_args.update(self.train_test_split_args)
            self.train_test_split_args = base_args

            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.data.drop(columns=["Class"]),
                    self.data["Class"],
                    **self.train_test_split_args
                )
            del self.data
        if self.input_type == "data_split":
            self.data_train = pd.merge(self.X_train, self.y_train, left_index=True, right_index=True)
            del self.X_train, self.y_train
            self.data_test = pd.merge(self.X_test, self.y_test, left_index=True, right_index=True)


class SepsisDataset(ResearchDataset):
    def __init__(self, params):
        self.path = params["load_path"]
        self.input_type: str = params["input_type"]
        self.train_test_split_args: dict = None if "train_test_split" not in params else params["train_test_split"]

    def load(self, seed=BASE_SEED):
        self.data = pd.read_csv(self.path)
        if self.train_test_split_args:
            base_args = {
                "stratify": self.data["hospital_outcome_1alive_0dead"],
                "random_state": seed,
            }
            base_args.update(self.train_test_split_args)
            self.train_test_split_args = base_args

            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.data.drop(columns=["hospital_outcome_1alive_0dead"]),
                    self.data["hospital_outcome_1alive_0dead"],
                    **self.train_test_split_args
                )
            del self.data
        if self.input_type == "data_split":
            self.data_train = pd.merge(self.X_train, self.y_train, left_index=True, right_index=True)
            del self.X_train, self.y_train
            self.data_test = pd.merge(self.X_test, self.y_test, left_index=True, right_index=True)


class MushroomDataset(ResearchDataset):
    def __init__(self, params):
        self.cat_columns = [
            'cap-color',
            'cap-shape',
            'cap-surface',
            'does-bruise-or-bleed',
            'gill-attachment',
            'gill-color',
            'gill-spacing',
            'habitat',
            'has-ring',
            'ring-type',
            'season',
            'spore-print-color',
            'stem-color',
            'stem-root',
            'stem-surface',
            'veil-color',
            'veil-type'
        ]
        self.path = params["load_path"]
        self.input_type: str = params["input_type"]
        self.train_test_split_args: dict = None if "train_test_split" not in params else params["train_test_split"]

    def load(self, seed=BASE_SEED):
        self.data = pd.read_csv(self.path)
        self.data["class"] = self.data["class"].apply(lambda x: 1 if x == "p" else 0)
        if self.train_test_split_args:
            base_args = {
                "stratify": self.data["class"],
                "random_state": seed,
            }
            base_args.update(self.train_test_split_args)
            self.train_test_split_args = base_args

            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.data.drop(columns=["class"]),
                    self.data["class"],
                    **self.train_test_split_args
                )
            del self.data
        if self.input_type == "data_split":
            self.data_train = pd.merge(self.X_train, self.y_train, left_index=True, right_index=True)
            del self.X_train, self.y_train
            self.data_test = pd.merge(self.X_test, self.y_test, left_index=True, right_index=True)