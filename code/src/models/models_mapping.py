from models.models import AutoSklearnWrapper, AutoGluonWrapper, LamaWrapper, OboeWrapper, EmptyCatboostWrapper, \
    AutoPytorchWrapper

MODEL_NAMES_MAPPING = {
    "auto-sklearn": AutoSklearnWrapper,
    "autogluon": AutoGluonWrapper,
    "lama": LamaWrapper,
    "oboe": OboeWrapper,
    "catboost": EmptyCatboostWrapper,
    "autopytorch": AutoPytorchWrapper
}