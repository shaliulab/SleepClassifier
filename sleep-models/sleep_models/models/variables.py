from collections import namedtuple
from sleep_models.constants import KNNHyperParams, NeuralNetworkHyperParams, EBMHyperParams

NeuralNetworkConfig = namedtuple(
    "TrainingConfig", NeuralNetworkHyperParams,
)

KNNConfig = namedtuple(
    "TrainingConfig", KNNHyperParams
)

EBMConfig = namedtuple(
    "TrainingConfig", EBMHyperParams
)

AllConfig = namedtuple(
    "AllConfig", ["training_config", "cluster", "output", "device", "model_name", "random_state"]
)
ModelProperties = namedtuple("ModelProperties", ["encoding", "target", "estimator_type"])

CONFIGS = {
    "NeuralNetwork": (NeuralNetworkConfig, NeuralNetworkHyperParams),
    "KNN": (KNNConfig, KNNHyperParams),
    "EBM": (EBMConfig, EBMHyperParams),
}