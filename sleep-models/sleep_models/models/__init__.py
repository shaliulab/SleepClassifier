from .models import EBM, MLP, KNN
from .torch.nn import NeuralNetwork

MODELS ={
    "EBM": EBM,
    "KNN": KNN,
    "MLP": MLP,
    "NeuralNetwork": NeuralNetwork,
}
