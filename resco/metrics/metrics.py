from torch import sum
from torch.functional import Tensor
from torch.nn import Softmax


def compute_accuracy(predictions: Tensor, targets: Tensor) -> Tensor:
    normalized_predictions = get_normalized_predictions(predictions)
    argmax_of_predictions = get_argmax_of_normalized_predictions(normalized_predictions)
    accuracy = get_accuracy(argmax_of_predictions, targets)
    return accuracy


def get_normalized_predictions(predictions: Tensor) -> Tensor:
    softmax = Softmax()
    return softmax(predictions)


def get_argmax_of_normalized_predictions(normalized_predictions: Tensor) -> Tensor:
    return normalized_predictions.argmax(1)


def get_accuracy(predictions: Tensor, targets: Tensor) -> Tensor:
    return sum(predictions.squeeze() == targets) / predictions.size(0)