from typing import Dict, Union, TYPE_CHECKING

from torch.functional import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

if TYPE_CHECKING:
    from models.models import PlainDNN, ResNetDNN

writer = SummaryWriter()


def get_cross_entropy_loss_module() -> CrossEntropyLoss:
    return CrossEntropyLoss()


def get_adam_optimizer(model: Module, lr: float = 0.0001) -> Adam:
    return Adam(model.parameters(), lr=lr)


def get_first_iteration_value() -> int:
    return 0


def get_next_iteration_value(iteration: int) -> int:
    return iteration + 1


def reshape_2d_input_tensor_into_1d_input_tensor(inputs: Tensor) -> Tensor:
    return inputs.view(inputs.shape[0], -1)


def get_predictions_of_the_model(inputs: Tensor, model: Module) -> Tensor:
    return model(inputs)


def get_loss(criterion: CrossEntropyLoss, predictions: Tensor, targets: Tensor) -> Tensor:
    return criterion(predictions.squeeze(), targets)


def compute_gradients(loss: Tensor) -> None:
    loss.backward()


def log_scalar_on_tensorboard(tag: str, dict_of_scalars: Dict, iteration: int) -> None:
    writer.add_scalars(
        tag,
        dict_of_scalars,
        iteration
    )


def log_histogram_on_tensorboard(tag: str, values: Tensor, iteration: int) -> None:
    writer.add_histogram(
        tag,
        values,
        iteration
    )


def get_gradients_of_the_first_layer(model: Union["PlainDNN", "ResNetDNN"]) -> Tensor:
    return model.neural_network[0].linear.weight.grad


def apply_gradient_descent(optimizer: Adam) -> None:
    optimizer.step()


def clean_gradients(optimizer: Adam) -> None:
    optimizer.zero_grad()
