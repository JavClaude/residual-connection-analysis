from typing import TYPE_CHECKING, Union

from tqdm import tqdm
from torch.functional import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader


from resco.metrics.metrics import compute_accuracy
if TYPE_CHECKING:
    from resco.models.models import (
        PlainDNN,
        ResNetDNN
    )
from resco.train_eval.eval import eval_model
from resco.train_eval.utils import (
    get_cross_entropy_loss_module,
    get_adam_optimizer,
    get_first_iteration_value,
    get_next_iteration_value,
    reshape_2d_input_tensor_into_1d_input_tensor,
    get_predictions_of_the_model,
    get_loss,
    compute_gradients,
    log_scalar_on_tensorboard,
    get_gradients_of_the_first_layer,
    log_histogram_on_tensorboard,
    apply_gradient_descent,
    clean_gradients
)


def train_model(
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    model: Union["PlainDNN", "ResNetDNN"],
    model_name: str,
    lr: float,
    n_epochs: int
) -> None:
    criterion = get_cross_entropy_loss_module()
    optimizer = get_adam_optimizer(model, lr)
    iteration = get_first_iteration_value()

    for _ in range(n_epochs):
        train_model_on_epoch(
            train_dataloader,
            criterion,
            optimizer,
            model,
            model_name,
            iteration
        )
        eval_model(
            eval_dataloader,
            model,
            model_name
        )


def train_model_on_epoch(
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    optimizer: Adam,
    model: Union["PlainDNN", "ResNetDNN"],
    model_name: str,
    iteration: int
) -> None:
    for inputs, targets in tqdm(dataloader, desc="Training model: {}".format(model_name)):
        train_on_batch(
            inputs,
            targets,
            criterion,
            optimizer,
            model,
            model_name,
            iteration
        )
        iteration = get_next_iteration_value(iteration)


def train_on_batch(
    inputs: Tensor,
    targets: Tensor,
    criterion: CrossEntropyLoss,
    optimizer: Adam,
    model: Union["PlainDNN", "ResNetDNN"],
    model_name: str,
    iteration: int
) -> None:
    inputs = reshape_2d_input_tensor_into_1d_input_tensor(inputs)
    predictions = get_predictions_of_the_model(inputs, model)
    loss = get_loss(criterion, predictions, targets)
    compute_gradients(loss)
    accuracy = compute_accuracy(predictions, targets)
    log_scalar_on_tensorboard(
        model_name, {"train loss": loss.item(), "train accuracy": accuracy},
        iteration
    )
    first_layer_gradient = get_gradients_of_the_first_layer(model)
    log_histogram_on_tensorboard(
        "1st gradient layer: {}".format(model_name),
        first_layer_gradient,
        iteration
    )
    apply_gradient_descent(optimizer)
    clean_gradients(optimizer)
