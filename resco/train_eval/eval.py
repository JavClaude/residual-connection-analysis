from typing import TYPE_CHECKING, Union

from tqdm import tqdm
from torch.functional import Tensor
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader


from resco.metrics.metrics import compute_accuracy
from resco.train_eval.utils import (
    get_cross_entropy_loss_module,
    get_first_iteration_value,
    reshape_2d_input_tensor_into_1d_input_tensor,
    get_predictions_of_the_model,
    get_loss,
    log_scalar_on_tensorboard
)
if TYPE_CHECKING:
    from models.models import PlainDNN, ResNetDNN


def eval_model(dataloader: DataLoader, model: Union["PlainDNN", "ResNetDNN"], model_name: str) -> None:
    criterion = get_cross_entropy_loss_module()
    iteration = get_first_iteration_value()
    eval_model_on_epoch(dataloader, criterion, model, model_name, iteration)


def eval_model_on_epoch(
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    model: Union["PlainDNN", "ResNetDNN"],
    model_name: str,
    iteration: int
) -> None:
    for inputs, targets in tqdm(dataloader, desc="Evaluating model: {}".format(model_name)):
        eval_model_on_batch(inputs, targets, criterion, model, model_name, iteration)


def eval_model_on_batch(
    inputs: Tensor,
    targets: Tensor,
    criterion: CrossEntropyLoss,
    model: Union["PlainDNN", "ResNetDNN"],
    model_name: str,
    iteration: int
) -> None:
    inputs = reshape_2d_input_tensor_into_1d_input_tensor(inputs)
    predictions = get_predictions_of_the_model(inputs, model)
    loss = get_loss(criterion, predictions, targets)
    accuracy = compute_accuracy(predictions, targets)
    log_scalar_on_tensorboard(
        model_name,
        {"eval loss": loss.item(), "eval accuracy": accuracy},
        iteration
    )
