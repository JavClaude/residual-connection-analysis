from unittest.mock import MagicMock, patch

from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam

from resco.train_eval.utils import (
    apply_gradient_descent,
    clean_gradients,
    compute_gradients,
    get_cross_entropy_loss_module,
    get_adam_optimizer,
    get_first_iteration_value,
    get_loss,
    get_next_iteration_value,
    log_histogram_on_tensorboard,
    reshape_2d_input_tensor_into_1d_input_tensor,
    get_predictions_of_the_model,
    log_scalar_on_tensorboard
)


def test_get_cross_entropy_loss_module_return_correct_object():
    output = get_cross_entropy_loss_module()

    assert isinstance(output, CrossEntropyLoss)


def test_get_adam_optimizer_should_return_correct_object():
    output = get_adam_optimizer(Linear(10, 10))

    assert isinstance(output, Adam)


def test_get_first_iteration_value_should_return_0():
    output = get_first_iteration_value()

    assert output == 0


def test_get_next_iteration_value_should_return_it_plus_one():
    it = 0
    output = get_next_iteration_value(it)

    assert output == 1


def test_reshape_2d_input_tensor_into_1d_input_tensor_should_call_view_method():    
    data = MagicMock()
    data.view = MagicMock()
    data.shape.__getitem__ = MagicMock(return_value=1)
    reshape_2d_input_tensor_into_1d_input_tensor(data)

    data.view.assert_called_with(1, -1)


def test_get_predictions_of_the_model_should_be_call_on_correct_arguments():
    model = MagicMock()
    get_predictions_of_the_model(1, model)

    model.assert_called_with(1)


def test_get_loss_should_be_call_with_correct_arguments():
    criterion = MagicMock()
    predictions = MagicMock()
    predictions.squeeze = MagicMock(return_value=3)

    get_loss(criterion, predictions, 1)

    criterion.assert_called_once_with(3, 1)


def test_compute_gradients_should_call_backward_method():
    loss = MagicMock()
    loss.backward = MagicMock()

    compute_gradients(loss)

    loss.backward.assert_called_once()


@patch("resco.train_eval.utils.writer")
def test_log_scalar_on_tensorboard_should_call_add_scalars_method_with_good_arguments(summary_writer_mock):
    tag = "test"
    dict_of_scalars = {"test_accuracy": 1}
    iteration = 0

    log_scalar_on_tensorboard(tag, dict_of_scalars, iteration)

    summary_writer_mock.add_scalars.assert_called_with(tag, dict_of_scalars, iteration)


@patch("resco.train_eval.utils.writer")
def test_log_histogram_on_tensorboard_should_call_add_scalars_method_with_good_arguments(summary_writer_mock):
    tag = "test"
    values = [1, 2, 3]
    iteration = 0

    log_histogram_on_tensorboard(tag, values, iteration)

    summary_writer_mock.add_histogram.assert_called_with(tag, values, iteration)


def test_apply_gradient_descent_should_call_step_method():
    optimizer = MagicMock()
    optimizer.step = MagicMock()

    apply_gradient_descent(optimizer)

    optimizer.step.assert_called()


def test_clean_gradients_should_call_zero_grad():
    optimizer = MagicMock()
    optimizer.zero_grad = MagicMock()

    clean_gradients(optimizer)

    optimizer.zero_grad.assert_called()
