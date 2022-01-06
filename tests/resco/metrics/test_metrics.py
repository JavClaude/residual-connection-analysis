from unittest.mock import patch

from resco.metrics.metrics import (
    compute_accuracy,
    get_accuracy,
    get_argmax_of_normalized_predictions,
    get_normalized_predictions
)


@patch("resco.metrics.metrics.Softmax")
def test_get_normalized_predictions(softmax_mock):
    get_normalized_predictions(50)
    softmax_mock().assert_called_with(50)


@patch("resco.metrics.metrics.Tensor")
def test_get_argmax_of_normalized_predictions(predictions_tensor_mock):
    get_argmax_of_normalized_predictions(predictions_tensor_mock)
    predictions_tensor_mock.argmax.assert_called_with(1)


@patch("resco.metrics.metrics.sum")
@patch("resco.metrics.metrics.Tensor")
@patch("resco.metrics.metrics.Tensor")
def test_get_accuracy(predictions_tensor_mock, targets_tensor_mock, sum_mock):
    get_accuracy(predictions_tensor_mock, targets_tensor_mock)

    predictions_tensor_mock.squeeze.assert_called()
    predictions_tensor_mock.size.assert_called_with(0)
    sum_mock.assert_called_with(predictions_tensor_mock == targets_tensor_mock)


@patch("resco.metrics.metrics.Tensor")
@patch("resco.metrics.metrics.Tensor")
@patch("resco.metrics.metrics.get_normalized_predictions")
@patch("resco.metrics.metrics.get_argmax_of_normalized_predictions")
@patch("resco.metrics.metrics.get_accuracy")
def test_compute_accuracy(
    get_accuracy_mock,
    get_argmax_of_normalized_predictions_mock,
    get_normalized_predictions_mock,
    predictions_tensor_mock,
    targets_tensor_mock
):
    compute_accuracy(predictions_tensor_mock, targets_tensor_mock)
    get_normalized_predictions_mock.assert_called()
    get_argmax_of_normalized_predictions_mock.assert_called()
    get_accuracy_mock.assert_called()
