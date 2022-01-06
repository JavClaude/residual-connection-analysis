from unittest.mock import patch

from resco.models.models import (
    PlainDNN,
    ResNetDNN
)


@patch("resco.models.models.torch.nn.Sequential")
def test_plain_ddn_should_call_neural_network_module(sequential_mock):
    plain_model = PlainDNN(10, 10, 5)
    sequential_mock().return_value = 10

    plain_model(10)

    plain_model.neural_network.assert_called_once_with(10)


@patch("resco.models.models.torch.nn.Sequential")
def test_plain_resnet_should_call_neural_network_module(sequential_mock):
    resnet_model = ResNetDNN(10, 10, 5)
    sequential_mock().return_value = 10

    resnet_model(10)

    resnet_model.neural_network.assert_called_once_with(10)
