from unittest.mock import patch

from resco.models.modules import (
    PlainBlock,
    ResidualBlock
)

@patch("resco.models.modules.Linear")
@patch("resco.models.modules.ReLU")
def test_forward_pass_plain_block(relu_mock, linear_mock):
    plain_block = PlainBlock(10, 10)
    linear_mock().return_value = 10
    relu_mock().return_value = 20

    output = plain_block(100)

    plain_block.linear.assert_called_with(100)
    plain_block.activation.assert_called_with(10)

    assert output == 20


@patch("resco.models.modules.Linear")
@patch("resco.models.modules.ReLU")
def test_forward_pass_res_block(relu_mock, linear_mock):
    res_block = ResidualBlock(10, 10)
    linear_mock().return_value = 10
    relu_mock().return_value = 20

    output = res_block(100)

    res_block.linear.assert_called_with(100)
    res_block.activation.assert_called_with(10)

    assert output == 120
