import torch
from torch.functional import Tensor

from resco.models.modules import (
    PlainBlock,
    ResidualBlock
)


class PlainDNN(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, n_plain_block: int) -> None:
        super().__init__()

        self.neural_network = torch.nn.Sequential(
            *[
                PlainBlock(in_features, in_features) for _ in range(n_plain_block)
            ],
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_features)
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.neural_network(input)


class ResNetDNN(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, n_res_block: int) -> None:
        super().__init__()

        self.neural_network = torch.nn.Sequential(
            *[
                ResidualBlock(in_features, in_features) for _ in range(n_res_block)
            ],
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_features)
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.neural_network(input)


def get_plain_dnn(in_features: int, out_features: int, n_plain_block: int) -> PlainDNN:
    return PlainDNN(in_features, out_features, n_plain_block)


def get_resnet_dnn(in_features: int, out_features: int, n_res_block: int) -> ResNetDNN:
    return ResNetDNN(in_features, out_features, n_res_block)
