from torch.functional import Tensor
from torch.nn import Linear, Module, ReLU


class PlainBlock(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.linear = Linear(in_features, out_features)
        self.activation = ReLU()

    def forward(self, input: Tensor) -> Tensor:
        return self.activation(self.linear(input))


class ResidualBlock(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.linear = Linear(in_features, out_features)
        self.activation = ReLU()

    def forward(self, input: Tensor) -> Tensor:
        identity_input = input
        return self.activation(self.linear(input)) + identity_input
