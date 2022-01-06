import argparse

from resco.data.data import (
    get_train_mnist_dataloader,
    get_train_mnist_dataset,
    get_test_mnist_dataloader,
    get_test_mnist_dataset
)
from resco.models.models import (
    get_plain_dnn,
    get_resnet_dnn
)
from resco.train_eval.train import train_model


def main() -> None:
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument(
        "--model_name",
        required=True,
        help="plain_dnn or resnet_dnn",
        type=str
    )
    argument_parser.add_argument(
        "--n_blocks",
        required=False,
        default=50,
        help="number of plain/res block",
        type=int
    )
    argument_parser.add_argument(
        "--lr",
        required=False,
        default=0.0001,
        help="learning rate used to train the model",
        type=float
    )
    argument_parser.add_argument(
        "--batch_size",
        required=False,
        default=256,
        help="number of iteration used to train the model",
        type=int
    )
    argument_parser.add_argument(
        "--n_epochs",
        required=False,
        default=4,
        help="number of iteration used to train the model",
        type=int
    )

    arguments = argument_parser.parse_args()

    batch_size = arguments.batch_size
    n_blocks = arguments.n_blocks
    model_name = arguments.model_name
    lr = arguments.lr
    n_epochs = arguments.n_epochs

    train_mnist_dataloder = get_train_mnist_dataloader(
        get_train_mnist_dataset(),
        batch_size
    )
    eval_mnist_dataloader = get_test_mnist_dataloader(
        get_test_mnist_dataset(),
        batch_size
    )

    if model_name == "plain_dnn":
        model = get_plain_dnn(784, 784, n_blocks)
    elif model_name == "resnet_dnn":
        model = get_resnet_dnn(784, 784, n_blocks)
    else:
        raise ValueError(
            "Incorrect value error for model_name: {}".format(model_name)
        )

    train_model(
        train_mnist_dataloder,
        eval_mnist_dataloader,
        model,
        model_name,
        lr,
        n_epochs
    )


if __name__ == "__main__":
    main()
