from unittest.mock import patch

from resco.data.data import (
    get_train_mnist_dataset,
    get_test_mnist_dataset,
    get_train_mnist_dataloader,
    get_test_mnist_dataloader
)


@patch("resco.data.data.MNIST")
@patch("resco.data.data.ToTensor")
def test_get_train_mnist_dataset(to_tensor_mock, mnist_mock):
    get_train_mnist_dataset()
    mnist_mock.assert_called_with("data", download=True, train=True, transform=to_tensor_mock())


@patch("resco.data.data.MNIST")
@patch("resco.data.data.ToTensor")
def test_get_test_mnist_dataset(to_tensor_mock, mnist_mock):
    get_test_mnist_dataset()
    mnist_mock.assert_called_with("data", download=True, train=False, transform=to_tensor_mock())


@patch("resco.data.data.get_train_mnist_dataset", return_value="mock_dataset")
@patch("resco.data.data.DataLoader")
def test_get_train_mnist_dataloader(dataloader_mock, get_train_mnist_dataset_mock):
    mnist_dataset = get_train_mnist_dataset_mock()
    get_train_mnist_dataloader(mnist_dataset, 256, True)
    dataloader_mock.assert_called_with("mock_dataset", 256, True)


@patch("resco.data.data.get_test_mnist_dataset", return_value="mock_dataset")
@patch("resco.data.data.DataLoader")
def test_get_test_mnist_dataloader(dataloader_mock, get_train_mnist_dataset_mock):
    mnist_dataset = get_train_mnist_dataset_mock()
    get_test_mnist_dataloader(mnist_dataset, 256, True)
    dataloader_mock.assert_called_with("mock_dataset", 256, True)
