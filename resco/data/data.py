from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def get_train_mnist_dataset(path_to_store_data: str = "data") -> Dataset:
    return MNIST(
        path_to_store_data,
        download=True,
        train=True,
        transform=ToTensor()
    )


def get_test_mnist_dataset(path_to_store_data: str = "data") -> Dataset:
    return MNIST(
        path_to_store_data,
        download=True,
        train=False,
        transform=ToTensor()
    )


def get_train_mnist_dataloader(
    mnist_dataset: Dataset, batch_size: int = 256, shuffle: bool = True
) -> DataLoader:
    return DataLoader(mnist_dataset, batch_size, shuffle)


def get_test_mnist_dataloader(
    mnist_dataset: Dataset, batch_size: int = 256, shuffle: bool = True
) -> DataLoader:
    return DataLoader(mnist_dataset, batch_size, shuffle)
