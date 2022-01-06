from unittest.mock import MagicMock, patch

from resco.train_eval.train import (
    train_model
)


@patch("resco.train_eval.train.get_cross_entropy_loss_module", return_value=1)
@patch("resco.train_eval.train.get_adam_optimizer", return_value=2)
@patch("resco.train_eval.train.get_first_iteration_value", return_value=3)
@patch("resco.train_eval.train.train_model_on_epoch")
def test_train_model_should_call_others_training_functions(
    train_model_on_epoch_mock,
    get_first_iteration_value_mock,
    get_adam_optimizer_mock,
    get_cross_entropy_loss_module_mock
):
    train_dataloader = MagicMock()
    eval_dataloader = MagicMock()
    model = MagicMock()
    model_name = "test_model"
    iteration = 3
    lr = 0.1
    n_epochs = 4

    train_model(train_dataloader, eval_dataloader, model, model_name, lr, n_epochs)

    get_cross_entropy_loss_module_mock.assert_called()
    get_adam_optimizer_mock.assert_called_with(model, lr)
    get_first_iteration_value_mock.assert_called()
    train_model_on_epoch_mock.assert_called_with(
        train_dataloader,
        1,
        2,
        model,
        model_name,
        iteration
    )
    assert train_model_on_epoch_mock.call_count == 4
