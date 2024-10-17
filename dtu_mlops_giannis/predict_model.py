import torch


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():  # Disable gradient computation
        for batch in dataloader:
            inputs = (
                batch[0] if isinstance(batch, (list, tuple)) else batch
            )  # Handle cases where dataloader returns (inputs, labels)
            outputs = model(inputs)
            predictions.append(outputs)
    return torch.cat([model(batch) for batch in dataloader], 0)
