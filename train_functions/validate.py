# validates one batch, returns total batch loss, number of true positives
from typing import Tuple


def validate_one_batch(model, inputs, targets, loss_function) -> Tuple[float, int]:
    predictions_val = model(inputs).detach()
    loss_validation = loss_function(predictions_val, targets)

    # calculate average loss
    batch_size_val = len(inputs)
    batch_loss = loss_validation.item() * batch_size_val

    # Calculate True positives
    predicted_class = predictions_val.argmax(axis=1)
    correct_class = targets.argmax(axis=1)

    true_positives_count = sum(predicted_class == correct_class).item()
    return batch_loss, true_positives_count
