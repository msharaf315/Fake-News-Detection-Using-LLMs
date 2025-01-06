# trains one batch, returns total batch loss
def train_one_batch(model, inputs, targets, optimizer, loss_function) -> float:
    # Predict/Forward Pass
    predictions = model(inputs)
    # Compute loss
    loss = loss_function(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Multiply the cross entropy loss which is the average by the batch size so we get the total loss for the batch, we can divide this by all data set
    # Size to get average loss for the epoch
    batch_size_train = len(inputs)
    batch_loss = loss.item() * batch_size_train
    return batch_loss
