import torch


def train_predictor(model, train_loader, epochs=600, lr=1e-4, momentum=0.9):
    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    total = 0
    correct = 0

    for t in range(epochs):
        for i_batch, sample_batched in enumerate(train_loader):
            x, y, a = sample_batched[0], sample_batched[1].squeeze(), sample_batched[2]

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)

            # print('y_pred.dims: {}'.format(y_pred.shape))
            # print('y.dims: {}'.format(y.shape))

            # Compute and print loss
            loss = criterion(y_pred, y)

            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            if t % 100 == 50:
                print('Epoch %d ==> Accuracy: %.2f \t Loss: %.3f' %(t, 100 * correct / total, loss.item()))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
