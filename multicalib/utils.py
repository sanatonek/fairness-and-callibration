import torch


def train_predictor(model, train_loader, epochs=600, lr=1e-4, momentum=0.9):
    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)    
    epochs = 100
    total = 0
    correct = 0
    running_loss = 0
        
    for t in range(epochs):
        for i_batch, sample_batched in enumerate(train_loader):
            x, y, a = sample_batched[0], sample_batched[1].squeeze(), sample_batched[2]
            
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)
            
            # Compute and print loss
            loss = criterion(y_pred, y)
            
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss += loss
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (t%10==0):
            print('epoch: {}, loss: {:.5f}, accuracy: {:.2f}%'.format(t, running_loss/total, 100*correct/total))
            running_loss = 0
            total = 0
            correct = 0