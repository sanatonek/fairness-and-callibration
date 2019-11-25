import torch
import numpy as np


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


def expected_accuracy(labels, predictions, regularized_predictions):
    predictions_b = (predictions>0.5).astype(int)
    calibrated_predictions_b = (regularized_predictions > 0.5).astype(int)
    labels_b = labels.reshape(-1,)
    return np.dot(labels_b ,predictions_b)/len(labels), np.dot(labels_b,calibrated_predictions_b)/len(labels)


def calibration_score(labels, predictions, regularized_predictions, lmbda=5):
    prediction_scores = []
    regularized_score = []
    v_range = np.arange(0, 1, 1. / lmbda)
    for v in v_range:
        S_v = [i for i in range(len(labels)) if predictions[i] < v + (1. / lmbda) and predictions[i] >= v]
        if len(S_v)==0:
            continue
        prediction_scores.append(np.mean(predictions[S_v])-np.mean(labels[S_v]))
    for v in v_range:
        S_v = [i for i in range(len(labels)) if regularized_predictions[i] < v + (1. / lmbda) and regularized_predictions[i] >= v]
        if len(S_v)==0:
            continue
        regularized_score.append(np.mean(regularized_predictions[S_v]) - np.mean(labels[S_v]))
    return np.mean(prediction_scores), np.mean(regularized_score)