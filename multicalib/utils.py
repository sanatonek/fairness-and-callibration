import torch
import numpy as np


def train_predictor(args, model, train_loader, epochs, lmbda, lr=1e-4, momentum=0.9):
    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = args.epochs
    total = 0
    correct = 0
    running_loss = 0
        
    for t in range(epochs):
        for i_batch, sample_batched in enumerate(train_loader):
            x, y, a = sample_batched[0], sample_batched[1].squeeze(), sample_batched[2]
            
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = torch.nn.Sigmoid()(model(x))[:,1]
            predicted = (y_pred)
            
            # _, predicted = torch.max(y_pred.data, 1)
            # predicted = (y_pred>0.5).float()
            # print(predicted[:10])
            total += y.size(0)
            predicted_class = (y_pred>0.5).float()
            correct += (predicted_class==y).sum().item()

            loss = criterion(y_pred.float(), y.float())

            if (args.reg=='eqo'):
                a = a.view(-1,)
                tpr_0 = 0 if torch.sum(y[a==0].float())==0 else \
                    torch.dot(predicted[a==0], y[a==0].float())/torch.sum(y[a==0].float())
                tpr_1 = 0 if torch.sum((1-y)[a==0].float())==0 else \
                    torch.dot(predicted[a == 0], (1-y)[a == 0].float())/torch.sum((1-y)[a==0].float())
                tnr_0 = 0 if torch.sum(y[a==1].float())==0 else \
                    torch.dot(predicted[a == 1], y[a == 1].float())/torch.sum(y[a==1].float())
                tnr_1 = 0 if torch.sum((1-y)[a==1].float())==0 else \
                    torch.dot(predicted[a == 1], (1-y)[a == 1].float())/torch.sum((1-y)[a==1].float())
                loss = loss + lmbda*((tpr_0-tpr_1)*(tpr_0-tpr_1) + (tnr_0-tnr_1)*(tnr_0-tnr_1))
            running_loss += loss

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if t%10==0:
            print('epoch: {}, loss: {:.5f}, accuracy: {:.2f}%'.format(t, running_loss/total, 100*correct/total))

        running_loss = 0
        total = 0
        correct = 0


def expected_accuracy(labels, predictions):
    # predictions_b = (predictions>0.5).astype(int)
    labels_b = labels.reshape(-1,)
    # prediction_accuracy = np.sum(labels_b==predictions_b)/len(labels)
    # return prediction_accuracy*100
    error_rate = np.linalg.norm((labels_b-predictions))*np.linalg.norm((labels_b-predictions))
    return error_rate/len(labels)


def calibration_score(labels, predictions, lmbda=5):
    prediction_scores = []
    v_range = np.arange(0, 1, 1. / lmbda)
    for v in v_range:
        S_v = [i for i in range(len(labels)) if predictions[i] < v + (1. / lmbda) and predictions[i] >= v]
        if len(S_v)==0:
            continue
        else:
            prediction_scores.append(abs(np.mean(predictions[S_v] - labels[S_v])))
    prediction_scores = np.array(prediction_scores)
    return np.mean(prediction_scores)