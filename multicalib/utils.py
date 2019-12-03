import torch
import numpy as np
from torch.autograd import Variable

class EqualizedOddsReg(torch.nn.Module):
    
    def __init__(self):
        super(EqualizedOddsReg,self).__init__()
        
    def forward(self,predicted,y,a):
        # Equalized odds definition
        predicted = torch.squeeze(predicted)
        y = torch.squeeze(y)
        a = torch.squeeze(a)

        tpr_0 = torch.div(torch.sum((predicted)*(y)*(1-a)),torch.sum(y))
        tnr_0 = torch.div(torch.sum((predicted)*(1-y)*(1-a)),torch.sum(1-y))
        tpr_1 = torch.div(torch.sum((predicted)*(y)*(a)),torch.sum(y))
        tnr_1 = torch.div(torch.sum((predicted)*(1-y)*(a)),torch.sum(1-y))

        #totloss = Variable(torch.abs(torch.tensor(tpr_0-tpr_1))+torch.abs(torch.tensor(tnr_0-tnr_1)), requires_grad=True)
        #totloss = Variable(torch.abs(torch.tensor(tpr_0-tpr_1)), requires_grad=True)
        #totloss = torch.mean(torch.abs((tpr_0-tpr_1)+(tnr_0-tnr_1)))
        totloss = 1000.0*torch.mean(torch.abs((tpr_0-tpr_1)+(tnr_0-tnr_1)))

        return totloss


def train_predictor(args, model, train_loader, epochs=600, lr=1e-4, momentum=0.9):
    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = args.epochs
    total = 0
    correct = 0
    pred_loss = 0
    eq_odds_loss = 0
    running_loss = 0
        
    for t in range(epochs):
        for i_batch, sample_batched in enumerate(train_loader):
            x, y, a = sample_batched[0], sample_batched[1].squeeze(), sample_batched[2]
            
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = torch.nn.Sigmoid()(model(x))[:,1]
            predicted = torch.sigmoid(y_pred)
            
            # _, predicted = torch.max(y_pred.data, 1)
            # predicted = (y_pred>0.5).float()
            # print(predicted[:10])
            total += y.size(0)
            correct += ((y_pred>0.5).float() == y).sum().item()

            loss = criterion(y_pred.float(), y.float())

            if (args.reg=='eqo'):
                tpr_0 = torch.div(torch.sum((predicted) * (y) * (1 - a)), torch.sum(y * (1 - a)))
                tnr_0 = torch.div(torch.sum((predicted) * (1 - y) * (1 - a)), torch.sum((1 - y) * (1 - a)))
                tpr_1 = torch.div(torch.sum((predicted) * (y) * (a)), torch.sum(y * (a)))
                tnr_1 = torch.div(torch.sum((predicted) * (1 - y) * (a)), torch.sum((1 - y) * a))
                loss = loss + 10*((tpr_0-tpr_1)*(tpr_0-tpr_1) + (tnr_0-tnr_1)*(tnr_0-tnr_1))
            running_loss += loss

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (t%10==0):
            print('epoch: {}, loss: {:.5f}, accuracy: {:.2f}%'.format(t, running_loss/total, 100*correct/total))

        pred_loss = 0
        eq_odds_loss = 0
        running_loss = 0
        total = 0
        correct = 0


def expected_accuracy(labels, predictions):
    predictions_b = (predictions>0.5).astype(int)
    labels_b = labels.reshape(-1,)
    prediction_accuracy = np.sum(labels_b==predictions_b)/len(labels)
    return prediction_accuracy*100


def calibration_score(labels, predictions, lmbda=5):
    prediction_scores = []
    v_range = np.arange(0, 1, 1. / lmbda)
    for v in v_range:
        S_v = [i for i in range(len(labels)) if predictions[i] < v + (1. / lmbda) and predictions[i] >= v]
        if len(S_v)==0:
            continue
        # prediction_scores.append(abs(np.mean(predictions[S_v])-np.mean(labels[S_v])))
        prediction_scores.append(abs(np.mean(predictions[S_v])-(v+(1. / (2*lmbda)))))
    prediction_scores = np.array(prediction_scores)
    return np.mean(prediction_scores)