import torch
import numpy as np
from torch.autograd import Variable

class EqualizedOddsReg(torch.nn.Module):
    
    def __init__(self):
        super(EqualizedOddsReg,self).__init__()
        
    def forward(self,predicted,y,a):
        # Equalized odds definition
        tpr_0= (torch.where((predicted==y)&(y==1)&(a==0))[0]).shape[0]/(torch.where((y==1))[0]).shape[0]
        tnr_0 = (torch.where((predicted==y)&(y==0)&(a==0))[0]).shape[0]/(torch.where((y==0))[0]).shape[0]

        tpr_1= (torch.where((predicted==y)&(y==1)&(a==1))[0]).shape[0]/(torch.where((y==1))[0]).shape[0]
        tnr_1 = (torch.where((predicted==y)&(y==0)&(a==1))[0]).shape[0]/(torch.where((y==0))[0]).shape[0]

        totloss = Variable(torch.tensor(abs(tpr_0-tpr_1))+torch.tensor(abs(tnr_0-tnr_1)), requires_grad=True)

        return totloss


def train_predictor(model, train_loader, epochs=600, lr=1e-4, momentum=0.9):
    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    criterion = torch.nn.CrossEntropyLoss()
    criterion_eq_odds = EqualizedOddsReg()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)    
    epochs = 100
    total = 0
    correct = 0
    pred_loss = 0
    eq_odds_loss = 0
    running_loss = 0
        
    for t in range(epochs):
        for i_batch, sample_batched in enumerate(train_loader):
            x, y, a = sample_batched[0], sample_batched[1].squeeze(), sample_batched[2]
            
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)
            
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            # Compute and print loss
            #loss = criterion(y_pred, y)
            loss = criterion(y_pred, y) + criterion_eq_odds(predicted, y, a)

            pred_loss += criterion(y_pred, y)
            eq_odds_loss += criterion_eq_odds(predicted, y, a)
            running_loss += loss

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (t%10==0):
            print('epoch: {}, pred_loss: {:.5f}, eq_odds_loss: {:.5f}, loss: {:.5f}, accuracy: {:.2f}%'.format(t, pred_loss/total, eq_odds_loss/total, running_loss/total, 100*correct/total))

        pred_loss = 0
        eq_odds_loss = 0
        running_loss = 0
        total = 0
        correct = 0


def expected_accuracy(labels, predictions, regularized_predictions):
    # print(predictions[:10])
    predictions_b = (predictions>0.5).astype(int)
    # print(predictions_b[:10])
    calibrated_predictions_b = (regularized_predictions > 0.5).astype(int)
    labels_b = labels.reshape(-1,)

    #multicalibrated_predictions_b = (multicalibrated_predictions > 0.5).astype(int)
    # return np.dot(labels_b ,predictions_b)/len(labels), np.dot(labels_b,calibrated_predictions_b)/len(labels)
    #print(len(labels),len(labels_b),labels_b.shape,predictions_b.shape)
    prediction_accuracy = np.sum(labels_b==predictions_b)/len(labels)
    calibrated_accuracy = np.sum(labels_b==calibrated_predictions_b)/len(labels)
    #multicalibrated_accuracy = np.sum(labels_b == multicalibrated_predictions_b/len(labels))
    return prediction_accuracy, calibrated_accuracy 


def calibration_score(labels, predictions, regularized_predictions, lmbda=5):
    prediction_scores = []
    regularized_score = []
    #multicalibrated_score = []
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