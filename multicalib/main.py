import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from multicalib.models import IncomeDataset, NNetPredictor
from multicalib.utils import train_predictor
from multicalib.multicalibration import calibrate,multi_calibrate


def main(data, features=[0,1]):
    # Load the datasets
    if (data == 'income'):
        trainset = IncomeDataset(file='adult_train.npz', root_dir='./data/')
        trainloader = DataLoader(trainset, batch_size=100, shuffle=True)

    # Train a preictor model
    model = NNetPredictor()
    train_predictor(model, trainloader, epochs=600)
    x = torch.stack([sample[0] for sample in list(trainset)])[:100]
    y = torch.stack([sample[1] for sample in list(trainset)])[:100]
    print('Data size: ', x.shape)
    predictions = model(x)
    predictions = torch.FloatTensor(predictions.shape).uniform_(0,1)

    # Calibrate output
    multi_calibrate(data=x.numpy(), lables=y.numpy(), predictions=predictions.detach().numpy(), sensitive_features=features, alpha=0.01, lmbda=5)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Use multicalibration to report clibrated results of a model')
    parser.add_argument('--data', type=str, default='income', help='Dataset to use for the experiment')
    parser.add_argument('--features', type=int, default=1, help='List of sensitive attributes')
    args = parser.parse_args()
    main(args.data)

