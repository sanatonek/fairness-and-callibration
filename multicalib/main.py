import sys
import torch
import argparse
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
sys.path.append('..')

from multicalib.models import IncomeDataset, CreditDataset, NNetPredictor
from multicalib.utils import train_predictor
from multicalib.multicalibration import calibrate, multi_calibrate


def main(args, features=[1]):
    # Load the datasets
    print('Loading %s dataset'%args.data)
    if args.data == 'income':
        trainset = IncomeDataset(file='adult_train.npz', root_dir=args.path+'data/')
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testset = IncomeDataset(file='adult_test.npz', root_dir=args.path+'data/')
        testloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    elif args.data == 'credit':
        trainset = CreditDataset(file='credit_card_default_train.xls', root_dir=args.path+'data/')
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testset = CreditDataset(file='credit_card_default_test.xls', root_dir=args.path+'data/')
        testloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    # Train a predictor model
    if args.train:
        print('Training the predictor model')
        model = NNetPredictor(trainset.__dim__())
        train_predictor(model, trainloader, epochs=args.epochs)
        if not os.path.exists('./models'):
            os.mkdir('./models')
        torch.save(model, args.path+'models/checkpoint_'+args.data+'.mdl')
        torch.save(model.state_dict(), args.path+'models/checkpoint_'+args.data+'.pth')
    else:
        print('Loading the predictor model')
        model = torch.load(args.path+'models/checkpoint_'+args.data+'.mdl')
        model.load_state_dict(torch.load(args.path+'models/checkpoint_'+args.data+'.pth'))

    x = torch.stack([sample[0] for sample in list(testset)])
    y = torch.stack([sample[1] for sample in list(testset)])
    predictions = torch.nn.Sigmoid()(model(x))[:,0]

    # Calibrate output
    if args.mode == 'calib':
        calibrate(data=x.numpy(), lables=y.numpy(), predictions=predictions.detach().numpy(), sensitive_features=features, alpha=0.1, lmbda=5)
    elif args.mode =='multicalib':
        multi_calibrate(data=x.numpy(), lables=y.numpy(), predictions=predictions.detach().numpy(), sensitive_features=features, alpha=0.1, lmbda=5)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Use multicalibration to report clibrated results of a model')
    parser.add_argument('--data', type=str, default='income', help='Dataset to use for the experiment')
    parser.add_argument('--features', type=int, default=1, help='List of sensitive attributes')
    parser.add_argument('--batch_size', type=int, default=100, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs')
    parser.add_argument('--train',  action='store_true', help='Train the predictor model first')
    parser.add_argument('--mode', type=str, default='calib', help='["calib", "multicalib"]')
    parser.add_argument('--path', type=str, default='./', help='Root directory path')

    args = parser.parse_args()
    main(args)

