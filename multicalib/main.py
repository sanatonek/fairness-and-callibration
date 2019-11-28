import sys
import torch
import argparse
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
sys.path.append('..')

from multicalib.models import IncomeDataset, CreditDataset, NNetPredictor
from multicalib.utils import expected_accuracy, calibration_score
from multicalib.multicalibration import calibrate,multicalibrate


# 1:age_u30, 65: race_black, 66: female
sensitive_features = {'income':[1, 65, 66]}


def main(args):
    # Load the datasets
    print('Loading %s dataset'%args.data)
    if args.data == 'income':
        testset = IncomeDataset(file='adult_test.npz', root_dir=args.path+'data/')
        testloader = DataLoader(testset, batch_size=100, shuffle=True)
    elif args.data == 'credit':
        testset = CreditDataset(file='credit_card_default_test.xls', root_dir=args.path+'data/')
        testloader = DataLoader(testset, batch_size=100, shuffle=True)
    features = sensitive_features[args.data]

    # Load a predictor model
    print('Loading the predictor model')
    model = NNetPredictor(testset.__dim__())
    model.load_state_dict(torch.load(args.path+'models/checkpoint_'+args.data+'.pth'))

    x = torch.stack([sample[0] for sample in list(testset)])
    y = torch.stack([sample[1] for sample in list(testset)])
    predictions = torch.nn.Sigmoid()(model(x))[:,1]

    # Calibrate output
    calibrated_predictions = calibrate(data=x.numpy(), labels=y.numpy(), predictions=predictions.detach().numpy(),
                                       sensitive_features=features, alpha=0.08, lmbda=5)
    multicalibrated_predictions = multicalibrate(data=x.numpy(), labels=y.numpy(), predictions=predictions.detach().numpy(),
                                        sensitive_features=features, alpha=0.08, lmbda=5)

    # Evaluate performance
    for sensitive_feature in features:
        # Find the two subset of the sensitive feature
        sensitive_set = [i for i in range(len(x)) if x.numpy()[i, sensitive_feature] == 1]
        true_acc, calibrated_acc = expected_accuracy(y.numpy()[sensitive_set], predictions.detach().numpy()[sensitive_set], calibrated_predictions[sensitive_set])
        _, multicalibrated_acc = expected_accuracy(y.numpy()[sensitive_set], predictions.detach().numpy()[sensitive_set], multicalibrated_predictions[sensitive_set])
        true_score, calibrated_score = calibration_score(y.numpy()[sensitive_set], predictions.detach().numpy()[sensitive_set], calibrated_predictions[sensitive_set])
        _,multicalibrated_score = calibration_score(y.numpy()[sensitive_set], predictions.detach().numpy()[sensitive_set],multicalibrated_predictions[sensitive_set])
        print("=====> Results for feature %d:"%sensitive_feature)
        print("Original labels: \tAccuracy: %.2f \tCalibration score: %.2f " %(true_acc, true_score))
        print("Calibrated labels: \tAccuracy: %.2f \tCalibration score: %.2f" % (calibrated_acc, calibrated_score))
        print("Multicalibrated labels: \tAccuracy: %.2f \tCalibration score: %.2f" % (multicalibrated_acc, multicalibrated_score))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Use multicalibration to report clibrated results of a model')
    parser.add_argument('--data', type=str, default='income', help='Dataset to use for the experiment')
    parser.add_argument('--path', type=str, default='./', help='Root directory path')

    args = parser.parse_args()
    main(args)

