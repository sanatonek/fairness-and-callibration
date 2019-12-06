import sys
import torch
import argparse
import os
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader
sys.path.append('..')

from multicalib.models import IncomeDataset, CreditDataset, RecidDataset, NNetPredictor
from multicalib.utils import expected_accuracy, calibration_score, EqualizedOddsReg
from multicalib.multicalibration import calibrate, multicalibrate


sensitive_features = {
                      # 1:age_u30, 65: race_black, 66: female
                      'income':[1, 65, 66], 
                      # 2: sex, 5: age
                      'credit': [2, 5], 
                      # 3: Age>45, 4: Age<25, 5: Black, 6: Asian, 7: Hispanic, 10: Female
                      'recidivism': [5, 10]
                     }


def main(args):
    # Load the datasets
    print('Loading %s dataset'%args.data)
    if args.data == 'income':
        testset = IncomeDataset(file='adult_test.npz', root_dir=args.path+'data/')
        testloader = DataLoader(testset, batch_size=100, shuffle=True)
    elif args.data == 'credit':
        testset = CreditDataset(file='credit_card_default_test.xls', root_dir=args.path+'data/')
        testloader = DataLoader(testset, batch_size=100, shuffle=True)
    elif (args.data == 'recidivism'):
        testset = RecidDataset(file='propublica_data_for_fairml_test.csv', root_dir=args.path+'data/')
        testloader = DataLoader(testset, batch_size=100, shuffle=True)    
    
    features = sensitive_features[args.data]

    # Load a predictor model
    print('Loading the predictor model')
    model = NNetPredictor(testset.__dim__())
    model.load_state_dict(torch.load(args.path+'models/checkpoint_'+args.data+'_reg_None.pth'))
    print('Loading the regularized predictor model')
    model_reg = NNetPredictor(testset.__dim__())
    model_reg.load_state_dict(torch.load(args.path+'models/checkpoint_'+args.data+'_reg_eqo.pth'))

    x = torch.stack([sample[0] for sample in list(testset)])
    y = torch.stack([sample[1] for sample in list(testset)])
    predictions = torch.nn.Sigmoid()(model(x))[:,1]
    predictions_reg = torch.nn.Sigmoid()(model_reg(x))[:, 1]


    calibrated_predictions = calibrate(data=x.numpy(), labels=y.numpy(), predictions=predictions.detach().numpy(),
                                       sensitive_features=sensitive_features[args.data], alpha=args.alpha, lmbda=args.lmbda)
    multicalibrated_predictions = multicalibrate(data=x.numpy(), labels=y.numpy(), predictions=predictions.detach().numpy(),
                                        sensitive_features=sensitive_features[args.data], alpha=args.alpha, lmbda=args.lmbda)

    # Evaluate performance
    for sensitive_feature in features:
        print('\n=====> Results for feature ', sensitive_feature)
        print('********** Overall Accuracy **********')
        print('Ground truth: %2f\t Regularized: %2f\t Calibration %2f\t Multicalibration %2f\t'
              %(expected_accuracy(y.numpy(), predictions.detach().numpy()) , expected_accuracy(y.numpy(), predictions_reg.detach().numpy()),
              expected_accuracy(y.numpy(), calibrated_predictions) , expected_accuracy(y.numpy(), multicalibrated_predictions)))
        # Find the two subset of the sensitive feature
        sensitive_set = [i for i in range(len(x)) if x.numpy()[i, sensitive_feature] == 1]
        # sensitive_set = list(set(range(len(x))) - set(sensitive_set))
        if len(sensitive_set)==0:
            continue

        y_s = y.numpy()[sensitive_set]
        prediction_s = predictions.detach().numpy()[sensitive_set]
        predictions_reg_s = predictions_reg.detach().numpy()[sensitive_set]
        calibrated_predictions_s = calibrated_predictions[sensitive_set]
        multicalibrated_predictions_s = multicalibrated_predictions[sensitive_set]

        ## TPR, TNR
        # ground truth
        tpr = 0 if np.sum(y_s)==0 else np.dot((prediction_s), (y_s))/ np.sum(y_s)
        tnr = 0 if np.sum(1-y_s)==0 else  np.dot((prediction_s), (1 - y_s))/ np.sum(1 - y_s)
        # reg
        tpr_reg = 0 if np.sum(y_s)==0 else  np.dot((predictions_reg_s), (y_s))/np.sum(y_s)
        tnr_reg = 0 if np.sum(1-y_s)==0 else  np.dot((predictions_reg_s), (1 - y_s))/ np.sum(1 - y_s)
        # calib
        tpr_reg_calib = 0 if np.sum(y_s)==0 else  np.dot(calibrated_predictions_s, y_s)/ np.sum(y_s)
        tnr_reg_calib = 0 if np.sum(1-y_s)==0 else  np.dot(calibrated_predictions_s, (1 - y_s))/ np.sum(1 - y_s)
        # multicalib
        tpr_reg_multicalib = 0 if np.sum(y_s)==0 else  np.dot(multicalibrated_predictions_s, y_s)/ np.sum(y_s)
        tnr_reg_multicalib = 0 if np.sum(1-y_s)==0 else  np.dot(multicalibrated_predictions_s, (1 - y_s))/ np.sum(1 - y_s)
        print('\n********** TPR **********')
        print('Ground truth: %2f\t Regularized: %2f\t Calibration %2f\t Multicalibration %2f\t'
              %(tpr, tpr_reg, tpr_reg_calib, tpr_reg_multicalib))
        print('********** TNR **********')
        print('Ground truth: %2f\t Regularized: %2f\t Calibration %2f\t Multicalibration %2f\t'
              %(tnr, tnr_reg, tnr_reg_calib, tnr_reg_multicalib))

        ##  Accuracies
        # Ground truth
        accuracy = expected_accuracy(y_s, prediction_s)
        # reg
        accuracy_reg = expected_accuracy(y_s, predictions_reg_s)
        # calib
        accuracy_calib = expected_accuracy(y_s, calibrated_predictions_s)
        # multicalib
        accuracy_multicalib = expected_accuracy(y_s, multicalibrated_predictions_s)
        print('********** Accuracy **********')
        print('Ground truth: %2f\t Regularized: %2f\t Calibration %2f\t Multicalibration %2f\t'
              %(accuracy, accuracy_reg, accuracy_calib, accuracy_multicalib))


        ##  Calibration
        # Ground truth
        calibration = calibration_score(y_s, prediction_s, args.lmbda)
        # reg
        calibration_reg = calibration_score(y_s, predictions_reg_s, args.lmbda)
        # calib
        calibration_calib = calibration_score(y_s, calibrated_predictions_s, args.lmbda)
        # multicalib
        calibration_multicalib = calibration_score(y_s, multicalibrated_predictions_s, args.lmbda)
        print('********** Calibration **********')
        print('Ground truth: %2f\t Regularized: %2f\t Calibration %2f\t Multicalibration %2f\t'
              %(calibration, calibration_reg, calibration_calib, calibration_multicalib))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Use multicalibration to report clibrated results of a model')
    parser.add_argument('--data', type=str, default='income', help='Dataset to use for the experiment')
    parser.add_argument('--path', type=str, default='./', help='Root directory path')
    parser.add_argument('--alpha', type=float, default=0.02, help='Alpha parameter for calibration')
    parser.add_argument('--lmbda', type=float, default=10, help='Lambda parameter for calibration')

    args = parser.parse_args()
    main(args)

