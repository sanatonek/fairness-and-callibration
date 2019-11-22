import numpy as np
from sklearn import preprocessing


def calibrate(data, lables, predictions, setsitive_features, alpha, lmbda):
    predictions = predictions[:,0] # Why!
    lables = lables[:, 0]
    calibrated_predictions = predictions.copy()
    print('Total number of samples to begin with: ', len(predictions))
    print('AE pre-calibration: ', abs(np.mean(lables)-np.mean(predictions)))
    v_range = np.arange(0,1,1./lmbda)
    change = 1
    while change>0:
        print('Only %d sets changed')
        change=0
        for sensitive_feature in setsitive_features:
            # Find the two subset of the sensitive feature
            sensitive_set = [i for i in range(len(data)) if data[i, sensitive_feature] == 1]
            sensitive_set_not = list(set(range(len(data))) - set(sensitive_set))
            print('Samples in each subgroup: ', len(sensitive_set), len(sensitive_set_not))
            for S in [sensitive_set, sensitive_set_not]:
                for v in v_range:
                    S_v = [i for i in S if calibrated_predictions[i]<v+(1./lmbda) and calibrated_predictions[i]>=v]
                    print('Cheking bin %.1f ...... size of s_v: '%v, len(S_v))
                    if len(S_v)==0:
                        continue
                    E_labels = np.mean(lables[S_v])
                    # oracle
                    E_predictions = np.mean(calibrated_predictions[S_v])
                    if abs(E_labels-E_predictions)<alpha/2.:
                        continue
                    update = np.random.uniform(E_labels-alpha/4., E_labels+alpha/4.)
                    print('Update value: ', update)
                    calibrated_predictions[S_v] = calibrated_predictions[S_v] + (update-E_predictions)
                    if (calibrated_predictions[S_v]<0).any() or (calibrated_predictions[S_v]>1).any():
                        calibrated_predictions[S_v] = normalize(calibrated_predictions[S_v])
                    if set(S_v)!=set([i for i in S if calibrated_predictions[i] < v + (1. / lmbda) and calibrated_predictions[i] >= v]):
                        change += 1
    print(calibrated_predictions)
    print(lables)
    print(predictions)
    print('AE post-calibration: ', abs(np.mean(lables)-np.mean(calibrated_predictions)))


def multi_calibrate(data, lables, predictions, alpha, lmbda):
    predictions = predictions[:,0] # Why!
    lables = lables[:, 0]
    calibrated_predictions = predictions.copy()
    print('Total number of samples to begin with: ', len(predictions))
    print('AE pre-calibration: ', abs(np.mean(lables)-np.mean(predictions)))
    v_range = np.arange(0,1,1./lmbda)
    change = 1
    while change>0:
        print('Only %d sets changed')
        change=0
        for sensitive_feature in setsitive_features:
            # Find the two subset of the sensitive feature
            sensitive_set = [i for i in range(len(data)) if data[i, sensitive_feature] == 1]
            sensitive_set_not = list(set(range(len(data))) - set(sensitive_set))
            print('Samples in each subgroup: ', len(sensitive_set), len(sensitive_set_not))
            for S in [sensitive_set, sensitive_set_not]:
                for v in v_range:
                    S_v = [i for i in S if calibrated_predictions[i]<v+(1./lmbda) and calibrated_predictions[i]>=v]
                    print('Cheking bin %.1f ...... size of s_v: '%v, len(S_v))
                    if len(S_v)==0:
                        continue
                    E_labels = np.mean(lables[S_v])
                    # oracle
                    E_predictions = np.mean(calibrated_predictions[S_v])
                    if abs(E_labels-E_predictions)<alpha/2.:
                        continue
                    update = np.random.uniform(E_labels-alpha/4., E_labels+alpha/4.)
                    print('Update value: ', update)
                    calibrated_predictions[S_v] = calibrated_predictions[S_v] + (update-E_predictions)
                    if (calibrated_predictions[S_v]<0).any() or (calibrated_predictions[S_v]>1).any():
                        calibrated_predictions[S_v] = normalize(calibrated_predictions[S_v])
                    if set(S_v)!=set([i for i in S if calibrated_predictions[i] < v + (1. / lmbda) and calibrated_predictions[i] >= v]):
                        change += 1
    print(calibrated_predictions)
    print(lables)
    print(predictions)
    print('AE post-calibration: ', abs(np.mean(lables)-np.mean(calibrated_predictions)))


def normalize(x):
    if min(x)==max(x):
        return x/min(x)
    else:
        return (x-min(x))/(max(x)-min(x))
