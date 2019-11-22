import numpy as np


def calibrate(data, lables, predictions, setsitive_features, alpha, lmbda):
    calibrated_predictions = predictions.copy()
    predictions = predictions[:,0] # Why!
    lables = lables[:, 0]
    for sensitive_feature in setsitive_features:
        # Find the two subset of the sensitive feature
        sensitive_set = [i for i in range(len(data)) if data[i,sensitive_feature]==1]
        sensitive_set_not = list(set(range(len(data))) - set(sensitive_set))
        print(len(sensitive_set))
        v_range = list(range(0,1,lmbda))

        for S in [sensitive_set, sensitive_set_not]:
            for v in v_range:
                S_v = [i for i in S if predictions[i]<v and predictions[i]>v-(1./lmbda)]
                E_labels = np.mean(lables[S_v])
                # oracle
                E_predictions = np.mean(predictions[S_v])
                if abs(E_labels-E_predictions)<alpha:
                    continue
                update = np.random.uniform(E_labels-alpha/4., E_labels+alpha/4.)
                calibrated_predictions[S_v] = calibrated_predictions[S_v] + update


