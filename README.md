# Fairness and Calibration

## Datasets used:

- Census Income Data Set
  https://archive.ics.uci.edu/ml/datasets/census+income

- Credit Default Risk Dataset
  https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    
- Criminal recidivism prediction experiment
  https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis

- Heart Disease Data Set
  https://archive.ics.uci.edu/ml/datasets/Heart+Disease


## Running the code

To execute the training sequence, run the following command from the repository root:
```
python main.py --data=<dataset> --mode=<train> --epochs=<no_of_epochs> --batch_size=<batch_size>
```

For calibration, the model will attempt to load the model for the chosen dataset if it already exists. 
The following command will run _only_ the calibration/multicalibration sequence on the data
```
python main.py --data=<dataset> --mode=<calib>
```
  
  
  
