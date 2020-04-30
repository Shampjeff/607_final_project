import argparse
import os
import pandas as pd

from sklearn.externals import joblib

from sklearn.ensemble import GradientBoostingClassifier

# This is a general framework for testing models in SageMaker. 
# I'm keeping the structure the same, but using some shortcuts to make the process smoother.

# Model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models are set automatically
    
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
# Add any additional arguments that you will need to pass into your model
# Since we already found parameters that we like, we can simply paste them in 


    # args holds all passed-in arguments
    args = parser.parse_args()

    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    model = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                           learning_rate=0.020833333333333332, loss='deviance',
                           max_depth=11, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=21, min_samples_split=16,
                           min_weight_fraction_leaf=0.0, n_estimators=87,
                           n_iter_no_change=None, presort='auto',
                           random_state=945945, subsample=0.97, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
    model.fit(X=train_x, y=train_y)
    
    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))