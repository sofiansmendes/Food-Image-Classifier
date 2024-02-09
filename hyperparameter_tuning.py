from utils import *
import pandas as pd
import subprocess
import optuna
from sklearn.model_selection import train_test_split


# run the Python script "unzip_image_folder.py"
subprocess.run(['python', 'unzip_image_folder.py'])

# Import the train_val and test metadata
if os.path.exists('train_val.csv') and os.path.exists('test.csv'):
    train_val_df = pd.read_csv('train_val.csv')
    test_df = pd.read_csv('test.csv')
else:
    df_imgs = prepare_datafile('image_metadata.csv', 'data')
    train_val_df, test_df = train_test_split(df_imgs, train_size=0.8)
    train_val_df.to_csv('train_val.csv', index=False)
    test_df.to_csv('test.csv', index=False)


# Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy 
def objective(trial):

    params = {
              'lr': trial.suggest_categorical('lr', [1e-3, 1e-2]),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam"]),
              'num_epochs': trial.suggest_int("num_epochs", 5, 18, step=3),
              'L2': trial.suggest_categorical('L2', [1e-2, 1e-1, 0]),
              'batch_size': trial.suggest_int("batch_size", 50, 150, step=50)
              }
    
    _, _, cv_val_auc = cross_val_loop(train_val_df, test_df, params)

    return cv_val_auc


# This takes a lot of time to run, so do not try unless you have a GPU available
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)

print(study.best_trial)

df_trials = study.trials_dataframe().sort_values(by=['value'], ascending=True)

df_trials.to_csv('hyperparameter_opt.csv', index=False)

best_model, final_val_loss, final_val_auc = cross_val_loop(train_val_df, test_df, study.best_params)