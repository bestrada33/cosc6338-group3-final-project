import time

# Data manipulation
import pandas as pd
import numpy as np

# Machine Learning
from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model, save_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras.metrics as km
from keras_tuner import GridSearch, Objective
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report
import tensorflow as tf

from functools import partial

from src.utils.file_operations import ensure_dir

def build_MLP_model(hp=None, params=None, input_dim=None, y_targets=None,
                    loss_function = 'categorical_crossentropy'):
      """
      Builds a multi-output Keras model for multi-label, multi-class classification.

      Args:
          hp: HyperParameters object (used during Keras Tuner search)
          params: dict of hyperparameters (used during manual model build)
          input_dim: number of input features
          y_targets: dict of target arrays, e.g. {'Class_Label': y1, 'Disease_Risk': y2}
          loss_function: loss to use for each output (default: categorical_crossentropy)

      One of hp or params must be provided.
      """
      if hp is not None:
          num_layers = hp.Choice('num_layers', [1])
          neurons = hp.Choice('neurons', [8, 12, 20, 30])
          learning_rate = hp.Choice('lr', [1e-3, 1e-4, 1e-5])
      elif params is not None:
          num_layers = params['num_layers']
          neurons = params['neurons']
          learning_rate = params['lr']
      else:
          raise ValueError("Either 'hp' or 'params' must be provided.")

      # Input Layer
      input_layer = Input(shape=(input_dim,))
      x = input_layer

      # Hidden layers with Dropout
      for i in range(num_layers):
          x = Dense(neurons, activation='tanh', kernel_initializer='he_normal')(x)
          x = Dropout(0.4)(x)

      # Output Layer (softmax activation function for classification output)
      outputs = {}
      for name, y_arr in y_targets.items():
          n_classes = y_arr.shape[1]
          outputs[name] = Dense(n_classes, activation='softmax', name=name)(x)

      # Build model
      model = Model(inputs=input_layer, outputs=list(outputs.values()))

      # Both outputs will share metrics
      shared_metrics = [
          km.CategoricalAccuracy(name='accuracy'),
          km.Precision(name='precision'),
          km.Recall(name='recall'),
          km.AUC(name='auc')
      ]

      # Compile (Each output must have own loss function and metrics defined)
      model.compile(
          loss={name: loss_function for name in outputs.keys()},
          optimizer=Adam(learning_rate=learning_rate),
          metrics={name: shared_metrics for name in outputs.keys()}
      )

      return model


def tune_model(x_train, y_train, validation_data, directory, project_name,
               loss_function = 'categorical_crossentropy', objective='val_loss',
               direction='min', batch_size=64, validation_batch_size=64,
               epochs=2000, patience=20):
        """
        Hyperparameter tuning using GridSearch from Keras Tuner.

        Args:
            x_train: training inputs (numpy array).
            y_train: training outputs (dict/list of numpy arrays).
            validation_data: (X_val, y_val) tuple where y_val can be a dict/list of outputs
            directory: folder where results will be saved.
            project_name: project identifier for KerasTuner.
            loss_function: loss to use for each output (default: categorical_crossentropy)
            objective: metric to optimize (default: val_loss).
            direction: 'min' or 'max' (default: 'min').
            batch_size: training batch size.
            validation_batch_size: validation batch size.
            epochs: training epochs per trial.
            patience: early stopping patience.
        """

        tuner = GridSearch(
            hypermodel=partial(build_MLP_model, loss_function = loss_function,
                               input_dim=x_train.shape[1], y_targets=y_train),
            objective=Objective(objective, direction=direction),
            seed=42,
            directory=directory,
            project_name=project_name,
            overwrite=False
        )

        early_stopping = EarlyStopping(
            monitor=objective,
            patience=patience,
            mode=direction,
            restore_best_weights=True,
            verbose=0
        )

        callbacks = [early_stopping]

        tuner.search(
            x_train, y_train,
            validation_data=validation_data,
            epochs = epochs,
            batch_size=batch_size,
            validation_batch_size=validation_batch_size,
            callbacks=callbacks,
            verbose=0
        )

        df_tuning_results = extract_tuning_results(tuner)

        print(tuner.results_summary())

        return df_tuning_results


def extract_tuning_results(tuner):
    trial_data = []

    for trial in tuner.oracle.trials.values():
        entry = {
            "trial_id": trial.trial_id,
            "status": trial.status,
            **trial.hyperparameters.values
        }

        # Find the epoch with the best score based on the given objective metric for this trial
        # This epoch is where the model weights are saved after early stopping kicks in
        best_epoch = trial.best_step
    
        # Add it to the entry dictionary
        entry["best_epoch"] = best_epoch

        # Find and save all metric values at the best epoch from above
        for metric_name, metric in trial.metrics.metrics.items():
            best_observation = metric._observations.get(best_epoch)

            entry[metric_name] = best_observation.value[0]

        trial_data.append(entry)

    df = pd.DataFrame(trial_data)
    
    return df


def find_best_trial_id(df_tuning_results, metric = 'val_loss', direction = 'min'):
    # Group by trial ID and calculate mean of metric
    df_grouped = df_tuning_results.groupby('trial_id')[metric].mean().reset_index()

    # Find the trial ID with the best mean metric score over all folds
    if direction == 'max':
        best_trial_id = df_grouped.loc[df_grouped[metric].idxmax(), 'trial_id']
    else:
        best_trial_id = df_grouped.loc[df_grouped[metric].idxmin(), 'trial_id']

    return best_trial_id


def kfold_cross_validation(kfolds, X_train, y_train, results_directory):
    tuning_results_directory = results_directory + 'tuning/'

    df_results_list = []
    for i, (train_index, test_index) in enumerate(kfolds.split(X_train)):
        print(f"Fold {i}:")

        # Define train and test X for fold
        X_fold_train = X_train[train_index]
        X_fold_test = X_train[test_index]

        # Define y_train for fold
        y_fold_train = {'Class_Label': y_train['Class_Label'][train_index], 
                        'Disease_Risk': y_train['Disease_Risk'][train_index]}
        
        # Group up validation data for fold
        validation_data = (X_fold_test, {'Class_Label': y_train['Class_Label'][test_index], 
                                         'Disease_Risk': y_train['Disease_Risk'][test_index]})

        # Set the project folder name as the current fold number
        tuning_project_name = f'fold_{i}'


        start = time.time()
        df_tuning_results = tune_model(X_fold_train, y_fold_train, validation_data, 
                                       tuning_results_directory, tuning_project_name)
        end = time.time()

        print(f"Tuning took {(end - start)/60:.2f} minutes")
        
        # Add fold # to tuning df for this fold
        df_tuning_results['fold'] = i

        # Add results for this fold to full list of results
        df_results_list.append(df_tuning_results)

    # Combine results across all folds  
    df_all_results = pd.concat(df_results_list, ignore_index=True)
    
    return df_all_results


def train_final_models(X_train, y_train, best_hyperparams, results_directory):
    # Split into train (90%) and validation (10%)
    X_train, X_valid, y_train_class, y_valid_class, y_train_risk, y_valid_risk = train_test_split(
        X_train,
        y_train['Class_Label'],
        y_train['Disease_Risk'],
        test_size=0.1,
        random_state=42,
    )

    y_train = {'Class_Label': y_train_class, 'Disease_Risk': y_train_risk}
    validation_data = (X_valid, {'Class_Label': y_valid_class, 'Disease_Risk': y_valid_risk})

    model_directory = results_directory + 'models/'
    ensure_dir(model_directory)

    start = time.time()
    
    # Train ensemble models using best hyperparameters
    ensemble_models = train_ensemble(X_train, y_train, validation_data, best_hyperparams, model_directory)
    end = time.time()

    print(f"Ensemble training took {(end - start)/60:.2f} minutes")

    return ensemble_models


def train_ensemble(X_train, y_train, validation_data, best_hyperparams, model_directory, n_models=30,
                    loss_function = 'categorical_crossentropy', objective = 'val_loss', direction = 'min', 
                    epochs = 10000, batch_size = 64, validation_batch_size = 64, patience = 20):
    model_ensemble = []
    for i in range(n_models):
        model_file_path = model_directory + f'hypermodel{i+1}.h5'

        try:
            model = load_model(model_file_path)
        except OSError:
            # Set a different seed per model for weight initialization diversity
            tf.random.set_seed(i)

            # Build model with optimal hyperparameters from k-fold cross-validation
            model = build_MLP_model(params = best_hyperparams, loss_function = loss_function,
                                     input_dim=X_train.shape[1], y_targets=y_train)

            # Define callbacks
            early_stopping = EarlyStopping(monitor = f'{objective}', mode = direction, patience = patience, 
                                        restore_best_weights = True, verbose = 0)

            callbacks = [early_stopping]

            model.fit(
                X_train, y_train,
                validation_data = validation_data,
                epochs = epochs,
                batch_size = batch_size,
                validation_batch_size = validation_batch_size,
                callbacks = callbacks,
                verbose = 0
                )
            
            save_model(model, model_file_path)
        
        model_ensemble.append(model)

    return model_ensemble


def model_training(X_train, y_train, results_directory):
    # Create folds for k-fold validation
    num_folds = 10
    kfolds = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Perform k-fold cross-validation and return metrics for all folds
    cv_metrics = kfold_cross_validation(kfolds, X_train, y_train, results_directory)
    
    # Find the trial ID with the best metric score on average over all folds
    best_trial_id = find_best_trial_id(cv_metrics)

    # Find all rows with the same trial ID to extract the metrics of best hyperparameters
    df_best_trial = cv_metrics[cv_metrics['trial_id'] == best_trial_id].copy()

    # Save metrics with best trial ID to csv
    tuning_results_directory = results_directory + 'tuning/'
    cv_metrics_path = tuning_results_directory + 'best_cv_metrics.csv'
    df_best_trial.to_csv(cv_metrics_path)

    # Extract best hyperparameters from first row of the best metrics df
    best_hyperparams = df_best_trial.loc[df_best_trial.index[0], ['num_layers', 'neurons', 'lr']].to_dict()
    
    # Train final model ensemble using full training set and best hyperparameters
    model_ensemble = train_final_models(X_train, y_train, best_hyperparams, results_directory)

    return model_ensemble


def predict(model_ensemble, X):
    # Ensemble predictions lists
    preds_class = []
    preds_risk = []

    for member in model_ensemble:
        # Each ensemble member outputs [Class_Label_pred, Disease_Risk_pred]
        y_pred_class, y_pred_risk = member.predict(X, verbose=0)

        preds_class.append(y_pred_class)
        preds_risk.append(y_pred_risk)

    # Convert to arrays: shape (n_members, n_samples, n_classes)
    preds_class = np.array(preds_class)
    preds_risk  = np.array(preds_risk)

    # Soft voting: average the class probabilities
    avg_pred_class = np.mean(preds_class, axis=0)
    avg_pred_risk  = np.mean(preds_risk, axis=0)

    # Get final class predictions
    final_pred_class = np.argmax(avg_pred_class, axis=1)
    final_pred_risk  = np.argmax(avg_pred_risk, axis=1)

    # Convert average class probabilities across all ensemble members to DataFrames
    # Saves the average probability per label, per class
    # Each row is a sample
    df_class_probs = pd.DataFrame(
        avg_pred_class,
        columns=[f'ClassLabel_Prob_{i}' for i in range(avg_pred_class.shape[1])]
    )

    df_risk_probs = pd.DataFrame(
        avg_pred_risk,
        columns=[f'DiseaseRisk_Prob_{i}' for i in range(avg_pred_risk.shape[1])]
    )

    # Add final predictions (argmax results)
    df_class_probs['Predicted_Class_Label'] = final_pred_class
    df_risk_probs['Predicted_Disease_Risk'] = final_pred_risk

    # Combine into one df
    df_predictions = pd.concat([df_class_probs, df_risk_probs], axis=1)
    
    return df_predictions
    

def get_classification_report(y_true, y_pred):
    class_report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    df_class_report = pd.DataFrame(class_report).transpose()

    return df_class_report