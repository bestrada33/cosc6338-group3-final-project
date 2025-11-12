# Data manipulation
import pandas as pd

# Machine Learning
from sklearn.model_selection import train_test_split

from src.utils.file_operations import ensure_dir
import src.models.MLP as mlp

def run_ml_pipeline(df_data, results_directory):
    # Preprocess dataset
    X, y = preprocess_data(df_data)

    # Split into train (90%) and test (10%)
    X_train, X_test, y_train_class, y_test_class, y_train_risk, y_test_risk = train_test_split(
        X,
        y['Class_Label'],
        y['Disease_Risk'],
        test_size=0.1,
        random_state=42,
    )

    y_train = {'Class_Label': y_train_class, 'Disease_Risk': y_train_risk}
    y_test = {'Class_Label': y_test_class, 'Disease_Risk': y_test_risk}

    # Find best hyperparameters using GridSearch & k-fold cross-validation and 
    # train final model(s) using full training set and best hyperparameters
    model_ensemble = mlp.model_training(X_train, y_train, results_directory)

    # Predict on independent test set
    df_predictions = mlp.predict(model_ensemble, X_test)

    # Ensure that test results directory exists for model, if not then create it
    test_results_directory = results_directory + 'test/'
    ensure_dir(test_results_directory)

    # Save test predictions to csv
    test_predictions_csv_path = test_results_directory + 'predictions.csv'
    df_predictions.to_csv(test_predictions_csv_path)

    # Get test performance metrics for each label
    results_dfs = []
    for label in y_test:
        y_pred = df_predictions[f'Predicted_{label}'].to_numpy()
        y_true = y_test[label].argmax(axis=1)
        df_class_report = mlp.get_classification_report(y_true, y_pred)
        df_class_report['target'] = label
        results_dfs.append(df_class_report)

    # Combine results dfs for both labels
    df_results = pd.concat(results_dfs, axis=0)
    df_results.reset_index(inplace=True)
    df_results.rename(columns={'index': 'class'}, inplace=True)

    # Reorder columns for readability
    df_results = df_results[['target', 'class', 'precision', 'recall', 'f1-score', 'support']]

    # Save test performance metrics to csv
    test_performance_csv_path = test_results_directory + 'results.csv'
    df_results.to_csv(test_performance_csv_path)


def preprocess_data(df):
    # Remove columns that will not be used as features or targets
    df_preprocess = df.drop(columns=['Sample_ID', 'Sequence_Length', 'Sequence'], axis = 1)

    # Split up features and targets
    df_feature = df_preprocess.drop(columns=['Class_Label', 'Disease_Risk'], axis = 1)
    df_target1 = df_preprocess[['Class_Label']]
    df_target2 = df_preprocess[['Disease_Risk']]

    # One-hot encoding for both multi-class labels ('Class_Label' & 'Disease_Risk')
    df_target1_encoded = pd.get_dummies(df_target1, columns=['Class_Label'])
    df_target2_encoded = pd.get_dummies(df_target2, columns=['Disease_Risk'])

    # Convert features and labels to numpy arrays
    X = df_feature.to_numpy()
    y1 = df_target1_encoded.to_numpy(dtype='float32')
    y2 = df_target2_encoded.to_numpy(dtype='float32')

    # Place labels in one dictionary
    y = {'Class_Label': y1, 'Disease_Risk': y2}

    return X, y