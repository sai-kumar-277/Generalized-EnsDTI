"""
moe_predict.py — Mixture-of-Experts ensemble prediction (bug-fixed version).

Bug fixed (original line 52): `model.predict_proba(input_csv)` used the unfitted
base estimator instead of the fitted `clf` grid-search object.

This script reads precomputed expert predictions from tmp/{dataset}/ and outputs
the MoE prediction to output/{dataset}/result.csv.
"""
import os
import sys
import joblib
import argparse
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

folds = 5
models = ['dcdti', 'dpdta', 'mdprd', 'dp', 'perceivercpi', 'smtdta']
param = {
    'max_iter': [15, 20, 25, 30],
    'hidden_layer_sizes': [10, 15, 20],
    'solver': ['adam'],
    'learning_rate_init': [0.01, 0.001, 0.0001, 0.00001]
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", '-d', help="Dataset name", default='test_data')
    dataset = parser.parse_args().dataset

    # Create output directories
    os.makedirs(f'tmp/{dataset}', exist_ok=True)
    os.makedirs(f'output/{dataset}', exist_ok=True)

    # Load expert predictions
    available_models = []
    input_csv_parts = []
    for model_name in models:
        fpath = f'tmp/{dataset}/{model_name}_result.csv'
        if os.path.exists(fpath):
            re_csv = pd.read_csv(fpath, index_col=[0, 1])
            input_csv_parts.append(re_csv)
            available_models.append(model_name)
        else:
            print(f"[SKIP] {fpath} not found, excluding {model_name}")

    if not input_csv_parts:
        sys.exit(f"No expert predictions found in tmp/{dataset}/. "
                 "Run main.py or expert_predict.py first.")

    input_csv = pd.concat(input_csv_parts, axis=1)
    input_csv.to_csv(f'tmp/{dataset}/all_prob.csv')

    train_index = pd.read_csv(f'tmp/{dataset}/train_index.csv', header=None).values.flatten()
    test_index = pd.read_csv(f'tmp/{dataset}/test_index.csv', header=None).values.flatten()

    train_X = input_csv.iloc[train_index, :-1]
    train_y = input_csv.iloc[train_index, -1]
    test_X = input_csv.iloc[test_index, :-1]
    test_y = input_csv.iloc[test_index, -1]

    base_model = MLPClassifier(early_stopping=True, activation='logistic', random_state=42)
    clf = GridSearchCV(base_model, param, cv=3, n_jobs=-1)
    clf.fit(train_X, train_y)

    # BUG FIX: use fitted clf (not the unfitted base_model) for prediction
    result_proba = clf.predict_proba(input_csv.iloc[:, :-1])
    result_csv = pd.DataFrame(result_proba, index=input_csv.index,
                              columns=['prob_0', 'prob_1'])
    # Use the full label column from input_csv (not just test labels)
    result_csv['label'] = input_csv.iloc[:, -1].values

    file_to_save = f'output/{dataset}/result.csv'
    os.makedirs(os.path.dirname(file_to_save), exist_ok=True)
    result_csv.to_csv(file_to_save, index=True)

    # Save fitted model
    joblib.dump(clf, f'output/{dataset}/moe_model.joblib')
    print(f'Successfully generated final prediction -> {file_to_save}')
