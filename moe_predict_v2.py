"""
moe_predict_v2.py — Leakage-controlled Mixture-of-Experts prediction.

Key differences from original moe_predict.py:
  1. Reads predictions from tmp/<dataset>/<split_type>/<expert>_predictions_*.csv
  2. Trains MLP gating ONLY on training-split predictions
  3. Predicts calibration and test separately
  4. Bug fixed: uses fitted clf object for all predictions
  5. Saves moe_predictions.csv with original_row_id, split, probability columns
  6. Saves fitted model to tmp/<dataset>/<split_type>/moe_model.joblib

Usage
-----
  python moe_predict_v2.py --dataset davis --split-type random_pair --seed 42
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

EXPERT_NAMES = ['dpdta', 'dcdti', 'dacmf']

MLP_PARAM_GRID = {
    'max_iter': [15, 20, 25, 30],
    'hidden_layer_sizes': [10, 15, 20],
    'solver': ['adam'],
    'learning_rate_init': [0.01, 0.001, 0.0001],
}


def load_expert_predictions(expert_names, data_dir, split_name):
    """
    Load predicted_probability columns for all available experts for a given split.
    Returns (merged_df, available_experts).
    """
    dfs = []
    available = []
    for expert in expert_names:
        path = os.path.join(data_dir, f'{expert}_predictions_{split_name}.csv')
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={'predicted_probability': f'prob_{expert}'})
        dfs.append(df.set_index('original_row_id'))
        available.append(expert)
    if not dfs:
        return None, []
    merged = pd.concat([d[[f'prob_{e}']] for d, e in zip(dfs, available)], axis=1)
    # Get label from first dataframe
    label_df = pd.read_csv(
        os.path.join(data_dir, f'{available[0]}_predictions_{split_name}.csv')
    ).set_index('original_row_id')
    merged['label'] = label_df['label']
    merged['drug_id'] = label_df['drug_id']
    merged['protein_id'] = label_df['protein_id']
    merged['split'] = split_name
    return merged, available


def main():
    parser = argparse.ArgumentParser(description='MoE gating (leakage-controlled)')
    parser.add_argument('--dataset', '-d', default='davis')
    parser.add_argument('--split-type', '-s', default='random_pair',
                        choices=['random_pair', 'cold_drug', 'cold_protein', 'double_cold'])
    parser.add_argument('--experts', nargs='+', default=EXPERT_NAMES,
                        help='List of expert model names')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = os.path.join('tmp', args.dataset, args.split_type)
    out_dir = os.path.join('output', args.dataset, args.split_type)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MoE Predict v2: {args.dataset} / {args.split_type}")
    print(f"Requested experts: {args.experts}")
    print(f"{'='*60}")

    # Load training predictions
    print("\nLoading training expert predictions...")
    train_data, available_experts = load_expert_predictions(args.experts, data_dir, 'train')
    if train_data is None or len(available_experts) == 0:
        sys.exit(f"No expert predictions found in {data_dir}/")

    print(f"  Available experts: {available_experts}")
    feature_cols = [f'prob_{e}' for e in available_experts]

    X_train = train_data[feature_cols].values
    y_train = train_data['label'].values.astype(int)

    # Load calibration predictions
    print("Loading calibration expert predictions...")
    cal_data, _ = load_expert_predictions(available_experts, data_dir, 'calibration')

    # Load test predictions
    print("Loading test expert predictions...")
    test_data, _ = load_expert_predictions(available_experts, data_dir, 'test')

    # Verify no row overlap between calibration and training
    train_ids = set(train_data.index)
    cal_ids = set(cal_data.index)
    test_ids = set(test_data.index)

    assert len(cal_ids & train_ids) == 0, \
        f"LEAKAGE: calibration rows overlap with training ({len(cal_ids & train_ids)} rows)"
    assert len(test_ids & cal_ids) == 0, \
        f"LEAKAGE: test rows overlap with calibration ({len(test_ids & cal_ids)} rows)"
    print("\n  [ASSERT OK] Calibration intersect Training = empty")
    print("  [ASSERT OK] Test intersect Calibration = empty")

    # Train MoE gating MLP (only on training predictions)
    print(f"\nTraining MoE MLP on {len(X_train)} training examples...")
    base_mlp = MLPClassifier(early_stopping=True, activation='logistic',
                              random_state=args.seed)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    clf = GridSearchCV(base_mlp, MLP_PARAM_GRID, cv=cv, n_jobs=-1,
                       scoring='roc_auc', refit=True)
    clf.fit(X_train, y_train)
    print(f"  Best params: {clf.best_params_}")
    print(f"  Best CV AUROC: {clf.best_score_:.4f}")

    # Save fitted model
    model_path = os.path.join(data_dir, 'moe_model.joblib')
    joblib.dump(clf, model_path)
    print(f"  MoE model saved -> {model_path}")

    # Generate predictions for all splits
    all_rows = []

    def predict_and_collect(data, split_name, label):
        if data is None:
            return
        X = data[feature_cols].values
        proba = clf.predict_proba(X)[:, 1]
        for i, (row_id, row) in enumerate(data.iterrows()):
            all_rows.append({
                'original_row_id': row_id,
                'drug_id': row['drug_id'],
                'protein_id': row['protein_id'],
                'label': int(row['label']),
                'split': split_name,
                'moe_predicted_probability': float(proba[i]),
            })

    predict_and_collect(train_data, 'train', y_train)
    predict_and_collect(cal_data, 'calibration', None)
    predict_and_collect(test_data, 'test', None)

    result_df = pd.DataFrame(all_rows)
    pred_path = os.path.join(data_dir, 'moe_predictions.csv')
    result_df.to_csv(pred_path, index=False)
    # Also save dedicated version named by expert count
    tag = f"moe_predictions_{len(available_experts)}expert.csv"
    result_df.to_csv(os.path.join(data_dir, tag), index=False)
    print(f"\nMoE predictions saved -> {pred_path} and {tag}")

    # Quick summary stats
    for split_name in ['train', 'calibration', 'test']:
        subset = result_df[result_df['split'] == split_name]
        if len(subset) > 0:
            print(f"  {split_name}: n={len(subset)}, "
                  f"mean_prob={subset['moe_predicted_probability'].mean():.4f}, "
                  f"pos_rate={subset['label'].mean():.4f}")


if __name__ == '__main__':
    main()
