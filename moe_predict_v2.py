"""
moe_predict_v2.py — Leakage-controlled Mixture-of-Experts prediction.

Key differences from original moe_predict.py:
  1. Reads predictions from tmp/<dataset>/<split_type>/<expert>_predictions_*.csv
  2. Trains MLP gating ONLY on training-split predictions
  3. Predicts calibration and test separately
  4. Bug fixed: uses fitted clf object for all predictions
  5. Saves moe_predictions.csv with original_row_id, split, probability columns
  6. Saves fitted model to tmp/<dataset>/<split_type>/moe_model.joblib
  7. Supports Shift-Aware / Novelty-Aware Gating modes:
     --gating-mode:
       - original: Unregularized MLP gating via GridSearchCV
       - regularized: L2-regularized MLP (alpha=1.0)
       - adaptive: Shift-aware fallback (simple average under entity novelty)
       - novelty_aware: MLP conditioned on entity novelty indicator flags

Usage
-----
  python moe_predict_v2.py --dataset davis --split-type random_pair --seed 42 --gating-mode adaptive
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


def load_expert_predictions(expert_names, data_dir, split_name, seed=42):
    """
    Load predicted_probability columns for all available experts for a given split.
    Checks data_dir first, then data_dir/seed_<seed>.
    Returns (merged_df, available_experts).
    """
    dfs = []
    available = []
    seed_dir = os.path.join(data_dir, f'seed_{seed}')
    for expert in expert_names:
        path = os.path.join(data_dir, f'{expert}_predictions_{split_name}.csv')
        if not os.path.exists(path) and os.path.exists(seed_dir):
            candidate = os.path.join(seed_dir, f'{expert}_predictions_{split_name}.csv')
            if os.path.exists(candidate):
                path = candidate
        if not os.path.exists(path):
            print(f"  [SKIP] {expert}_predictions_{split_name}.csv not found")
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={'predicted_probability': f'prob_{expert}'})
        dfs.append(df.set_index('original_row_id'))
        available.append(expert)
    if not dfs:
        return None, []
    merged = pd.concat([d[[f'prob_{e}']] for d, e in zip(dfs, available)], axis=1)
    
    # Get metadata from first dataframe
    label_path = os.path.join(data_dir, f'{available[0]}_predictions_{split_name}.csv')
    if not os.path.exists(label_path) and os.path.exists(seed_dir):
        label_path = os.path.join(seed_dir, f'{available[0]}_predictions_{split_name}.csv')
    label_df = pd.read_csv(label_path).set_index('original_row_id')
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
    parser.add_argument('--gating-mode', choices=['original', 'regularized', 'adaptive', 'novelty_aware'],
                        default='adaptive', help='Gating strategy to use')
    args = parser.parse_args()

    data_dir = os.path.join('tmp', args.dataset, args.split_type)
    out_dir = os.path.join('output', args.dataset, args.split_type)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MoE Predict v2: {args.dataset} / {args.split_type} (mode: {args.gating_mode})")
    print(f"Requested experts: {args.experts}")
    print(f"{'='*60}")

    # Load training predictions
    print("\nLoading training expert predictions...")
    train_data, available_experts = load_expert_predictions(args.experts, data_dir, 'train', seed=args.seed)
    if train_data is None or len(available_experts) == 0:
        sys.exit(f"No expert predictions found in {data_dir}/")

    print(f"  Available experts: {available_experts}")
    feature_cols = [f'prob_{e}' for e in available_experts]

    X_train_raw = train_data[feature_cols].values
    y_train = train_data['label'].values.astype(int)

    # Load calibration predictions
    print("Loading calibration expert predictions...")
    cal_data, _ = load_expert_predictions(available_experts, data_dir, 'calibration', seed=args.seed)

    # Load test predictions
    print("Loading test expert predictions...")
    test_data, _ = load_expert_predictions(available_experts, data_dir, 'test', seed=args.seed)

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

    train_drugs = set(train_data['drug_id'].values)
    train_prots = set(train_data['protein_id'].values)

    # Train MoE gating model according to gating_mode
    if args.gating_mode == 'original':
        print(f"\nTraining Original MoE MLP on {len(X_train_raw)} training examples...")
        base_mlp = MLPClassifier(early_stopping=True, activation='logistic',
                                  random_state=args.seed)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
        clf = GridSearchCV(base_mlp, MLP_PARAM_GRID, cv=cv, n_jobs=-1,
                           scoring='roc_auc', refit=True)
        clf.fit(X_train_raw, y_train)
        print(f"  Best params: {clf.best_params_}")
        print(f"  Best CV AUROC: {clf.best_score_:.4f}")

    elif args.gating_mode == 'regularized':
        print(f"\nTraining Regularized MoE MLP (alpha=1.0) on {len(X_train_raw)} training examples...")
        clf = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', alpha=1.0,
                            max_iter=100, random_state=args.seed)
        clf.fit(X_train_raw, y_train)

    elif args.gating_mode == 'novelty_aware':
        print(f"\nTraining Novelty-Aware MoE MLP on {len(X_train_raw)} training examples with simulated entity holdouts...")
        # Simulate cold entities inside training set to learn novelty sensitivity
        np.random.seed(args.seed)
        train_drugs_list = list(train_drugs)
        train_prots_list = list(train_prots)
        sim_novel_drugs = set(np.random.choice(train_drugs_list, size=int(0.25 * len(train_drugs_list)), replace=False))
        sim_novel_prots = set(np.random.choice(train_prots_list, size=int(0.25 * len(train_prots_list)), replace=False))

        is_d_sim = train_data['drug_id'].isin(sim_novel_drugs).astype(float).values
        is_p_sim = train_data['protein_id'].isin(sim_novel_prots).astype(float).values
        is_inter_sim = is_d_sim * is_p_sim
        X_train_nov = np.column_stack([X_train_raw, is_d_sim, is_p_sim, is_inter_sim])

        clf = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', alpha=1.0,
                            max_iter=100, random_state=args.seed)
        clf.fit(X_train_nov, y_train)

    elif args.gating_mode == 'adaptive':
        print(f"\nTraining In-Distribution Base MLP (alpha=1.0) for Adaptive Fallback...")
        clf = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', alpha=1.0,
                            max_iter=100, random_state=args.seed)
        clf.fit(X_train_raw, y_train)

    # Save fitted model
    model_path = os.path.join(data_dir, 'moe_model.joblib')
    joblib.dump(clf, model_path)
    print(f"  MoE model saved -> {model_path}")

    # Generate predictions for all splits
    all_rows = []

    def predict_split(data, split_name):
        if data is None:
            return
        X = data[feature_cols].values
        n_samples = len(data)

        if args.gating_mode in ['original', 'regularized']:
            proba = clf.predict_proba(X)[:, 1]
        elif args.gating_mode == 'novelty_aware':
            is_d = (~data['drug_id'].isin(train_drugs)).astype(float).values
            is_p = (~data['protein_id'].isin(train_prots)).astype(float).values
            is_inter = is_d * is_p
            X_nov = np.column_stack([X, is_d, is_p, is_inter])
            proba = clf.predict_proba(X_nov)[:, 1]
        elif args.gating_mode == 'adaptive':
            # Check novelty per sample: if novel drug or protein, fallback to simple average
            is_d = (~data['drug_id'].isin(train_drugs)).values
            is_p = (~data['protein_id'].isin(train_prots)).values
            is_shift = is_d | is_p
            p_mlp = clf.predict_proba(X)[:, 1]
            p_avg = np.mean(X, axis=1)
            proba = np.where(is_shift, p_avg, p_mlp)

        for i, (row_id, row) in enumerate(data.iterrows()):
            all_rows.append({
                'original_row_id': row_id,
                'drug_id': row['drug_id'],
                'protein_id': row['protein_id'],
                'label': int(row['label']),
                'split': split_name,
                'moe_predicted_probability': float(proba[i]),
            })

    predict_split(train_data, 'train')
    predict_split(cal_data, 'calibration')
    predict_split(test_data, 'test')

    result_df = pd.DataFrame(all_rows)
    pred_path = os.path.join(data_dir, 'moe_predictions.csv')
    result_df.to_csv(pred_path, index=False)
    tag = f"moe_predictions_{len(available_experts)}expert.csv"
    result_df.to_csv(os.path.join(data_dir, tag), index=False)
    print(f"\nMoE predictions saved -> {pred_path} and {tag}")

    for split_name in ['train', 'calibration', 'test']:
        subset = result_df[result_df['split'] == split_name]
        if len(subset) > 0:
            print(f"  {split_name}: n={len(subset)}, "
                  f"mean_prob={subset['moe_predicted_probability'].mean():.4f}, "
                  f"pos_rate={subset['label'].mean():.4f}")


if __name__ == '__main__':
    main()
