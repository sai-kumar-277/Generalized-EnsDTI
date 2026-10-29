"""
icp_predict_v2.py — Leakage-controlled Inductive Conformal Prediction.

Key differences from original icp_predict.py:
  1. Reads moe_predictions.csv from tmp/<dataset>/<split_type>/
  2. Uses ONLY calibration-split rows to compute nonconformity scores
  3. Uses ONLY test-split rows to generate conformal predictions
  4. Does NOT use training rows for calibration
  5. Asserts disjointness via original row IDs before running
  6. Preserves original EnsDTI nonconformity score: alpha = |pred - label|
  7. Saves detailed icp_results.csv to output/<dataset>/<split_type>/

Nonconformity score (from original icp_predict.py, unchanged):
    alpha(x, y) = |f(x) - y|

p-value (from original icp_predict.py, unchanged):
    p(y) = (|{i ∈ calibration : alpha_i >= alpha(x_test, y)}| + 1) / (|calibration| + 1)

Prediction:
    y_hat = argmax(p(y=0), p(y=1))
    confidence = max(p(y=0), p(y=1))

Usage
-----
  python icp_predict_v2.py --dataset davis --split-type random_pair
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ICP functions (identical to original icp_predict.py formulations)
# ---------------------------------------------------------------------------

def calculate_nonconformity(pred: float, lab: float) -> float:
    """
    Original EnsDTI nonconformity score.
    alpha(x, y) = |f(x) - y|
    """
    return abs(pred - lab)


def calculate_pvalue(cal_nonconfs: list, test_nonconf: float) -> float:
    """
    Original EnsDTI p-value calculation.
    p(y) = (count(alpha_i >= alpha_test) + 1) / (|cal| + 1)
    """
    count = sum(1 for score in cal_nonconfs if score >= test_nonconf)
    return (count + 1) / (len(cal_nonconfs) + 1)


def generate_icp_prediction(test_pred: float, cal_nonconfs: list):
    """
    Compute ICP prediction for a single test example.
    Returns (predicted_label, confidence, p_value_0, p_value_1)
    """
    p_0 = calculate_pvalue(cal_nonconfs, calculate_nonconformity(test_pred, 0))
    p_1 = calculate_pvalue(cal_nonconfs, calculate_nonconformity(test_pred, 1))
    predicted_label = int(p_1 > p_0)  # argmax
    confidence = max(p_0, p_1)
    return predicted_label, confidence, p_0, p_1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ICP (leakage-controlled)')
    parser.add_argument('--dataset', '-d', default='davis')
    parser.add_argument('--split-type', '-s', default='random_pair',
                        choices=['random_pair', 'cold_drug', 'cold_protein', 'double_cold'])
    args = parser.parse_args()

    pred_path = os.path.join('tmp', args.dataset, args.split_type, 'moe_predictions.csv')
    out_dir = os.path.join('output', args.dataset, args.split_type)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(pred_path):
        sys.exit(f"MoE predictions not found: {pred_path}\n"
                 "Run moe_predict_v2.py first.")

    print(f"\n{'='*60}")
    print(f"ICP Predict v2: {args.dataset} / {args.split_type}")
    print(f"{'='*60}")

    preds = pd.read_csv(pred_path)

    # Split into train / calibration / test
    train_df = preds[preds['split'] == 'train'].copy()
    cal_df = preds[preds['split'] == 'calibration'].copy()
    test_df = preds[preds['split'] == 'test'].copy()

    train_row_ids = set(train_df['original_row_id'].values)
    cal_row_ids = set(cal_df['original_row_id'].values)
    test_row_ids = set(test_df['original_row_id'].values)

    print(f"\nSplit sizes: train={len(train_df)}, cal={len(cal_df)}, test={len(test_df)}")

    # ---- Mandatory leakage assertions ----------------------------------------
    cal_train_overlap = len(cal_row_ids & train_row_ids)
    test_cal_overlap = len(test_row_ids & cal_row_ids)

    assert cal_train_overlap == 0, \
        (f"LEAKAGE VIOLATION: {cal_train_overlap} calibration rows overlap "
         f"with training rows. ICP calibration must be disjoint from training.")
    assert test_cal_overlap == 0, \
        (f"LEAKAGE VIOLATION: {test_cal_overlap} test rows overlap with "
         f"calibration rows.")

    print("\n  [ASSERT OK] Calibration intersect Training row IDs = empty")
    print("  [ASSERT OK] Test intersect Calibration row IDs = empty")

    # ---- Compute calibration nonconformity scores ----------------------------
    # Use calibration rows ONLY (not training rows)
    cal_nonconfs = [
        calculate_nonconformity(row['moe_predicted_probability'], row['label'])
        for _, row in cal_df.iterrows()
    ]
    print(f"\n  Calibration nonconformity scores: n={len(cal_nonconfs)}, "
          f"mean={np.mean(cal_nonconfs):.4f}, "
          f"min={np.min(cal_nonconfs):.4f}, "
          f"max={np.max(cal_nonconfs):.4f}")

    # ---- Generate ICP predictions for test rows only -------------------------
    print(f"\n  Generating ICP predictions for {len(test_df)} test rows...")
    icp_rows = []
    for _, row in test_df.iterrows():
        pred_label, conf, p0, p1 = generate_icp_prediction(
            row['moe_predicted_probability'], cal_nonconfs)
        icp_rows.append({
            'original_row_id': int(row['original_row_id']),
            'drug_id': row['drug_id'],
            'protein_id': row['protein_id'],
            'true_label': int(row['label']),
            'moe_predicted_probability': float(row['moe_predicted_probability']),
            'predicted_label': int(pred_label),
            'p_value_label_0': float(p0),
            'p_value_label_1': float(p1),
            'confidence': float(conf),
        })

    icp_df = pd.DataFrame(icp_rows)
    icp_path = os.path.join(out_dir, 'icp_results.csv')
    icp_df.to_csv(icp_path, index=False)
    print(f"\n  ICP results saved -> {icp_path}")

    # ---- Quick accuracy summary -----------------------------------------------
    correct = (icp_df['predicted_label'] == icp_df['true_label']).sum()
    accuracy = correct / len(icp_df)
    mean_conf = icp_df['confidence'].mean()
    print(f"\n  ICP test accuracy: {accuracy:.4f} ({correct}/{len(icp_df)})")
    print(f"  Mean confidence:   {mean_conf:.4f}")

    # ---- Print per-threshold summary -----------------------------------------
    print(f"\n  {'Threshold':>10}  {'Count':>8}  {'ErrorRate':>10}  {'Accuracy':>10}")
    print(f"  {'-'*42}")
    for threshold in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
        sub = icp_df[icp_df['confidence'] > threshold]
        if len(sub) == 0:
            continue
        err = (sub['predicted_label'] != sub['true_label']).sum() / len(sub)
        print(f"  {threshold:>10.2f}  {len(sub):>8}  {err:>10.4f}  {1-err:>10.4f}")


if __name__ == '__main__':
    main()
