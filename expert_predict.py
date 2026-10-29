"""
expert_predict.py — Sklearn-based expert model substitutes for EnsDTI.

The original EnsDTI expert models (DeepConvDTI, DeepDTA, DeepPurpose, MDeePred)
require torch_geometric, ogb, and the private drug_target_interaction package,
none of which are available. This module provides lightweight sklearn-based
substitute experts that operate on the same raw features (SMILES -> Morgan FP,
sequence -> amino-acid composition) to enable end-to-end pipeline testing.

These are SUBSTITUTE models, NOT reproductions of the paper's models.
Results from this module are NOT directly comparable to Table II of the paper.

Expert models:
  1. lr_expert : Logistic Regression on drug+protein features
  2. rf_expert : Random Forest on drug+protein features
  3. gb_expert : Histogram-based Gradient Boosting on drug+protein features

Optimized with vectorized feature table lookups for sub-minute execution.
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
DRUG_FP_BITS = 1024  # 1024 bits is fast and compact
EXPERT_NAMES = ['lr_expert', 'rf_expert', 'gb_expert']

# Create Morgan generator once
_mfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=DRUG_FP_BITS)


def smiles_to_morgan_fp(smi: str) -> np.ndarray:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.zeros(DRUG_FP_BITS, dtype=np.float32)
        fp = _mfp_gen.GetFingerprintAsNumPy(mol)
        return fp.astype(np.float32)
    except Exception:
        return np.zeros(DRUG_FP_BITS, dtype=np.float32)


def seq_to_aac(seq: str) -> np.ndarray:
    seq = str(seq).upper()
    n = max(len(seq), 1)
    return np.array([seq.count(aa) / n for aa in AMINO_ACIDS], dtype=np.float32)


def build_feature_extractor(full_df: pd.DataFrame):
    """Precompute feature dictionaries for all unique drugs and proteins in the dataset."""
    unique_smis = full_df['smi'].unique()
    unique_seqs = full_df['seq'].unique()

    print(f"  Precomputing features for {len(unique_smis)} unique drugs...")
    fp_cache = {smi: smiles_to_morgan_fp(smi) for smi in unique_smis}

    print(f"  Precomputing features for {len(unique_seqs)} unique proteins...")
    aac_cache = {seq: seq_to_aac(seq) for seq in unique_seqs}

    return fp_cache, aac_cache


def get_features(df: pd.DataFrame, fp_cache: dict, aac_cache: dict) -> np.ndarray:
    """Vectorized lookup of precomputed features."""
    fps = np.array([fp_cache[smi] for smi in df['smi'].values], dtype=np.float32)
    aacs = np.array([aac_cache[seq] for seq in df['seq'].values], dtype=np.float32)
    return np.hstack([fps, aacs])


def build_expert(name: str, seed: int):
    if name == 'lr_expert':
        clf = LogisticRegression(max_iter=200, C=1.0, random_state=seed,
                                  class_weight='balanced', solver='lbfgs')
        return Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    elif name == 'rf_expert':
        clf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                      random_state=seed, class_weight='balanced',
                                      n_jobs=-1)
        return Pipeline([('clf', clf)])
    elif name == 'gb_expert':
        clf = HistGradientBoostingClassifier(max_iter=50, max_depth=6,
                                              learning_rate=0.1, random_state=seed,
                                              class_weight='balanced')
        return Pipeline([('clf', clf)])
    else:
        raise ValueError(f"Unknown expert: {name}")


def save_predictions(df: pd.DataFrame, proba: np.ndarray, split_name: str,
                     out_path: str, label_col: str = 'lab'):
    result = pd.DataFrame({
        'original_row_id': df.index,
        'drug_id': df['lig'].values,
        'protein_id': df['pro'].values,
        'label': df[label_col].values,
        'split': split_name,
        'predicted_probability': proba[:, 1],
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Train expert substitutes and generate predictions')
    parser.add_argument('--dataset', '-d', default='davis')
    parser.add_argument('--split-type', '-s', default='random_pair',
                        choices=['random_pair', 'cold_drug', 'cold_protein', 'double_cold'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label-col', default='lab')
    args = parser.parse_args()

    manifest_path = os.path.join('output', args.dataset, args.split_type, 'split_manifest.csv')
    data_path = os.path.join('dataset', args.dataset, 'data.csv')

    if not os.path.exists(manifest_path):
        sys.exit(f"Split manifest not found: {manifest_path}")
    if not os.path.exists(data_path):
        sys.exit(f"Dataset not found: {data_path}")

    print(f"\n{'='*60}")
    print(f"Expert Predict: {args.dataset} / {args.split_type}")
    print(f"{'='*60}")

    full_df = pd.read_csv(data_path)
    manifest = pd.read_csv(manifest_path)

    train_ids = manifest[manifest['split'] == 'train']['original_row_id'].values
    cal_ids = manifest[manifest['split'] == 'calibration']['original_row_id'].values
    test_ids = manifest[manifest['split'] == 'test']['original_row_id'].values

    train_df = full_df.loc[train_ids]
    cal_df = full_df.loc[cal_ids]
    test_df = full_df.loc[test_ids]

    print(f"\nSplit sizes: train={len(train_df)}, cal={len(cal_df)}, test={len(test_df)}")

    fp_cache, aac_cache = build_feature_extractor(full_df)

    print("  Assembling feature matrices...")
    X_train = get_features(train_df, fp_cache, aac_cache)
    y_train = train_df[args.label_col].values.astype(int)

    X_cal = get_features(cal_df, fp_cache, aac_cache)
    X_test = get_features(test_df, fp_cache, aac_cache)

    print(f"  Feature shape: train={X_train.shape}, cal={X_cal.shape}, test={X_test.shape}")

    out_dir = os.path.join('tmp', args.dataset, args.split_type)
    os.makedirs(out_dir, exist_ok=True)

    results_summary = []

    for expert_name in EXPERT_NAMES:
        print(f"\n--- Training {expert_name} ---")
        t0 = time.time()

        model = build_expert(expert_name, args.seed)
        model.fit(X_train, y_train)

        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.1f}s")

        train_proba = model.predict_proba(X_train)
        cal_proba = model.predict_proba(X_cal)
        test_proba = model.predict_proba(X_test)

        save_predictions(train_df, train_proba, 'train',
                         os.path.join(out_dir, f'{expert_name}_predictions_train.csv'), args.label_col)
        save_predictions(cal_df, cal_proba, 'calibration',
                         os.path.join(out_dir, f'{expert_name}_predictions_calibration.csv'), args.label_col)
        save_predictions(test_df, test_proba, 'test',
                         os.path.join(out_dir, f'{expert_name}_predictions_test.csv'), args.label_col)

        joblib.dump(model, os.path.join(out_dir, f'{expert_name}_model.joblib'))

        train_acc = (model.predict(X_train) == y_train).mean()
        print(f"  Train accuracy: {train_acc:.4f}")

        results_summary.append({
            'expert': expert_name,
            'train_accuracy': float(train_acc),
        })

    summary_path = os.path.join(out_dir, 'expert_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nAll expert predictions saved to {out_dir}/")


if __name__ == '__main__':
    main()
