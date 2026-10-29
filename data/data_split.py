"""
data_split.py — Three-way data splitting for EnsDTI.

Implements:
  - split_random_pair  : random interaction-pair splits (no entity constraints)
  - split_cold_drug    : drug-level cold-start splits
  - split_cold_protein : protein-level cold-start splits
  - split_double_cold  : simultaneous drug+protein cold splits

All functions:
  - Use numpy.random.default_rng(seed) for reproducibility
  - Preserve original DataFrame index (row IDs from source data.csv)
  - Return (train_df, calibration_df, test_df)

Also retains original split_dataset() and split_dataset_by_fold() for
backward compatibility with main.py.
"""

import os
import torch
import random
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from torch.utils.data import SubsetRandomSampler, DataLoader
from random import sample


# ===========================================================================
# NEW: Three-way split functions
# ===========================================================================

def split_random_pair(df: pd.DataFrame,
                      test_frac: float = 0.2,
                      calibration_frac: float = 0.1,
                      seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Randomly split interaction pairs into train / calibration / test.

    Each drug–protein pair appears in exactly one split.
    Drugs and proteins MAY appear in multiple splits (no cold-start guarantee).

    Parameters
    ----------
    df               : Full dataset DataFrame with original index preserved.
    test_frac        : Fraction of rows to assign to test.
    calibration_frac : Fraction of rows to assign to calibration
                       (drawn from the non-test portion).
    seed             : Random seed for reproducibility.

    Returns
    -------
    (train_df, calibration_df, test_df)
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = df.index.tolist()
    shuffled = rng.permutation(idx)

    n_test = int(round(n * test_frac))
    n_cal = int(round(n * calibration_frac))

    test_idx = shuffled[:n_test]
    cal_idx = shuffled[n_test: n_test + n_cal]
    train_idx = shuffled[n_test + n_cal:]

    train_df = df.loc[train_idx].copy()
    cal_df = df.loc[cal_idx].copy()
    test_df = df.loc[test_idx].copy()

    _add_split_col(train_df, cal_df, test_df)
    return train_df, cal_df, test_df


def split_cold_drug(df: pd.DataFrame,
                    drug_col: str = 'lig',
                    test_frac: float = 0.2,
                    calibration_frac: float = 0.1,
                    seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cold-drug split: unique drugs are partitioned into train / calibration / test pools.

    Rules:
      - No test drug appears in training or calibration.
      - No calibration drug appears in training.
      - All rows for a given drug belong to exactly one split.

    Parameters
    ----------
    df               : Full dataset DataFrame with original index preserved.
    drug_col         : Column name for drug identity.
    test_frac        : Fraction of unique drugs for test pool.
    calibration_frac : Fraction of unique drugs for calibration pool.
    seed             : Random seed.

    Returns
    -------
    (train_df, calibration_df, test_df)
    """
    rng = np.random.default_rng(seed)
    unique_drugs = np.array(sorted(df[drug_col].unique()))
    n_drugs = len(unique_drugs)
    shuffled = rng.permutation(unique_drugs)

    n_test = max(1, int(round(n_drugs * test_frac)))
    n_cal = max(1, int(round(n_drugs * calibration_frac)))

    test_drugs = set(shuffled[:n_test])
    cal_drugs = set(shuffled[n_test: n_test + n_cal])
    train_drugs = set(shuffled[n_test + n_cal:])

    assert len(test_drugs & cal_drugs) == 0
    assert len(test_drugs & train_drugs) == 0
    assert len(cal_drugs & train_drugs) == 0

    test_df = df[df[drug_col].isin(test_drugs)].copy()
    cal_df = df[df[drug_col].isin(cal_drugs)].copy()
    train_df = df[df[drug_col].isin(train_drugs)].copy()

    _add_split_col(train_df, cal_df, test_df)
    return train_df, cal_df, test_df


def split_cold_protein(df: pd.DataFrame,
                       protein_col: str = 'pro',
                       test_frac: float = 0.2,
                       calibration_frac: float = 0.1,
                       seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cold-protein split: unique proteins are partitioned into train / calibration / test pools.

    Rules:
      - No test protein appears in training or calibration.
      - No calibration protein appears in training.
      - All rows for a given protein belong to exactly one split.

    Parameters
    ----------
    df               : Full dataset DataFrame with original index preserved.
    protein_col      : Column name for protein identity.
    test_frac        : Fraction of unique proteins for test pool.
    calibration_frac : Fraction of unique proteins for calibration pool.
    seed             : Random seed.

    Returns
    -------
    (train_df, calibration_df, test_df)
    """
    rng = np.random.default_rng(seed)
    unique_proteins = np.array(sorted(df[protein_col].unique()))
    n_proteins = len(unique_proteins)
    shuffled = rng.permutation(unique_proteins)

    n_test = max(1, int(round(n_proteins * test_frac)))
    n_cal = max(1, int(round(n_proteins * calibration_frac)))

    test_proteins = set(shuffled[:n_test])
    cal_proteins = set(shuffled[n_test: n_test + n_cal])
    train_proteins = set(shuffled[n_test + n_cal:])

    assert len(test_proteins & cal_proteins) == 0
    assert len(test_proteins & train_proteins) == 0
    assert len(cal_proteins & train_proteins) == 0

    test_df = df[df[protein_col].isin(test_proteins)].copy()
    cal_df = df[df[protein_col].isin(cal_proteins)].copy()
    train_df = df[df[protein_col].isin(train_proteins)].copy()

    _add_split_col(train_df, cal_df, test_df)
    return train_df, cal_df, test_df


def split_double_cold(df: pd.DataFrame,
                      drug_col: str = 'lig',
                      protein_col: str = 'pro',
                      test_frac: float = 0.2,
                      calibration_frac: float = 0.1,
                      seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Double-cold split: both drugs AND proteins are partitioned independently.

    Each pool (train / calibration / test) contains only rows where BOTH
    the drug AND the protein belong to the corresponding pool.
    Rows where drug and protein pools disagree are DISCARDED.

    Rules:
      - Disjoint drug pools for train, calibration, test.
      - Disjoint protein pools for train, calibration, test.
      - Only pairs with drug ∈ pool_X and protein ∈ pool_X are kept per split.
      - Reports retained and discarded pair counts.

    Parameters
    ----------
    df               : Full dataset DataFrame with original index preserved.
    drug_col         : Column name for drug identity.
    protein_col      : Column name for protein identity.
    test_frac        : Fraction of unique drugs/proteins for test pool.
    calibration_frac : Fraction of unique drugs/proteins for calibration pool.
    seed             : Random seed.

    Returns
    -------
    (train_df, calibration_df, test_df)
    """
    rng = np.random.default_rng(seed)

    # Split drugs
    unique_drugs = np.array(sorted(df[drug_col].unique()))
    n_drugs = len(unique_drugs)
    shuffled_drugs = rng.permutation(unique_drugs)
    n_dtest = max(1, int(round(n_drugs * test_frac)))
    n_dcal = max(1, int(round(n_drugs * calibration_frac)))
    test_drugs = set(shuffled_drugs[:n_dtest])
    cal_drugs = set(shuffled_drugs[n_dtest: n_dtest + n_dcal])
    train_drugs = set(shuffled_drugs[n_dtest + n_dcal:])

    # Split proteins (use different sub-seed to avoid correlation with drug split)
    rng2 = np.random.default_rng(seed + 1)
    unique_proteins = np.array(sorted(df[protein_col].unique()))
    n_proteins = len(unique_proteins)
    shuffled_proteins = rng2.permutation(unique_proteins)
    n_ptest = max(1, int(round(n_proteins * test_frac)))
    n_pcal = max(1, int(round(n_proteins * calibration_frac)))
    test_proteins = set(shuffled_proteins[:n_ptest])
    cal_proteins = set(shuffled_proteins[n_ptest: n_ptest + n_pcal])
    train_proteins = set(shuffled_proteins[n_ptest + n_pcal:])

    # Verify disjointness
    for name_d, d_pool in [('test', test_drugs), ('cal', cal_drugs), ('train', train_drugs)]:
        for name_p, p_pool in [('test', test_proteins), ('cal', cal_proteins), ('train', train_proteins)]:
            pass  # pools are for different axes, no cross-axis requirement

    assert len(test_drugs & cal_drugs) == 0
    assert len(test_drugs & train_drugs) == 0
    assert len(cal_drugs & train_drugs) == 0
    assert len(test_proteins & cal_proteins) == 0
    assert len(test_proteins & train_proteins) == 0
    assert len(cal_proteins & train_proteins) == 0

    # Assign split: pair is kept only if both drug and protein belong to same pool
    def pool_label(row):
        d_in_train = row[drug_col] in train_drugs
        d_in_cal = row[drug_col] in cal_drugs
        d_in_test = row[drug_col] in test_drugs
        p_in_train = row[protein_col] in train_proteins
        p_in_cal = row[protein_col] in cal_proteins
        p_in_test = row[protein_col] in test_proteins

        if d_in_train and p_in_train:
            return 'train'
        elif d_in_cal and p_in_cal:
            return 'calibration'
        elif d_in_test and p_in_test:
            return 'test'
        else:
            return 'discard'

    df = df.copy()
    df['_pool'] = df.apply(pool_label, axis=1)

    n_total = len(df)
    n_discard = (df['_pool'] == 'discard').sum()
    print(f"[double_cold] Total pairs: {n_total}, Retained: {n_total - n_discard}, "
          f"Discarded: {n_discard} ({100*n_discard/n_total:.1f}%)")

    train_df = df[df['_pool'] == 'train'].drop(columns=['_pool']).copy()
    cal_df = df[df['_pool'] == 'calibration'].drop(columns=['_pool']).copy()
    test_df = df[df['_pool'] == 'test'].drop(columns=['_pool']).copy()

    _add_split_col(train_df, cal_df, test_df)
    return train_df, cal_df, test_df


def _add_split_col(train_df, cal_df, test_df):
    """Add a 'split' column to each DataFrame."""
    train_df['split'] = 'train'
    cal_df['split'] = 'calibration'
    test_df['split'] = 'test'


def save_split_manifest(train_df: pd.DataFrame,
                        cal_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        dataset: str,
                        split_type: str,
                        drug_col: str = 'lig',
                        protein_col: str = 'pro',
                        label_col: str = 'lab') -> str:
    """
    Save a split manifest CSV to output/<dataset>/<split_type>/split_manifest.csv.

    Columns: original_row_id, drug_id, protein_id, label, split
    """
    out_dir = os.path.join('output', dataset, split_type)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, 'split_manifest.csv')

    parts = []
    for df in [train_df, cal_df, test_df]:
        part = pd.DataFrame({
            'original_row_id': df.index,
            'drug_id': df[drug_col].values,
            'protein_id': df[protein_col].values,
            'label': df[label_col].values,
            'split': df['split'].values,
        })
        parts.append(part)

    manifest = pd.concat(parts, ignore_index=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"[manifest] Saved to {manifest_path}  "
          f"(train={len(train_df)}, cal={len(cal_df)}, test={len(test_df)})")
    return manifest_path


# ===========================================================================
# LEGACY: Preserved for backward compatibility with main.py
# ===========================================================================

def split_dataset(datasets: Dict[str, torch.utils.data.Dataset],
                  model_names: List[str],
                  exp_mode: str,
                  split_type: str = 'random') -> Tuple[Dict, Dict]:
    """
    Original random row-level 90/10 train/val split.
    Preserved for backward compatibility with main.py.
    """
    num = len(datasets[model_names[0]])
    indices = list(range(num))
    random.shuffle(indices)

    train_idx = indices[:int(0.9 * num)]
    valid_idx = indices[int(0.9 * num):]

    train_datasets = {m: datasets[m][torch.tensor(train_idx)] for m in model_names}
    valid_datasets = {m: datasets[m][torch.tensor(valid_idx)] for m in model_names}

    print(f"[->] Split dataset with train:val = {len(train_idx)}:{len(valid_idx)}")
    return train_datasets, valid_datasets


def split_dataset_by_fold(dataset, indices, fold, batch_size) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Original k-fold split by interaction row index.
    Preserved for backward compatibility with main.py.
    """
    fold_size = int(0.2 * len(indices))
    test_idx = indices[fold * fold_size:] if fold == 4 else indices[fold * fold_size:(fold + 1) * fold_size]
    train_idx = list(set(indices) - set(test_idx))
    valid_idx = sample(train_idx, int(0.1 * len(train_idx)))

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx))
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))

    return train_loader, valid_loader, test_loader
