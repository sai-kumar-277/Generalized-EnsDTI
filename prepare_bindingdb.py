"""
prepare_bindingdb.py:
Prepares BindingDB dataset from experts/GIFDTI/data/BindingDB/BindingDB.txt
and creates strict leakage-controlled 3-way split manifests:
- output/bindingdb/random_pair/split_manifest.csv
- output/bindingdb/cold_drug/split_manifest.csv
- output/bindingdb/cold_protein/split_manifest.csv
- output/bindingdb/double_cold/split_manifest.csv
"""

import os
import hashlib
import numpy as np
import pandas as pd

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_TXT = os.path.join(CUR_DIR, '..', 'experts', 'GIFDTI', 'data', 'BindingDB', 'BindingDB.txt')
DATA_DIR = os.path.join(CUR_DIR, 'dataset', 'bindingdb')
OUT_DIR = os.path.join(CUR_DIR, 'output', 'bindingdb')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def get_id(val, prefix):
    h = hashlib.md5(str(val).encode('utf-8')).hexdigest()[:8]
    return f"{prefix}_{h}"


def load_raw_bindingdb():
    print(f"Reading raw BindingDB from {SRC_TXT}...")
    records = []
    with open(SRC_TXT, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            smi = parts[0]
            seq = parts[1]
            lbl = int(parts[2])
            d_id = get_id(smi, 'BDB_D')
            p_id = get_id(seq, 'BDB_P')
            records.append({
                'original_row_id': idx,
                'drug_id': d_id,
                'protein_id': p_id,
                'compound_iso_smiles': smi,
                'target_sequence': seq,
                'label': lbl
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records across {df['drug_id'].nunique()} unique drugs and {df['protein_id'].nunique()} unique targets.")
    csv_path = os.path.join(DATA_DIR, 'data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved standardized data.csv to {csv_path}")
    return df


def create_splits(df, seed=42):
    np.random.seed(seed)
    n = len(df)

    # 1. Random Pair Split (70/10/20)
    print("\n--- Generating random_pair split ---")
    df_rp = df.copy()
    indices = np.random.permutation(n)
    n_tr = int(0.70 * n)
    n_cal = int(0.10 * n)

    split_col = np.empty(n, dtype=object)
    split_col[indices[:n_tr]] = 'train'
    split_col[indices[n_tr:n_tr + n_cal]] = 'calibration'
    split_col[indices[n_tr + n_cal:]] = 'test'
    df_rp['split'] = split_col

    save_manifest(df_rp, 'random_pair')

    # 2. Cold Drug Split (disjoint drugs 70/10/20)
    print("\n--- Generating cold_drug split ---")
    df_cd = df.copy()
    unique_drugs = np.array(list(df['drug_id'].unique()))
    np.random.shuffle(unique_drugs)
    n_d = len(unique_drugs)
    nd_tr = int(0.70 * n_d)
    nd_cal = int(0.10 * n_d)

    tr_drugs = set(unique_drugs[:nd_tr])
    cal_drugs = set(unique_drugs[nd_tr:nd_tr + nd_cal])
    te_drugs = set(unique_drugs[nd_tr + nd_cal:])

    split_col = np.empty(n, dtype=object)
    for i, d in enumerate(df['drug_id'].values):
        if d in tr_drugs:
            split_col[i] = 'train'
        elif d in cal_drugs:
            split_col[i] = 'calibration'
        else:
            split_col[i] = 'test'
    df_cd['split'] = split_col
    save_manifest(df_cd, 'cold_drug')

    # 3. Cold Protein Split (disjoint targets 70/10/20)
    print("\n--- Generating cold_protein split ---")
    df_cp = df.copy()
    unique_prots = np.array(list(df['protein_id'].unique()))
    np.random.shuffle(unique_prots)
    n_p = len(unique_prots)
    np_tr = int(0.70 * n_p)
    np_cal = int(0.10 * n_p)

    tr_prots = set(unique_prots[:np_tr])
    cal_prots = set(unique_prots[np_tr:np_tr + np_cal])
    te_prots = set(unique_prots[np_tr + np_cal:])

    split_col = np.empty(n, dtype=object)
    for i, p in enumerate(df['protein_id'].values):
        if p in tr_prots:
            split_col[i] = 'train'
        elif p in cal_prots:
            split_col[i] = 'calibration'
        else:
            split_col[i] = 'test'
    df_cp['split'] = split_col
    save_manifest(df_cp, 'cold_protein')

    # 4. Double Cold Split (strictly disjoint in both drugs AND targets)
    print("\n--- Generating double_cold split ---")
    df_dc = df.copy()
    split_col = np.full(n, 'unassigned', dtype=object)

    for i in range(n):
        d = df['drug_id'].iloc[i]
        p = df['protein_id'].iloc[i]
        if d in tr_drugs and p in tr_prots:
            split_col[i] = 'train'
        elif d in cal_drugs and p in cal_prots:
            split_col[i] = 'calibration'
        elif d in te_drugs and p in te_prots:
            split_col[i] = 'test'

    df_dc['split'] = split_col
    df_dc = df_dc[df_dc['split'] != 'unassigned'].copy()
    print(f"Double cold assigned: total={len(df_dc)}, train={(df_dc['split']=='train').sum()}, cal={(df_dc['split']=='calibration').sum()}, test={(df_dc['split']=='test').sum()}")
    save_manifest(df_dc, 'double_cold')


def save_manifest(df, split_type):
    out_dir = os.path.join(OUT_DIR, split_type)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, 'split_manifest.csv')
    df.to_csv(manifest_path, index=False)
    for s in ['train', 'calibration', 'test']:
        sub = df[df['split'] == s].copy()
        sub.to_csv(os.path.join(out_dir, f'{s}.csv'), index=False)
    print(f"Saved {split_type} manifests to {out_dir}/ (train={(df['split']=='train').sum()}, cal={(df['split']=='calibration').sum()}, test={(df['split']=='test').sum()})")


def main():
    df = load_raw_bindingdb()
    create_splits(df, seed=42)
    print("\nAll BindingDB manifests generated successfully.")


if __name__ == '__main__':
    main()
