"""
train_dacmf_colab.py:
GPU-accelerated training and prediction generation for DACMF-DTI (AttentionDTA)
across all four EnsDTI leakage-controlled evaluation regimes:
- random_pair
- cold_drug
- cold_protein
- double_cold

Designed for execution on Google Colab (Tesla T4 GPU) or any CUDA-enabled environment.

Strict Constraints Adhered To:
1. Model architecture (AttentionDTA, ScaledDotProductAttention, Swish) is completely unmodified.
2. Accepts leakage-controlled split manifests (original_row_id, drug_id, protein_id, label, split).
3. Trains strictly on the 'train' split.
4. Generates honest out-of-sample predictions for 'calibration' and 'test' splits.
5. Saves predictions matching EnsDTI schema:
   original_row_id, drug_id, protein_id, label, split, predicted_probability
"""

import os
import sys
import argparse
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from models.dacmf import AttentionDTA

SEED = 42

CHARISOSMISET_DACMF = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64
}

CHARPROTSET_DACMF = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
}

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def encode_smiles_dacmf(smi: str, max_len: int = 100) -> np.ndarray:
    enc = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(smi[:max_len]):
        enc[i] = CHARISOSMISET_DACMF.get(ch, 0)
    return enc

def encode_protein_dacmf(seq: str, max_len: int = 1200) -> np.ndarray:
    enc = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(seq[:max_len]):
        enc[i] = CHARPROTSET_DACMF.get(ch, 0)
    return enc

class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, drug_feat_dict: dict, prot_feat_dict: dict):
        self.df = df.reset_index(drop=True)
        self.drug_feats = drug_feat_dict
        self.prot_feats = prot_feat_dict
        self.drug_ids = self.df['drug_id'].astype(str).values
        self.prot_ids = self.df['protein_id'].astype(str).values
        self.labels = self.df['label'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d_id = self.drug_ids[idx]
        p_id = self.prot_ids[idx]
        d_feat = self.drug_feats[d_id]
        p_feat = self.prot_feats[p_id]
        lab = self.labels[idx]
        return d_feat, p_feat, lab

def train_and_eval_dacmf(split_type: str, dataset: str = "davis",
                         base_dir: str = ".", epochs: int = 5,
                         batch_size: int = 64, lr: float = 1e-4):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(f" [DACMF-DTI] Training Regime: {split_type} on {device}")
    print("=" * 70)

    manifest_path = os.path.join(base_dir, "output", dataset, split_type, "split_manifest.csv")
    dataset_raw_path = os.path.join(base_dir, "dataset", dataset, "data.csv")
    output_dir = os.path.join(base_dir, "tmp", dataset, split_type)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "dacmf_predictions.csv")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not os.path.exists(dataset_raw_path):
        raise FileNotFoundError(f"Raw data not found: {dataset_raw_path}")

    manifest = pd.read_csv(manifest_path)
    raw_df = pd.read_csv(dataset_raw_path)

    lig_smi = dict(zip(raw_df['lig'].astype(str), raw_df['smi']))
    pro_seq = dict(zip(raw_df['pro'].astype(str), raw_df['seq']))

    unique_drugs = manifest['drug_id'].astype(str).unique()
    unique_prots = manifest['protein_id'].astype(str).unique()
    print(f"  Pre-encoding {len(unique_drugs)} drugs and {len(unique_prots)} proteins...")
    d_feats = {d: encode_smiles_dacmf(lig_smi[d], max_len=100) for d in unique_drugs}
    p_feats = {p: encode_protein_dacmf(pro_seq[p], max_len=1200) for p in unique_prots}

    train_df = manifest[manifest['split'] == 'train'].copy()
    calib_df = manifest[manifest['split'] == 'calibration'].copy()
    test_df = manifest[manifest['split'] == 'test'].copy()
    print(f"  Partition counts: Train={len(train_df)}, Calibration={len(calib_df)}, Test={len(test_df)}")

    train_ds = ManifestDataset(train_df, d_feats, p_feats)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2 if torch.cuda.is_available() else 0,
                              pin_memory=torch.cuda.is_available())

    model = AttentionDTA(head_num=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"  Starting training for {epochs} epochs...")
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for d_batch, p_batch, y_batch in train_loader:
            d_batch = d_batch.long().to(device)
            p_batch = p_batch.long().to(device)
            y_batch = y_batch.long().to(device)

            optimizer.zero_grad()
            out = model(d_batch, p_batch, None, None, None, None, None, None)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            n_batches += 1
        elapsed = time.time() - t0
        print(f"    [Epoch {ep+1}/{epochs}] Train Loss: {ep_loss/n_batches:.4f} | Time: {elapsed:.2f}s")

    # Generate out-of-sample predictions
    model.eval()
    all_ds = ManifestDataset(manifest, d_feats, p_feats)
    all_loader = DataLoader(all_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2 if torch.cuda.is_available() else 0,
                            pin_memory=torch.cuda.is_available())

    preds = []
    print("  Generating predictions across all splits...")
    with torch.no_grad():
        for d_batch, p_batch, _ in all_loader:
            d_batch = d_batch.long().to(device)
            p_batch = p_batch.long().to(device)
            logits = model(d_batch, p_batch, None, None, None, None, None, None)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().flatten()
            preds.extend(probs)

    out_df = pd.DataFrame({
        'original_row_id': manifest['original_row_id'],
        'drug_id': manifest['drug_id'],
        'protein_id': manifest['protein_id'],
        'label': manifest['label'],
        'split': manifest['split'],
        'predicted_probability': np.round(preds, 6)
    })

    out_df.to_csv(output_csv, index=False)
    print(f"  [SAVED] {output_csv} ({len(out_df)} rows)")

    # Save per-split CSVs required by EnsDTI MoE
    for s_name in ['train', 'calibration', 'test']:
        sub = out_df[out_df['split'] == s_name]
        sub_path = os.path.join(output_dir, f'dacmf_predictions_{s_name}.csv')
        sub.to_csv(sub_path, index=False)
    print(f"  [SAVED] Split-specific CSVs (train, calibration, test) -> {output_dir}")

    # Compute validation test metrics
    test_sub = out_df[out_df['split'] == 'test']
    y_true = test_sub['label'].values
    y_prob = test_sub['predicted_probability'].values
    y_pred = (y_prob >= 0.5).astype(int)

    test_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    test_prc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    test_acc = accuracy_score(y_true, y_pred)
    print(f"  [TEST METRICS] Split: {split_type} | AUROC: {test_auc:.4f} | AUPRC: {test_prc:.4f} | Acc: {test_acc*100:.2f}%\n")
    return {
        'split': split_type,
        'auroc': test_auc,
        'auprc': test_prc,
        'accuracy': test_acc
    }

def main():
    parser = argparse.ArgumentParser(description="Train DACMF-DTI on Google Colab / GPU")
    parser.add_argument("--split", type=str, default="all",
                        choices=["random_pair", "cold_drug", "cold_protein", "double_cold", "all"],
                        help="Split regime to run")
    parser.add_argument("--dataset", type=str, default="davis")
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    splits = ["random_pair", "cold_drug", "cold_protein", "double_cold"] if args.split == "all" else [args.split]
    results = []
    t_start = time.time()
    for s in splits:
        res = train_and_eval_dacmf(s, args.dataset, args.base_dir, args.epochs, args.batch_size, args.lr)
        results.append(res)
    total_elapsed = time.time() - t_start

    print("=" * 70)
    print(f" DACMF-DTI TRAINING COMPLETE ({total_elapsed/60:.2f} minutes total)")
    print("=" * 70)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
