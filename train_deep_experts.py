"""
train_deep_experts.py:
Standardized training and evaluation harness for EnsDTI's native PyTorch expert models:
- DeepDTA (models/dpdta.py)
- DeepConv-DTI (models/dcdti.py)
- DeepPurpose (models/dp.py)
- GIFDTI (models/gifdti.py)
- MDeePred (models/mdprd.py)

Strict Constraints Adhered To:
1. Accepts our leakage-controlled split manifests (original_row_id, drug_id, protein_id, label, split).
2. Trains ONLY on the train split.
3. Generates out-of-sample predictions for calibration and test splits.
4. Outputs tmp/<dataset>/<split_type>/<expert>_predictions.csv with schema:
   original_row_id, drug_id, protein_id, label, split, predicted_probability
5. Reproducible random seed fixed throughout.
"""

import os
import sys
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem

# Insert EnsDTI root into python path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from data.dataset import CHARCANSMISET, CHARPROTSET
from models.dpdta import DeepDTA
from models.dcdti import DeepConvDTI
from models.gifdti import CNNFormerDTI, CHARISOSMISET, CHARPROTSET as GIF_CHARPROTSET

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def encode_smiles(smi: str, max_len: int = 85) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    can_smi = Chem.MolToSmiles(mol) if mol is not None else smi
    enc = [CHARCANSMISET.get(ch, 0) for ch in can_smi[:max_len]]
    if len(enc) < max_len:
        enc += [0] * (max_len - len(enc))
    return np.array(enc, dtype=np.int64)

def encode_protein(seq: str, max_len: int = 1200) -> np.ndarray:
    enc = [CHARPROTSET.get(ch, 0) for ch in seq[:max_len]]
    if len(enc) < max_len:
        enc += [0] * (max_len - len(enc))
    return np.array(enc, dtype=np.int64)

def get_morgan_fp(smi: str, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits, useChirality=True)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, drug_feat_dict: dict, prot_feat_dict: dict):
        self.df = df.reset_index(drop=True)
        self.drug_feats = drug_feat_dict
        self.prot_feats = prot_feat_dict
        self.drug_ids = self.df['drug_id'].astype(str).values
        self.prot_ids = self.df['protein_id'].astype(str).values
        self.labels = self.df['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d_id = self.drug_ids[idx]
        p_id = self.prot_ids[idx]
        d_feat = self.drug_feats[d_id]
        p_feat = self.prot_feats[p_id]
        lab = self.labels[idx]
        return d_feat, p_feat, lab


def train_and_eval_deepdta(manifest_path: str, dataset_raw_path: str, output_csv: str, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3):
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DeepDTA] Reading manifest from {manifest_path}...")
    manifest = pd.read_csv(manifest_path)
    raw_df = pd.read_csv(dataset_raw_path)

    # Build sequence lookup dicts
    lig_smi = dict(zip(raw_df['lig'].astype(str), raw_df['smi']))
    pro_seq = dict(zip(raw_df['pro'].astype(str), raw_df['seq']))

    # Pre-encode unique drugs & proteins
    unique_drugs = manifest['drug_id'].astype(str).unique()
    unique_prots = manifest['protein_id'].astype(str).unique()
    print(f"[DeepDTA] Pre-encoding {len(unique_drugs)} drugs and {len(unique_prots)} proteins...")
    d_feats = {d: encode_smiles(lig_smi[d], max_len=85) for d in unique_drugs}
    p_feats = {p: encode_protein(pro_seq[p], max_len=1200) for p in unique_prots}

    train_df = manifest[manifest['split'] == 'train'].copy()
    calib_df = manifest[manifest['split'] == 'calibration'].copy()
    test_df = manifest[manifest['split'] == 'test'].copy()

    train_ds = ManifestDataset(train_df, d_feats, p_feats)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print(f"[DeepDTA] Building model on device: {device}...")
    model = DeepDTA(85, 1200, 1, 'classification').to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[DeepDTA] Starting training for {epochs} epochs on train set ({len(train_df)} samples)...")
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for d_batch, p_batch, y_batch in train_loader:
            d_batch = d_batch.to(device)
            p_batch = p_batch.to(device)
            y_batch = y_batch.unsqueeze(1).to(device)

            optimizer.zero_grad()
            out = model(d_batch, p_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            n_batches += 1
        elapsed = time.time() - t0
        print(f"  [Epoch {ep+1}/{epochs}] Loss: {ep_loss/n_batches:.4f} | Time: {elapsed:.2f}s")

    # Generate predictions for all rows (calibration, test, and train)
    model.eval()
    all_ds = ManifestDataset(manifest, d_feats, p_feats)
    all_loader = DataLoader(all_ds, batch_size=batch_size, shuffle=False)

    preds = []
    print("[DeepDTA] Generating predictions across entire manifest...")
    with torch.no_grad():
        for d_batch, p_batch, _ in all_loader:
            d_batch = d_batch.to(device)
            p_batch = p_batch.to(device)
            logits = model(d_batch, p_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(probs)

    out_df = pd.DataFrame({
        'original_row_id': manifest['original_row_id'],
        'drug_id': manifest['drug_id'],
        'protein_id': manifest['protein_id'],
        'label': manifest['label'],
        'split': manifest['split'],
        'predicted_probability': np.round(preds, 6)
    })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[DeepDTA] Predictions successfully saved to {output_csv} (Total: {len(out_df)} rows)")

    # Print quick validation metrics on test split
    from sklearn.metrics import roc_auc_score, average_precision_score
    test_sub = out_df[out_df['split'] == 'test']
    if len(test_sub['label'].unique()) > 1:
        test_auc = roc_auc_score(test_sub['label'], test_sub['predicted_probability'])
        test_prc = average_precision_score(test_sub['label'], test_sub['predicted_probability'])
        print(f"[DeepDTA Test Metrics] AUROC: {test_auc:.4f} | AUPRC: {test_prc:.4f}")


def train_and_eval_dcdti(manifest_path: str, dataset_raw_path: str, output_csv: str, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3):
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DeepConv-DTI] Reading manifest from {manifest_path}...")
    manifest = pd.read_csv(manifest_path)
    raw_df = pd.read_csv(dataset_raw_path)

    lig_smi = dict(zip(raw_df['lig'].astype(str), raw_df['smi']))
    pro_seq = dict(zip(raw_df['pro'].astype(str), raw_df['seq']))

    unique_drugs = manifest['drug_id'].astype(str).unique()
    unique_prots = manifest['protein_id'].astype(str).unique()
    print(f"[DeepConv-DTI] Pre-encoding {len(unique_drugs)} drugs (Morgan 2048) and {len(unique_prots)} proteins (Seq 2500)...")
    d_feats = {d: get_morgan_fp(lig_smi[d], n_bits=2048) for d in unique_drugs}
    p_feats = {p: encode_protein(pro_seq[p], max_len=2500) for p in unique_prots}

    train_df = manifest[manifest['split'] == 'train'].copy()
    train_ds = ManifestDataset(train_df, d_feats, p_feats)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print(f"[DeepConv-DTI] Building model on device: {device}...")
    model = DeepConvDTI(1, 'classification').to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[DeepConv-DTI] Starting training for {epochs} epochs on train set ({len(train_df)} samples)...")
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for d_batch, p_batch, y_batch in train_loader:
            d_batch = d_batch.float().to(device)
            p_batch = p_batch.long().to(device)
            y_batch = y_batch.unsqueeze(1).to(device)

            optimizer.zero_grad()
            out = model(d_batch, p_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            n_batches += 1
        elapsed = time.time() - t0
        print(f"  [Epoch {ep+1}/{epochs}] Loss: {ep_loss/n_batches:.4f} | Time: {elapsed:.2f}s")

    model.eval()
    all_ds = ManifestDataset(manifest, d_feats, p_feats)
    all_loader = DataLoader(all_ds, batch_size=batch_size, shuffle=False)

    preds = []
    print("[DeepConv-DTI] Generating predictions across entire manifest...")
    with torch.no_grad():
        for d_batch, p_batch, _ in all_loader:
            d_batch = d_batch.float().to(device)
            p_batch = p_batch.long().to(device)
            logits = model(d_batch, p_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(probs)

    out_df = pd.DataFrame({
        'original_row_id': manifest['original_row_id'],
        'drug_id': manifest['drug_id'],
        'protein_id': manifest['protein_id'],
        'label': manifest['label'],
        'split': manifest['split'],
        'predicted_probability': np.round(preds, 6)
    })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[DeepConv-DTI] Predictions successfully saved to {output_csv} (Total: {len(out_df)} rows)")

    from sklearn.metrics import roc_auc_score, average_precision_score
    test_sub = out_df[out_df['split'] == 'test']
    if len(test_sub['label'].unique()) > 1:
        test_auc = roc_auc_score(test_sub['label'], test_sub['predicted_probability'])
        test_prc = average_precision_score(test_sub['label'], test_sub['predicted_probability'])
        print(f"[DeepConv-DTI Test Metrics] AUROC: {test_auc:.4f} | AUPRC: {test_prc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['dpdta', 'dcdti'], help='Expert model name')
    parser.add_argument('--manifest', type=str, required=True, help='Path to split manifest CSV')
    parser.add_argument('--dataset-raw', type=str, default='dataset/davis/data.csv', help='Path to raw data.csv')
    parser.add_argument('--output', type=str, required=True, help='Path to output predictions CSV')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    args = parser.parse_args()
    if args.model == 'dpdta':
        train_and_eval_deepdta(args.manifest, args.dataset_raw, args.output, args.epochs, args.batch_size, args.lr)
    elif args.model == 'dcdti':
        train_and_eval_dcdti(args.manifest, args.dataset_raw, args.output, args.epochs, args.batch_size, args.lr)
