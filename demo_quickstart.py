"""
demo_quickstart.py:
Interactive demo to test EnsDTI on a small sample subset of Davis.
Shows:
1. Individual Expert Model Performance (DeepDTA & DeepConv-DTI)
2. MoE Gating Combination
3. ICP Conformal Confidence & Selective Error Filtering
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.neural_network import MLPClassifier

# Ensure EnsDTI is in path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.dpdta import DeepDTA
from models.dcdti import DeepConvDTI
from train_deep_experts import encode_smiles, encode_protein, get_morgan_fp, ManifestDataset

def main():
    parser = argparse.ArgumentParser(description="Quickstart Demo for EnsDTI on Davis sample")
    parser.add_argument("--samples", type=int, default=500, help="Number of train samples to use for fast demo")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--split", type=str, default="cold_drug", choices=["random_pair", "cold_drug", "cold_protein", "double_cold"])
    args = parser.parse_args()

    print("=" * 70)
    print(f" EnsDTI QUICKSTART DEMO (Split: {args.split}, Train Subsample: {args.samples})")
    print("=" * 70)

    # 1. Load Manifest and Raw Data
    manifest_path = os.path.join(CUR_DIR, "output", "davis", args.split, "split_manifest.csv")
    raw_path = os.path.join(CUR_DIR, "dataset", "davis", "data.csv")

    if not os.path.exists(manifest_path) or not os.path.exists(raw_path):
        sys.exit(f"Data files not found! Checked: {manifest_path} and {raw_path}")

    manifest = pd.read_csv(manifest_path)
    raw_df = pd.read_csv(raw_path)

    lig_smi = dict(zip(raw_df["lig"].astype(str), raw_df["smi"]))
    pro_seq = dict(zip(raw_df["pro"].astype(str), raw_df["seq"]))

    # Subsample training data for fast demonstration
    train_full = manifest[manifest["split"] == "train"]
    train_sub = train_full.sample(n=min(args.samples, len(train_full)), random_state=42)
    calib_sub = manifest[manifest["split"] == "calibration"].sample(n=min(200, len(manifest[manifest["split"] == "calibration"])), random_state=42)
    test_sub = manifest[manifest["split"] == "test"].sample(n=min(300, len(manifest[manifest["split"] == "test"])), random_state=42)

    demo_df = pd.concat([train_sub, calib_sub, test_sub]).reset_index(drop=True)
    print(f"Data partitioned: Train={len(train_sub)}, Calibration={len(calib_sub)}, Test={len(test_sub)}")

    # 2. Pre-encode Features
    unique_drugs = demo_df["drug_id"].astype(str).unique()
    unique_prots = demo_df["protein_id"].astype(str).unique()
    print(f"Pre-encoding {len(unique_drugs)} unique drugs and {len(unique_prots)} unique proteins...")

    d_dta = {d: encode_smiles(lig_smi[d], max_len=85) for d in unique_drugs}
    p_dta = {p: encode_protein(pro_seq[p], max_len=1200) for p in unique_prots}
    d_conv = {d: get_morgan_fp(lig_smi[d], n_bits=2048) for d in unique_drugs}
    p_conv = {p: encode_protein(pro_seq[p], max_len=2500) for p in unique_prots}

    # 3. Train & Evaluate DeepDTA
    print("\n" + "-" * 50)
    print(" [STAGE 1A] Training Expert Model: DeepDTA")
    print("-" * 50)
    m_dta = DeepDTA(85, 1200, 1, "classification")
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(m_dta.parameters(), lr=1e-3)

    train_ds_dta = ManifestDataset(train_sub, d_dta, p_dta)
    train_loader = DataLoader(train_ds_dta, batch_size=32, shuffle=True)

    m_dta.train()
    for ep in range(args.epochs):
        losses = []
        for d_b, p_b, y_b in train_loader:
            opt.zero_grad()
            out = m_dta(d_b, p_b)
            loss = crit(out, y_b.unsqueeze(1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"  DeepDTA Epoch {ep+1}/{args.epochs} - Loss: {np.mean(losses):.4f}")

    m_dta.eval()
    with torch.no_grad():
        all_ds = ManifestDataset(demo_df, d_dta, p_dta)
        d_all = torch.tensor(np.stack([s[0] for s in all_ds])).long()
        p_all = torch.tensor(np.stack([s[1] for s in all_ds])).long()
        dta_preds = torch.sigmoid(m_dta(d_all, p_all)).numpy().flatten()
    demo_df["prob_dpdta"] = dta_preds

    # 4. Train & Evaluate DeepConv-DTI
    print("\n" + "-" * 50)
    print(" [STAGE 1B] Training Expert Model: DeepConv-DTI")
    print("-" * 50)
    m_conv = DeepConvDTI(1, "classification")
    opt_c = torch.optim.Adam(m_conv.parameters(), lr=1e-3)

    train_ds_conv = ManifestDataset(train_sub, d_conv, p_conv)
    train_loader_c = DataLoader(train_ds_conv, batch_size=32, shuffle=True)

    m_conv.train()
    for ep in range(args.epochs):
        losses = []
        for d_b, p_b, y_b in train_loader_c:
            opt_c.zero_grad()
            out = m_conv(d_b.float(), p_b.long())
            loss = crit(out, y_b.unsqueeze(1))
            loss.backward()
            opt_c.step()
            losses.append(loss.item())
        print(f"  DeepConv-DTI Epoch {ep+1}/{args.epochs} - Loss: {np.mean(losses):.4f}")

    m_conv.eval()
    with torch.no_grad():
        all_ds_c = ManifestDataset(demo_df, d_conv, p_conv)
        conv_preds = torch.sigmoid(m_conv(torch.tensor(np.stack([s[0] for s in all_ds_c])).float(), torch.tensor(np.stack([s[1] for s in all_ds_c])).long())).numpy().flatten()
    demo_df["prob_dcdti"] = conv_preds

    # 5. Evaluate Individual Model Performance on Test Split
    test_eval = demo_df[demo_df["split"] == "test"]
    y_test = test_eval["label"].values

    auc_dta = roc_auc_score(y_test, test_eval["prob_dpdta"]) if len(np.unique(y_test)) > 1 else float("nan")
    prc_dta = average_precision_score(y_test, test_eval["prob_dpdta"]) if len(np.unique(y_test)) > 1 else float("nan")
    acc_dta = accuracy_score(y_test, (test_eval["prob_dpdta"] >= 0.5).astype(int))

    auc_conv = roc_auc_score(y_test, test_eval["prob_dcdti"]) if len(np.unique(y_test)) > 1 else float("nan")
    prc_conv = average_precision_score(y_test, test_eval["prob_dcdti"]) if len(np.unique(y_test)) > 1 else float("nan")
    acc_conv = accuracy_score(y_test, (test_eval["prob_dcdti"] >= 0.5).astype(int))

    print("\n" + "=" * 50)
    print(" INDIVIDUAL EXPERT PERFORMANCES (Test Split)")
    print("=" * 50)
    print(f"  1. DeepDTA      | AUROC: {auc_dta:.4f} | AUPRC: {prc_dta:.4f} | Accuracy: {acc_dta*100:.2f}%")
    print(f"  2. DeepConv-DTI | AUROC: {auc_conv:.4f} | AUPRC: {prc_conv:.4f} | Accuracy: {acc_conv*100:.2f}%")

    # 6. Stage 2: Mixture-of-Experts Gating
    print("\n" + "-" * 50)
    print(" [STAGE 2] Mixture-of-Experts Gating Network")
    print("-" * 50)
    train_eval = demo_df[demo_df["split"] == "train"]
    X_train_moe = train_eval[["prob_dpdta", "prob_dcdti"]].values
    y_train_moe = train_eval["label"].values

    moe = MLPClassifier(hidden_layer_sizes=(10,), max_iter=30, random_state=42)
    moe.fit(X_train_moe, y_train_moe)

    X_all_moe = demo_df[["prob_dpdta", "prob_dcdti"]].values
    demo_df["moe_prob"] = moe.predict_proba(X_all_moe)[:, 1]

    test_moe = demo_df[demo_df["split"] == "test"]
    auc_moe = roc_auc_score(y_test, test_moe["moe_prob"]) if len(np.unique(y_test)) > 1 else float("nan")
    prc_moe = average_precision_score(y_test, test_moe["moe_prob"]) if len(np.unique(y_test)) > 1 else float("nan")
    acc_moe = accuracy_score(y_test, (test_moe["moe_prob"] >= 0.5).astype(int))

    print(f"  MoE Combined    | AUROC: {auc_moe:.4f} | AUPRC: {prc_moe:.4f} | Accuracy: {acc_moe*100:.2f}%")

    # 7. Stage 3: Conformal Prediction Calibration & Selective Classification
    print("\n" + "-" * 50)
    print(" [STAGE 3] Inductive Conformal Prediction (ICP) & Confidence Scoring")
    print("-" * 50)
    cal_df = demo_df[demo_df["split"] == "calibration"]
    # Nonconformity: alpha = |prob - true_label|
    cal_nonconf = np.abs(cal_df["moe_prob"].values - cal_df["label"].values)

    test_probs = test_moe["moe_prob"].values
    confidences = []
    for p in test_probs:
        # p-values for y=0 and y=1
        alpha_0 = np.abs(p - 0.0)
        alpha_1 = np.abs(p - 1.0)
        pval_0 = (np.sum(cal_nonconf >= alpha_0) + 1.0) / (len(cal_nonconf) + 1.0)
        pval_1 = (np.sum(cal_nonconf >= alpha_1) + 1.0) / (len(cal_nonconf) + 1.0)
        conf = max(pval_0, pval_1)
        confidences.append(conf)

    test_eval = demo_df[demo_df["split"] == "test"].copy()
    test_eval["confidence"] = confidences
    test_eval["pred_label"] = (test_eval["moe_prob"] >= 0.5).astype(int)

    print(f"  Average Confidence on Test Set: {np.mean(confidences):.4f}")
    print("\n  Selective Prediction Performance (Rejecting Low-Confidence Samples):")
    print("  " + "-" * 45)
    print("  Retention Rate | Retained Count | Selective Accuracy")
    print("  " + "-" * 45)
    for q in [1.0, 0.8, 0.6, 0.5]:
        thresh = test_eval["confidence"].quantile(1.0 - q)
        retained = test_eval[test_eval["confidence"] >= thresh]
        acc_q = accuracy_score(retained["label"], retained["pred_label"])
        print(f"      {q*100:5.1f}%     |      {len(retained):4d}      |      {acc_q*100:.2f}%")

    print("\n[SUCCESS] Demo completed cleanly! All three stages verified on sample data.")

if __name__ == "__main__":
    main()
