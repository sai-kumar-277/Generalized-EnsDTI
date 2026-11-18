"""
run_bindingdb_pipeline.py:
Executes Phase 2: BindingDB Integration, Expert Training, MoE Gating, and Conformal Evaluation.
- Trains 3 multi-expert models: lr_expert, rf_expert, gb_expert
- Benchmarks 4 Gating Variants:
  1. original_mlp
  2. regularized_mlp (alpha=1.0)
  3. adaptive_fallback
  4. novelty_aware
- Evaluates Inductive Conformal Prediction (ICP) under Regime A vs Regime C
Across all 4 regimes: random_pair, cold_drug, cold_protein, double_cold.
Outputs:
- tmp/bindingdb/gating_comparison.csv
- tmp/bindingdb/cold_aware_icp_results.csv
- output/bindingdb/bindingdb_gating_comparison.png
"""

import os
import sys
import time
import tempfile
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.neural_network import MLPClassifier

os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(CUR_DIR, 'tmp', 'bindingdb')
OUT_DIR = os.path.join(CUR_DIR, 'output', 'bindingdb')
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SPLITS = ['random_pair', 'cold_drug', 'cold_protein', 'double_cold']
EXPERTS = ['lr_expert', 'rf_expert', 'gb_expert']
VARIANTS = ['original_mlp', 'regularized_mlp', 'adaptive_fallback', 'novelty_aware']

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
DRUG_FP_BITS = 1024
_mfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=DRUG_FP_BITS)


def smiles_to_morgan_fp(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return np.zeros(DRUG_FP_BITS, dtype=np.float32)
        fp = _mfp_gen.GetFingerprintAsNumPy(mol)
        return fp.astype(np.float32)
    except Exception:
        return np.zeros(DRUG_FP_BITS, dtype=np.float32)


def seq_to_aac(seq):
    seq = str(seq).upper()
    n = max(1, len(seq))
    counts = np.array([seq.count(aa) for aa in AMINO_ACIDS], dtype=np.float32)
    return counts / n


def get_feature_matrix(df, fp_cache, aac_cache):
    smiles = df['compound_iso_smiles'].values
    seqs = df['target_sequence'].values
    n = len(df)
    fp_mat = np.empty((n, DRUG_FP_BITS), dtype=np.float32)
    aac_mat = np.empty((n, 20), dtype=np.float32)
    for i in range(n):
        fp_mat[i] = fp_cache[smiles[i]]
        aac_mat[i] = aac_cache[seqs[i]]
    return np.hstack([fp_mat, aac_mat])


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan')
    auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan')
    acc = float(accuracy_score(y_true, y_pred))

    conf = np.abs(y_prob - 0.5) * 2.0
    n = len(y_true)
    n_ret = max(1, int(round(n * 0.5)))
    order = np.argsort(-conf)
    sel_acc_50 = float(accuracy_score(y_true[order[:n_ret]], y_pred[order[:n_ret]]))

    thresholds = np.linspace(0, 1, 50)
    covs, risks = [], []
    for thr in thresholds:
        mask = (conf >= thr)
        if np.sum(mask) == 0:
            continue
        covs.append(np.sum(mask) / n)
        risks.append(float(np.mean(y_pred[mask] != y_true[mask])))
    if len(covs) > 1:
        sort_ord = np.argsort(covs)
        aurc = float(np.trapezoid(np.array(risks)[sort_ord], np.array(covs)[sort_ord]))
    else:
        aurc = 0.0

    if np.isnan(aurc):
        aurc = 0.0

    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': acc,
        'aurc': aurc,
        'selective_acc_50': sel_acc_50,
        'confidence': conf
    }


def build_expert(name, seed=42):
    if name == 'lr_expert':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=300, C=0.1, random_state=seed, solver='lbfgs'))
        ])
    elif name == 'rf_expert':
        return RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=1, random_state=seed)
    elif name == 'gb_expert':
        return HistGradientBoostingClassifier(max_iter=100, max_depth=10, random_state=seed)
    else:
        raise ValueError(f"Unknown expert {name}")


def run_bindingdb_all():
    print("=" * 80)
    print("PHASE 2: BINDINGDB EXPERT TRAINING AND GATING BENCHMARK")
    print("=" * 80)

    # 1. Precompute feature caches across entire BindingDB
    full_df = pd.read_csv(os.path.join(CUR_DIR, 'dataset', 'bindingdb', 'data.csv'))
    print(f"Precomputing features for {full_df['compound_iso_smiles'].nunique()} drugs and {full_df['target_sequence'].nunique()} targets...")

    fp_cache = {}
    for smi in full_df['compound_iso_smiles'].unique():
        fp_cache[smi] = smiles_to_morgan_fp(smi)

    aac_cache = {}
    for seq in full_df['target_sequence'].unique():
        aac_cache[seq] = seq_to_aac(seq)

    records = []
    icp_records = []

    for split in SPLITS:
        print(f"\n==================== Split: {split} ====================")
        split_dir = os.path.join(TMP_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

        manifest = pd.read_csv(os.path.join(OUT_DIR, split, 'split_manifest.csv'))
        tr_df = manifest[manifest['split'] == 'train'].copy()
        cal_df = manifest[manifest['split'] == 'calibration'].copy()
        te_df = manifest[manifest['split'] == 'test'].copy()

        y_tr = tr_df['label'].values.astype(int)
        y_cal = cal_df['label'].values.astype(int)
        y_te = te_df['label'].values.astype(int)

        X_tr = get_feature_matrix(tr_df, fp_cache, aac_cache)
        X_cal = get_feature_matrix(cal_df, fp_cache, aac_cache)
        X_te = get_feature_matrix(te_df, fp_cache, aac_cache)

        print(f"  Matrix shapes: train={X_tr.shape}, cal={X_cal.shape}, test={X_te.shape}")

        P_cal = np.empty((len(y_cal), len(EXPERTS)), dtype=np.float32)
        P_te = np.empty((len(y_te), len(EXPERTS)), dtype=np.float32)

        # Train experts
        for i, exp in enumerate(EXPERTS):
            t0 = time.time()
            model = build_expert(exp, seed=42)
            model.fit(X_tr, y_tr)
            elapsed = time.time() - t0

            p_cal_exp = model.predict_proba(X_cal)[:, 1]
            p_te_exp = model.predict_proba(X_te)[:, 1]

            P_cal[:, i] = p_cal_exp
            P_te[:, i] = p_te_exp

            m = compute_metrics(y_te, p_te_exp)
            records.append({
                'dataset': 'bindingdb',
                'split_type': split,
                'model_type': 'single_expert',
                'name': exp,
                'auroc': m['auroc'],
                'auprc': m['auprc'],
                'accuracy': m['accuracy'],
                'aurc': m['aurc'],
                'selective_acc_50': m['selective_acc_50']
            })
            print(f"  Expert {exp:12s} ({elapsed:.1f}s): AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}, Acc={m['accuracy']:.4f}")

        # Compute novelty features
        train_drugs = set(tr_df['drug_id'].values)
        train_prots = set(tr_df['protein_id'].values)

        is_d_cal = (~cal_df['drug_id'].isin(train_drugs)).astype(float).values
        is_p_cal = (~cal_df['protein_id'].isin(train_prots)).astype(float).values
        is_inter_cal = is_d_cal * is_p_cal
        nov_cal = np.column_stack([is_d_cal, is_p_cal, is_inter_cal])

        is_d_te = (~te_df['drug_id'].isin(train_drugs)).astype(float).values
        is_p_te = (~te_df['protein_id'].isin(train_prots)).astype(float).values
        is_inter_te = is_d_te * is_p_te
        nov_te = np.column_stack([is_d_te, is_p_te, is_inter_te])

        # Evaluate 4 Gating Variants
        for var in VARIANTS:
            if var == 'original_mlp':
                gate = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=42, alpha=1e-4)
                gate.fit(P_cal, y_cal)
                p_gate_te = gate.predict_proba(P_te)[:, 1]
                p_gate_cal = gate.predict_proba(P_cal)[:, 1]

            elif var == 'regularized_mlp':
                gate = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=42, alpha=1.0)
                gate.fit(P_cal, y_cal)
                p_gate_te = gate.predict_proba(P_te)[:, 1]
                p_gate_cal = gate.predict_proba(P_cal)[:, 1]

            elif var == 'adaptive_fallback':
                if split == 'random_pair':
                    gate = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=42, alpha=1e-4)
                    gate.fit(P_cal, y_cal)
                    p_gate_te = gate.predict_proba(P_te)[:, 1]
                    p_gate_cal = gate.predict_proba(P_cal)[:, 1]
                else:
                    p_gate_te = np.mean(P_te, axis=1)
                    p_gate_cal = np.mean(P_cal, axis=1)

            elif var == 'novelty_aware':
                X_cal_nov = np.hstack([P_cal, nov_cal])
                X_te_nov = np.hstack([P_te, nov_te])
                gate = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=42, alpha=1.0)
                gate.fit(X_cal_nov, y_cal)
                p_gate_te = gate.predict_proba(X_te_nov)[:, 1]
                p_gate_cal = gate.predict_proba(X_cal_nov)[:, 1]

            m = compute_metrics(y_te, p_gate_te)
            records.append({
                'dataset': 'bindingdb',
                'split_type': split,
                'model_type': 'gating_variant',
                'name': var,
                'auroc': m['auroc'],
                'auprc': m['auprc'],
                'accuracy': m['accuracy'],
                'aurc': m['aurc'],
                'selective_acc_50': m['selective_acc_50']
            })
            print(f"  Gating {var:18s}: AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}, Acc={m['accuracy']:.4f}, SelAcc50={m['selective_acc_50']:.4f}")

            # Conformal evaluation for Novelty-Aware Gating
            if var == 'novelty_aware':
                for alpha in [0.05, 0.10, 0.20]:
                    cal_scores = np.abs(p_gate_cal - y_cal)
                    q_val = np.quantile(cal_scores, 1.0 - alpha, method='higher')
                    s0 = np.abs(p_gate_te - 0.0)
                    s1 = np.abs(p_gate_te - 1.0)
                    in0 = (s0 <= q_val)
                    in1 = (s1 <= q_val)
                    cov = np.where(y_te == 1, in1, in0).mean()
                    avg_size = (in0.astype(int) + in1.astype(int)).mean()
                    icp_records.append({
                        'dataset': 'bindingdb',
                        'split_type': split,
                        'gating_variant': var,
                        'alpha': alpha,
                        'nominal_coverage': 1.0 - alpha,
                        'empirical_coverage': float(cov),
                        'avg_set_size': float(avg_size),
                        'coverage_gap': float(cov - (1.0 - alpha))
                    })

    # Save results
    df_res = pd.DataFrame(records)
    res_path = os.path.join(TMP_DIR, 'gating_comparison.csv')
    df_res.to_csv(res_path, index=False)
    print(f"\nSaved BindingDB gating comparison to {res_path}")

    df_icp = pd.DataFrame(icp_records)
    icp_path = os.path.join(TMP_DIR, 'cold_aware_icp_results.csv')
    df_icp.to_csv(icp_path, index=False)
    print(f"Saved BindingDB ICP results to {icp_path}")

    # Create visualization
    create_bindingdb_plot(df_res)


def create_bindingdb_plot(df):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    splits = ['cold_drug', 'double_cold']
    titles = [
        '(A) BindingDB - cold_drug Performance Across Gating Architectures',
        '(B) BindingDB - double_cold Performance Across Gating Architectures'
    ]

    for idx, split in enumerate(splits):
        ax = axes[idx]
        sub = df[(df['split_type'] == split) & (df['model_type'] == 'gating_variant')]

        bars = ax.bar(sub['name'], sub['auroc'], color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'], alpha=0.85)
        ax.set_title(titles[idx], fontsize=12, fontweight='bold')
        ax.set_ylabel('Test AUROC', fontsize=11, fontweight='bold')
        ax.set_ylim([max(0.4, sub['auroc'].min() - 0.05), min(1.0, sub['auroc'].max() + 0.05)])
        ax.grid(True, linestyle='--', alpha=0.5)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'bindingdb_gating_comparison.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved BindingDB plot to {plot_path}")


if __name__ == '__main__':
    run_bindingdb_all()
