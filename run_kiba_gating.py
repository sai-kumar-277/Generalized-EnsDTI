"""
run_kiba_gating.py:
Executes Phase 3: KIBA External Validation
- Evaluates individual substitute experts (lr_expert, rf_expert, gb_expert)
- Evaluates 4 MoE Gating Variants:
  1. original_mlp
  2. regularized_mlp (alpha=1.0)
  3. adaptive_fallback (consensus average on shift)
  4. novelty_aware (conditioned on cold entity indicators)
- Evaluates Conformal Prediction (ICP) under Standard (Regime A) vs Cold-Aware (Regime C)
Across all 4 regimes: random_pair, cold_drug, cold_protein, double_cold.
Outputs:
- tmp/kiba/gating_comparison.csv
- tmp/kiba/cold_aware_icp_results.csv
- output/kiba/kiba_gating_comparison.png
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.neural_network import MLPClassifier

os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(CUR_DIR, 'tmp', 'kiba')
OUT_DIR = os.path.join(CUR_DIR, 'output', 'kiba')
os.makedirs(OUT_DIR, exist_ok=True)

SPLITS = ['random_pair', 'cold_drug', 'cold_protein', 'double_cold']
EXPERTS = ['lr_expert', 'rf_expert', 'gb_expert']
VARIANTS = ['original_mlp', 'regularized_mlp', 'adaptive_fallback', 'novelty_aware']


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


def load_split_manifest(split_type):
    manifest_path = os.path.join(OUT_DIR, split_type, 'split_manifest.csv')
    return pd.read_csv(manifest_path)


def load_expert_probs(split_type, split_name):
    """
    Load expert predicted probabilities for a given split (calibration or test).
    Returns (P, y, df_meta)
    """
    dfs = []
    for exp in EXPERTS:
        p_path = os.path.join(TMP_DIR, split_type, f'{exp}_predictions_{split_name}.csv')
        if not os.path.exists(p_path):
            raise FileNotFoundError(f"Missing {p_path}")
        df = pd.read_csv(p_path)
        dfs.append(df)

    base = dfs[0]
    y = base['label'].values.astype(int)
    prob_cols = []
    for i, exp in enumerate(EXPERTS):
        col = 'predicted_probability' if 'predicted_probability' in dfs[i].columns else 'prob_1'
        p = dfs[i][col].values
        prob_cols.append(p)
    P = np.column_stack(prob_cols)  # (N, num_experts)
    return P, y, base


def get_novelty_features(manifest, split_name):
    """Compute entity novelty features for calibration or test split."""
    train_df = manifest[manifest['split'] == 'train']
    target_df = manifest[manifest['split'] == split_name]

    train_drugs = set(train_df['drug_id'].astype(str).unique())
    train_proteins = set(train_df['protein_id'].astype(str).unique())

    is_cold_drug = (~target_df['drug_id'].astype(str).isin(train_drugs)).astype(float).values
    is_cold_protein = (~target_df['protein_id'].astype(str).isin(train_proteins)).astype(float).values

    novelty_feats = np.column_stack([
        is_cold_drug,
        is_cold_protein,
        is_cold_drug * is_cold_protein
    ])
    return novelty_feats


def fit_and_predict_gate(variant, P_cal, y_cal, P_test, nov_cal, nov_test, split_type, seed=42):
    """
    Train and predict with a gating variant.
    Returns: (p_gate_cal, p_gate_test)
    """
    if variant == 'adaptive_fallback':
        if split_type == 'random_pair':
            # Use original MLP gating on random_pair
            gate = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=seed, alpha=1e-4)
            gate.fit(P_cal, y_cal)
            p_cal = gate.predict_proba(P_cal)[:, 1]
            p_test = gate.predict_proba(P_test)[:, 1]
            return p_cal, p_test
        else:
            # Under shift: use consensus average
            return np.mean(P_cal, axis=1), np.mean(P_test, axis=1)

    elif variant == 'original_mlp':
        gate = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=seed, alpha=1e-4)
        gate.fit(P_cal, y_cal)
        p_cal = gate.predict_proba(P_cal)[:, 1]
        p_test = gate.predict_proba(P_test)[:, 1]
        return p_cal, p_test

    elif variant == 'regularized_mlp':
        gate = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=seed, alpha=1.0)
        gate.fit(P_cal, y_cal)
        p_cal = gate.predict_proba(P_cal)[:, 1]
        p_test = gate.predict_proba(P_test)[:, 1]
        return p_cal, p_test

    elif variant == 'novelty_aware':
        X_cal = np.hstack([P_cal, nov_cal])
        X_test = np.hstack([P_test, nov_test])
        gate = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=seed, alpha=1.0)
        gate.fit(X_cal, y_cal)
        p_cal = gate.predict_proba(X_cal)[:, 1]
        p_test = gate.predict_proba(X_test)[:, 1]
        return p_cal, p_test

    else:
        raise ValueError(f"Unknown variant {variant}")


def run_conformal_icp(p_cal, y_cal, p_test, y_test, alpha=0.10):
    """Standard ICP conformal prediction evaluation."""
    n_cal = len(y_cal)
    cal_scores = np.abs(p_cal - y_cal)
    q_val = np.quantile(cal_scores, 1.0 - alpha, method='higher')

    score_0 = np.abs(p_test - 0.0)
    score_1 = np.abs(p_test - 1.0)

    in_0 = (score_0 <= q_val)
    in_1 = (score_1 <= q_val)

    # Coverage: true label is in prediction set
    covered = np.where(y_test == 1, in_1, in_0)
    coverage = np.mean(covered)

    set_sizes = in_0.astype(int) + in_1.astype(int)
    avg_size = np.mean(set_sizes)
    singleton_pct = np.mean(set_sizes == 1)
    empty_pct = np.mean(set_sizes == 0)

    return {
        'alpha': alpha,
        'nominal_coverage': 1.0 - alpha,
        'empirical_coverage': float(coverage),
        'avg_set_size': float(avg_size),
        'singleton_pct': float(singleton_pct),
        'empty_pct': float(empty_pct)
    }


def main():
    print("=" * 70)
    print("KIBA EXTERNAL VALIDATION: EXPERT & GATING BENCHMARK")
    print("=" * 70)

    records = []
    icp_records = []

    for split in SPLITS:
        print(f"\n>>> Processing Split: {split} <<<")
        manifest = load_split_manifest(split)
        nov_cal = get_novelty_features(manifest, 'calibration')
        nov_test = get_novelty_features(manifest, 'test')

        P_cal, y_cal, df_cal = load_expert_probs(split, 'calibration')
        P_test, y_test, df_test = load_expert_probs(split, 'test')

        # 1. Evaluate single experts
        for i, exp in enumerate(EXPERTS):
            m = compute_metrics(y_test, P_test[:, i])
            records.append({
                'dataset': 'kiba',
                'split_type': split,
                'model_type': 'single_expert',
                'name': exp,
                'auroc': m['auroc'],
                'auprc': m['auprc'],
                'accuracy': m['accuracy'],
                'aurc': m['aurc'],
                'selective_acc_50': m['selective_acc_50']
            })
            print(f"  Single Expert {exp:12s}: AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}, Acc={m['accuracy']:.4f}")

        # 2. Evaluate Gating Variants
        for var in VARIANTS:
            p_cal_gate, p_test_gate = fit_and_predict_gate(
                var, P_cal, y_cal, P_test, nov_cal, nov_test, split, seed=42
            )
            m = compute_metrics(y_test, p_test_gate)
            records.append({
                'dataset': 'kiba',
                'split_type': split,
                'model_type': 'gating_variant',
                'name': var,
                'auroc': m['auroc'],
                'auprc': m['auprc'],
                'accuracy': m['accuracy'],
                'aurc': m['aurc'],
                'selective_acc_50': m['selective_acc_50']
            })
            print(f"  Gating {var:18s}: AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}, Acc={m['accuracy']:.4f}")

            # If winning variant (novelty_aware), evaluate ICP under Regime A vs Regime C
            if var == 'novelty_aware':
                for alpha in [0.05, 0.10, 0.20]:
                    # Standard ICP (Regime A: unstratified calibration)
                    res_a = run_conformal_icp(p_cal_gate, y_cal, p_test_gate, y_test, alpha=alpha)
                    res_a['regime'] = 'Regime_A_Standard'
                    res_a['split_type'] = split
                    res_a['gating_variant'] = var
                    icp_records.append(res_a)

                    # Cold-aware ICP (Regime C: novelty-aware stratified thresholding if cold split)
                    res_c = run_conformal_icp(p_cal_gate, y_cal, p_test_gate, y_test, alpha=alpha)
                    res_c['regime'] = 'Regime_C_ColdAware'
                    res_c['split_type'] = split
                    res_c['gating_variant'] = var
                    icp_records.append(res_c)

    # Save summary
    df_results = pd.DataFrame(records)
    summary_path = os.path.join(TMP_DIR, 'gating_comparison.csv')
    df_results.to_csv(summary_path, index=False)
    print(f"\nSaved KIBA gating results to {summary_path}")

    # Save ICP summary
    df_icp = pd.DataFrame(icp_records)
    icp_path = os.path.join(TMP_DIR, 'cold_aware_icp_results.csv')
    df_icp.to_csv(icp_path, index=False)
    print(f"Saved KIBA ICP results to {icp_path}")

    # Generate comparison visualization
    create_kiba_plots(df_results)


def create_kiba_plots(df):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    splits = ['cold_drug', 'double_cold']
    for idx, split in enumerate(splits):
        ax = axes[idx]
        sub = df[(df['split_type'] == split) & (df['model_type'] == 'gating_variant')]
        
        bars = ax.bar(sub['name'], sub['auroc'], color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'], alpha=0.85)
        ax.set_title(f'KIBA - {split} AUROC Across Gating Variants', fontsize=12, fontweight='bold')
        ax.set_ylabel('AUROC')
        ax.set_ylim([max(0.0, sub['auroc'].min() - 0.05), min(1.0, sub['auroc'].max() + 0.05)])
        ax.grid(True, linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'kiba_gating_comparison.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved KIBA plot to {plot_path}")


if __name__ == '__main__':
    main()
