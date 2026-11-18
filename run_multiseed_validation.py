"""
run_multiseed_validation.py:
Executes:
Phase 1: Multi-Seed Validation of 4 Gating Variants across seeds 42, 123, 999.
- Original unregularized MLP gating
- Regularized MLP gating (L2 alpha=1.0)
- Adaptive fallback gating (consensus average on shift)
- Novelty-aware MLP gating (conditioned on novelty flags)
For each seed and gating variant, runs MoE gating and ICP (Regime C) across all 4 regimes:
random_pair, cold_drug, cold_protein, double_cold.
Outputs:
- tmp/davis/gating_multiseed/<gating_variant>/seed_<seed>/<split_type>_results.csv
- tmp/davis/gating_multiseed_summary.csv
- tmp/davis/gating_multiseed_mean_std.csv
- output/davis/gating_multiseed_per_seed.png
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
TMP_DIR = os.path.join(CUR_DIR, 'tmp', 'davis')
OUT_DIR = os.path.join(CUR_DIR, 'output', 'davis')
SPLITS = ['random_pair', 'cold_drug', 'cold_protein', 'double_cold']
SEEDS = [42, 123, 999]
EXPERTS = ['dpdta', 'dcdti', 'dacmf']
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


def compute_icp_conformal(p_cal, y_cal, p_test):
    """Compute ICP p-values, predicted label, and confidence using calibration nonconformity."""
    y_cal = np.asarray(y_cal, dtype=int)
    cal_scores = np.abs(p_cal - y_cal)
    n_cal = len(cal_scores)

    test_scores_0 = np.abs(p_test - 0.0)
    test_scores_1 = np.abs(p_test - 1.0)

    cal_sorted = np.sort(cal_scores)
    idx_0 = np.searchsorted(cal_sorted, test_scores_0, side='left')
    p0 = (n_cal - idx_0 + 1.0) / (n_cal + 1.0)

    idx_1 = np.searchsorted(cal_sorted, test_scores_1, side='left')
    p1 = (n_cal - idx_1 + 1.0) / (n_cal + 1.0)

    conf = np.maximum(p0, p1)
    pred_label = (p1 > p0).astype(int)

    return pred_label, conf, p0, p1


def load_seed_split_data(split, seed):
    """Load train, calibration, and test predictions for a given seed and split."""
    seed_dir = os.path.join(TMP_DIR, split, f'seed_{seed}')
    dfs_tr = [pd.read_csv(os.path.join(seed_dir, f'{e}_predictions_train.csv')).set_index('original_row_id') for e in EXPERTS]
    dfs_cal = [pd.read_csv(os.path.join(seed_dir, f'{e}_predictions_calibration.csv')).set_index('original_row_id') for e in EXPERTS]
    dfs_te = [pd.read_csv(os.path.join(seed_dir, f'{e}_predictions_test.csv')).set_index('original_row_id') for e in EXPERTS]

    X_tr = np.column_stack([d['predicted_probability'].values for d in dfs_tr])
    y_tr = dfs_tr[0]['label'].values.astype(int)

    X_cal = np.column_stack([d['predicted_probability'].values for d in dfs_cal])
    y_cal = dfs_cal[0]['label'].values.astype(int)

    X_te = np.column_stack([d['predicted_probability'].values for d in dfs_te])
    y_te = dfs_te[0]['label'].values.astype(int)

    meta_tr = dfs_tr[0][['drug_id', 'protein_id']].copy()
    meta_cal = dfs_cal[0][['drug_id', 'protein_id']].copy()
    meta_te = dfs_te[0][['drug_id', 'protein_id']].copy()

    return (X_tr, y_tr, meta_tr), (X_cal, y_cal, meta_cal), (X_te, y_te, meta_te)


def main():
    print("=" * 80)
    print(" PHASE 1: MULTI-SEED VALIDATION OF GATING VARIANTS (SEEDS 42, 123, 999)")
    print("=" * 80)

    summary_records = []

    for variant in VARIANTS:
        print(f"\n---> Evaluating Gating Variant: {variant}")
        for seed in SEEDS:
            var_seed_dir = os.path.join(TMP_DIR, 'gating_multiseed', variant, f'seed_{seed}')
            os.makedirs(var_seed_dir, exist_ok=True)

            for split in SPLITS:
                (X_tr, y_tr, meta_tr), (X_cal, y_cal, meta_cal), (X_te, y_te, meta_te) = load_seed_split_data(split, seed)

                train_drugs = set(meta_tr['drug_id'].values)
                train_prots = set(meta_tr['protein_id'].values)

                is_d_cal = (~meta_cal['drug_id'].isin(train_drugs)).astype(float).values
                is_p_cal = (~meta_cal['protein_id'].isin(train_prots)).astype(float).values
                is_inter_cal = is_d_cal * is_p_cal

                is_d_te = (~meta_te['drug_id'].isin(train_drugs)).astype(float).values
                is_p_te = (~meta_te['protein_id'].isin(train_prots)).astype(float).values
                is_inter_te = is_d_te * is_p_te

                is_shift_cal = (is_d_cal > 0) | (is_p_cal > 0)
                is_shift_te = (is_d_te > 0) | (is_p_te > 0)

                # Fit gating model
                if variant == 'original_mlp':
                    clf = MLPClassifier(
                        hidden_layer_sizes=(15,), activation='logistic', solver='adam',
                        learning_rate_init=0.01, max_iter=35, random_state=seed, early_stopping=True
                    )
                    clf.fit(X_tr, y_tr)
                    p_cal = clf.predict_proba(X_cal)[:, 1]
                    p_te = clf.predict_proba(X_te)[:, 1]

                elif variant == 'regularized_mlp':
                    clf = MLPClassifier(
                        hidden_layer_sizes=(10,), activation='logistic', alpha=1.0,
                        max_iter=100, random_state=seed
                    )
                    clf.fit(X_tr, y_tr)
                    p_cal = clf.predict_proba(X_cal)[:, 1]
                    p_te = clf.predict_proba(X_te)[:, 1]

                elif variant == 'adaptive_fallback':
                    clf = MLPClassifier(
                        hidden_layer_sizes=(15,), activation='logistic', solver='adam',
                        learning_rate_init=0.01, max_iter=35, random_state=seed, early_stopping=True
                    )
                    clf.fit(X_tr, y_tr)
                    p_mlp_cal = clf.predict_proba(X_cal)[:, 1]
                    p_avg_cal = np.mean(X_cal, axis=1)
                    p_cal = np.where(is_shift_cal, p_avg_cal, p_mlp_cal)

                    p_mlp_te = clf.predict_proba(X_te)[:, 1]
                    p_avg_te = np.mean(X_te, axis=1)
                    p_te = np.where(is_shift_te, p_avg_te, p_mlp_te)

                elif variant == 'novelty_aware':
                    np.random.seed(seed)
                    tr_drugs_list = list(train_drugs)
                    tr_prots_list = list(train_prots)
                    sim_d = set(np.random.choice(tr_drugs_list, size=int(0.25 * len(tr_drugs_list)), replace=False))
                    sim_p = set(np.random.choice(tr_prots_list, size=int(0.25 * len(tr_prots_list)), replace=False))

                    is_d_tr = meta_tr['drug_id'].isin(sim_d).astype(float).values
                    is_p_tr = meta_tr['protein_id'].isin(sim_p).astype(float).values
                    is_inter_tr = is_d_tr * is_p_tr
                    X_tr_nov = np.column_stack([X_tr, is_d_tr, is_p_tr, is_inter_tr])

                    X_cal_nov = np.column_stack([X_cal, is_d_cal, is_p_cal, is_inter_cal])
                    X_te_nov = np.column_stack([X_te, is_d_te, is_p_te, is_inter_te])

                    clf = MLPClassifier(
                        hidden_layer_sizes=(10,), activation='logistic', alpha=1.0,
                        max_iter=100, random_state=seed
                    )
                    clf.fit(X_tr_nov, y_tr)
                    p_cal = clf.predict_proba(X_cal_nov)[:, 1]
                    p_te = clf.predict_proba(X_te_nov)[:, 1]

                # Run Conformal ICP (Regime C: calibrated on split's own calibration set)
                pred_label, conf, p0, p1 = compute_icp_conformal(p_cal, y_cal, p_te)

                # Save per-seed split results
                res_df = pd.DataFrame({
                    'original_row_id': meta_te.index.values,
                    'drug_id': meta_te['drug_id'].values,
                    'protein_id': meta_te['protein_id'].values,
                    'true_label': y_te,
                    'predicted_probability': p_te,
                    'predicted_label': pred_label,
                    'confidence': conf,
                    'p_value_0': p0,
                    'p_value_1': p1
                })
                res_path = os.path.join(var_seed_dir, f'{split}_results.csv')
                res_df.to_csv(res_path, index=False)

                # Compute performance metrics
                m = compute_metrics(y_te, p_te)
                # Selective accuracy computed from ICP confidence
                n_ret = max(1, int(round(len(y_te) * 0.5)))
                order = np.argsort(-conf)
                sel_acc_50 = float(accuracy_score(y_te[order[:n_ret]], pred_label[order[:n_ret]]))

                summary_records.append({
                    'gating_variant': variant,
                    'seed': seed,
                    'split_type': split,
                    'auroc': m['auroc'],
                    'auprc': m['auprc'],
                    'accuracy': m['accuracy'],
                    'aurc': m['aurc'],
                    'selective_acc_50': sel_acc_50
                })

    # Save summary DataFrame
    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(TMP_DIR, 'gating_multiseed_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[SAVED] Multi-seed summary (48 rows) -> {summary_path}")

    # Compute mean ± std across seeds
    mean_std_records = []
    for variant in VARIANTS:
        for split in SPLITS:
            sub = df_summary[(df_summary['gating_variant'] == variant) & (df_summary['split_type'] == split)]
            mean_std_records.append({
                'gating_variant': variant,
                'split_type': split,
                'auroc_mean': sub['auroc'].mean(),
                'auroc_std': sub['auroc'].std(),
                'auprc_mean': sub['auprc'].mean(),
                'auprc_std': sub['auprc'].std(),
                'accuracy_mean': sub['accuracy'].mean(),
                'accuracy_std': sub['accuracy'].std(),
                'aurc_mean': sub['aurc'].mean(),
                'aurc_std': sub['aurc'].std(),
                'selective_acc_50_mean': sub['selective_acc_50'].mean(),
                'selective_acc_50_std': sub['selective_acc_50'].std()
            })

    df_mean_std = pd.DataFrame(mean_std_records)
    mean_std_path = os.path.join(TMP_DIR, 'gating_multiseed_mean_std.csv')
    df_mean_std.to_csv(mean_std_path, index=False)
    print(f"[SAVED] Multi-seed mean ± std table (16 rows) -> {mean_std_path}")

    # Generate multi-seed comparison plot (cold_drug and double_cold per-seed AUROC)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    variant_labels = {
        'original_mlp': 'Original MLP',
        'regularized_mlp': 'Regularized MLP (α=1.0)',
        'adaptive_fallback': 'Adaptive Fallback',
        'novelty_aware': 'Novelty-Aware MLP'
    }
    variant_colors = {
        'original_mlp': '#e74c3c',
        'regularized_mlp': '#3498db',
        'adaptive_fallback': '#f39c12',
        'novelty_aware': '#2ecc71'
    }

    target_splits = [('cold_drug', axes[0], 'cold_drug (Eliminating Seed 999 Inversion)'),
                     ('double_cold', axes[1], 'double_cold (Reversing Regression)')]

    for split_name, ax, title in target_splits:
        x = np.arange(len(SEEDS))
        width = 0.20
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(VARIANTS))

        for idx, variant in enumerate(VARIANTS):
            vals = [
                df_summary[(df_summary['gating_variant'] == variant) &
                           (df_summary['split_type'] == split_name) &
                           (df_summary['seed'] == s)]['auroc'].values[0]
                for s in SEEDS
            ]
            ax.bar(x + offsets[idx], vals, width, label=variant_labels[variant],
                   color=variant_colors[variant], alpha=0.9, edgecolor='black', linewidth=0.8)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Random Seed', fontsize=11, fontweight='bold')
        ax.set_ylabel('Test AUROC', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=11)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Random Guess (0.50)' if split_name == 'cold_drug' else None)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        if split_name == 'cold_drug':
            ax.set_ylim(0.15, 0.85)
            ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
        else:
            ax.set_ylim(0.35, 0.65)
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.suptitle("Multi-Seed Gating Stability Comparison (Seeds 42, 123, 999)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'gating_multiseed_per_seed.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Multi-seed comparison plot -> {plot_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(" MULTI-SEED AUROC SUMMARY (MEAN ± STD)")
    print("=" * 80)
    pivot_mean = df_mean_std.pivot(index='gating_variant', columns='split_type', values='auroc_mean')[SPLITS]
    pivot_std = df_mean_std.pivot(index='gating_variant', columns='split_type', values='auroc_std')[SPLITS]
    
    formatted_rows = []
    for v in VARIANTS:
        row = {'gating_variant': v}
        for sp in SPLITS:
            m_val = pivot_mean.loc[v, sp]
            s_val = pivot_std.loc[v, sp]
            row[sp] = f"{m_val:.4f} ± {s_val:.4f}"
        formatted_rows.append(row)
    print(pd.DataFrame(formatted_rows).to_string(index=False))


if __name__ == '__main__':
    main()
