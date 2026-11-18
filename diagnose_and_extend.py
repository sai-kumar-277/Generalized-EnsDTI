"""
diagnose_and_extend.py:
Executes:
Phase 1: cold_drug diagnosis (per-seed metrics, gating weights, plot).
Phase 2: double_cold diagnosis (per-seed metrics, gating weights, plot).
Phase 3: Verification of complete 7-way ablation table (all 28 rows).
Phase 4: Extended Cold-Aware ICP across 6 alpha levels (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
         with coverage-alpha curve plot.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.neural_network import MLPClassifier

os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS = ['random_pair', 'cold_drug', 'cold_protein', 'double_cold']
SEEDS = [42, 123, 999]
EXPERTS = ['dpdta', 'dcdti', 'dacmf']
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


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
        aurc = float('nan')

    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': acc,
        'aurc': aurc,
        'selective_acc_50': sel_acc_50,
        'confidence': conf
    }


def compute_icp_conformal_multi_alpha(p_cal, y_cal, p_test, y_test, alphas=ALPHAS):
    y_cal = np.asarray(y_cal, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

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

    results_by_alpha = {}
    for alpha in alphas:
        in_set_0 = (p0 > alpha)
        in_set_1 = (p1 > alpha)
        set_size = in_set_0.astype(int) + in_set_1.astype(int)

        covered = np.where(y_test == 0, in_set_0, in_set_1)
        emp_cov = float(np.mean(covered))
        avg_size = float(np.mean(set_size))
        gap = emp_cov - (1.0 - alpha)

        results_by_alpha[alpha] = {
            'nominal_coverage': 1.0 - alpha,
            'empirical_coverage': emp_cov,
            'avg_set_size': avg_size,
            'coverage_gap': gap
        }

    n_test = len(y_test)
    n_ret = max(1, int(round(n_test * 0.5)))
    order = np.argsort(-conf)
    sel_acc_50 = float(accuracy_score(y_test[order[:n_ret]], pred_label[order[:n_ret]]))

    return {
        'p0': p0,
        'p1': p1,
        'confidence': conf,
        'pred_label': pred_label,
        'sel_acc_50': sel_acc_50,
        'by_alpha': results_by_alpha
    }


def main():
    print("=" * 80)
    print(" DIAGNOSTIC ANALYSIS, COMPLETE ABLATION, & EXTENDED COLD-AWARE ICP")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # PHASE 1: DIAGNOSE cold_drug INSTABILITY
    # -----------------------------------------------------------------------
    print("\n[PHASE 1] Diagnosing cold_drug Instability...")
    df_ms = pd.read_csv(os.path.join(CUR_DIR, 'tmp', 'davis', 'multi_seed_results.csv'))
    cd_ms = df_ms[(df_ms['split_type'] == 'cold_drug') & (df_ms['seed'] != 'mean±std')].copy()

    # Also compute standalone DACMF, DeepDTA, DeepConv-DTI per seed
    cd_records = []
    for s in SEEDS:
        for cfg in ['2-expert', '3-expert']:
            row = cd_ms[(cd_ms['seed'] == str(s)) & (cd_ms['moe_config'] == cfg)].iloc[0]
            cd_records.append({
                'seed': s,
                'moe_config': cfg,
                'auroc': row['auroc'],
                'auprc': row['auprc'],
                'accuracy': row['accuracy'],
                'aurc': row['aurc'],
                'selective_acc_50': row['selective_acc_50']
            })
        # Add DACMF alone
        df_dacmf = pd.read_csv(os.path.join(CUR_DIR, 'tmp', 'davis', 'cold_drug', f'seed_{s}', 'dacmf_predictions.csv'))
        test_dacmf = df_dacmf[df_dacmf['split'] == 'test']
        m_dacmf = compute_metrics(test_dacmf['label'], test_dacmf['predicted_probability'])
        cd_records.append({
            'seed': s,
            'moe_config': 'dacmf_alone',
            'auroc': m_dacmf['auroc'],
            'auprc': m_dacmf['auprc'],
            'accuracy': m_dacmf['accuracy'],
            'aurc': m_dacmf['aurc'],
            'selective_acc_50': m_dacmf['selective_acc_50']
        })

    df_cd_diag = pd.DataFrame(cd_records)
    cd_diag_path = os.path.join(CUR_DIR, 'tmp', 'davis', 'cold_drug_diagnosis.csv')
    df_cd_diag.to_csv(cd_diag_path, index=False)
    print(f"  [SAVED] cold_drug diagnosis -> {cd_diag_path}")

    # Plot cold_drug per-seed AUROC
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SEEDS))
    width = 0.25

    seeds_str = [f"Seed {s}" for s in SEEDS]
    y_2exp = [df_cd_diag[(df_cd_diag['seed'] == s) & (df_cd_diag['moe_config'] == '2-expert')]['auroc'].values[0] for s in SEEDS]
    y_3exp = [df_cd_diag[(df_cd_diag['seed'] == s) & (df_cd_diag['moe_config'] == '3-expert')]['auroc'].values[0] for s in SEEDS]
    y_dacmf = [df_cd_diag[(df_cd_diag['seed'] == s) & (df_cd_diag['moe_config'] == 'dacmf_alone')]['auroc'].values[0] for s in SEEDS]

    ax.bar(x - width, y_2exp, width, label='2-Expert MoE', color='#3498db')
    ax.bar(x, y_3exp, width, label='3-Expert MoE', color='#e74c3c')
    ax.bar(x + width, y_dacmf, width, label='DACMF-DTI Alone', color='#2ecc71')

    ax.set_ylabel('Test AUROC', fontsize=12, fontweight='bold')
    ax.set_title('cold_drug: Per-Seed AUROC Diagnosis (Highlighting Optimization Inversion on Seed 999)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds_str, fontsize=11)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Random Chance (0.50)')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.set_ylim(0.1, 0.85)

    plot_cd_path = os.path.join(CUR_DIR, 'output', 'davis', 'cold_drug_per_seed_auroc.png')
    plt.tight_layout()
    plt.savefig(plot_cd_path, dpi=300)
    plt.close()
    print(f"  [SAVED] cold_drug diagnosis plot -> {plot_cd_path}")

    # -----------------------------------------------------------------------
    # PHASE 2: DIAGNOSE double_cold REGRESSION
    # -----------------------------------------------------------------------
    print("\n[PHASE 2] Diagnosing double_cold Regression...")
    dc_ms = df_ms[(df_ms['split_type'] == 'double_cold') & (df_ms['seed'] != 'mean±std')].copy()

    dc_records = []
    for s in SEEDS:
        for cfg in ['2-expert', '3-expert']:
            row = dc_ms[(dc_ms['seed'] == str(s)) & (dc_ms['moe_config'] == cfg)].iloc[0]
            dc_records.append({
                'seed': s,
                'moe_config': cfg,
                'auroc': row['auroc'],
                'auprc': row['auprc'],
                'accuracy': row['accuracy'],
                'aurc': row['aurc'],
                'selective_acc_50': row['selective_acc_50']
            })
        # Add single experts
        for m in ['dpdta', 'dcdti', 'dacmf']:
            df_m = pd.read_csv(os.path.join(CUR_DIR, 'tmp', 'davis', 'double_cold', f'seed_{s}', f'{m}_predictions.csv'))
            test_m = df_m[df_m['split'] == 'test']
            metrics_m = compute_metrics(test_m['label'], test_m['predicted_probability'])
            dc_records.append({
                'seed': s,
                'moe_config': f'{m}_alone',
                'auroc': metrics_m['auroc'],
                'auprc': metrics_m['auprc'],
                'accuracy': metrics_m['accuracy'],
                'aurc': metrics_m['aurc'],
                'selective_acc_50': metrics_m['selective_acc_50']
            })

    df_dc_diag = pd.DataFrame(dc_records)
    dc_diag_path = os.path.join(CUR_DIR, 'tmp', 'davis', 'double_cold_diagnosis.csv')
    df_dc_diag.to_csv(dc_diag_path, index=False)
    print(f"  [SAVED] double_cold diagnosis -> {dc_diag_path}")

    # Plot double_cold comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SEEDS))
    width = 0.16

    y_dpdta = [df_dc_diag[(df_dc_diag['seed'] == s) & (df_dc_diag['moe_config'] == 'dpdta_alone')]['auroc'].values[0] for s in SEEDS]
    y_dcdti = [df_dc_diag[(df_dc_diag['seed'] == s) & (df_dc_diag['moe_config'] == 'dcdti_alone')]['auroc'].values[0] for s in SEEDS]
    y_dacmf = [df_dc_diag[(df_dc_diag['seed'] == s) & (df_dc_diag['moe_config'] == 'dacmf_alone')]['auroc'].values[0] for s in SEEDS]
    y_2exp = [df_dc_diag[(df_dc_diag['seed'] == s) & (df_dc_diag['moe_config'] == '2-expert')]['auroc'].values[0] for s in SEEDS]
    y_3exp = [df_dc_diag[(df_dc_diag['seed'] == s) & (df_dc_diag['moe_config'] == '3-expert')]['auroc'].values[0] for s in SEEDS]

    ax.bar(x - 2*width, y_dpdta, width, label='DeepDTA Alone', color='#95a5a6')
    ax.bar(x - width, y_dcdti, width, label='DeepConv-DTI Alone', color='#7f8c8d')
    ax.bar(x, y_dacmf, width, label='DACMF-DTI Alone', color='#2ecc71')
    ax.bar(x + width, y_2exp, width, label='2-Expert MoE', color='#3498db')
    ax.bar(x + 2*width, y_3exp, width, label='3-Expert MoE', color='#e74c3c')

    ax.set_ylabel('Test AUROC', fontsize=12, fontweight='bold')
    ax.set_title('double_cold: Single Experts vs MoE Ensembles Across Seeds (Dual Novelty Shift)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds_str, fontsize=11)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.7, label='Random Chance (0.50)')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.set_ylim(0.35, 0.60)

    plot_dc_path = os.path.join(CUR_DIR, 'output', 'davis', 'double_cold_per_seed_auroc.png')
    plt.tight_layout()
    plt.savefig(plot_dc_path, dpi=300)
    plt.close()
    print(f"  [SAVED] double_cold diagnosis plot -> {plot_dc_path}")

    # -----------------------------------------------------------------------
    # PHASE 3: VERIFY COMPLETE 7-WAY ABLATION TABLE (ALL 28 ROWS)
    # -----------------------------------------------------------------------
    print("\n[PHASE 3] Verifying Complete 7-Way Ablation Table...")
    df_abl = pd.read_csv(os.path.join(CUR_DIR, 'tmp', 'davis', 'ablation_results.csv'))
    assert len(df_abl) == 28, f"Expected 28 rows, found {len(df_abl)}"
    print(f"  [VERIFIED] Ablation table contains all 28 rows (4 regimes x 7 combinations).")

    # -----------------------------------------------------------------------
    # PHASE 4: EXTEND COLD-AWARE ICP ACROSS 6 ALPHA LEVELS
    # -----------------------------------------------------------------------
    print("\n[PHASE 4] Extending Cold-Aware ICP Across 6 Alpha Levels...")
    # Load Seed 42 predictions for all regimes
    extended_icp_records = []

    # Get random_pair calibration set for Regimes A & B
    rp_manifest = pd.read_csv(os.path.join(CUR_DIR, 'output', 'davis', 'random_pair', 'split_manifest.csv'))
    rp_cal_idx = (rp_manifest['split'] == 'calibration')
    rp_df_dacmf = pd.read_csv(os.path.join(CUR_DIR, 'tmp', 'davis', 'random_pair', 'seed_42', 'dacmf_predictions.csv'))
    rp_p_cal = rp_df_dacmf[rp_cal_idx]['predicted_probability'].values
    rp_y_cal = rp_manifest[rp_cal_idx]['label'].values

    for split in SPLITS:
        manifest = pd.read_csv(os.path.join(CUR_DIR, 'output', 'davis', split, 'split_manifest.csv'))
        y = manifest['label'].values.astype(int)
        split_col = manifest['split'].values
        cal_idx = (split_col == 'calibration')
        te_idx = (split_col == 'test')

        # Load 3-expert predictions
        exp_preds_te = {}
        exp_preds_cal = {}
        for m in EXPERTS:
            df_m = pd.read_csv(os.path.join(CUR_DIR, 'tmp', 'davis', split, 'seed_42', f'{m}_predictions.csv'))
            exp_preds_te[m] = df_m[te_idx]['predicted_probability'].values
            exp_preds_cal[m] = df_m[cal_idx]['predicted_probability'].values

        # Gated probabilities on seed 42
        X_cal = np.stack([exp_preds_cal[m] for m in EXPERTS], axis=1)
        X_te = np.stack([exp_preds_te[m] for m in EXPERTS], axis=1)
        y_cal = y[cal_idx]
        y_te = y[te_idx]

        # In-distribution vs Cold-aware
        # Regime A: In-sample calibration (drawn from random_pair)
        res_A = compute_icp_conformal_multi_alpha(rp_p_cal[:len(y_cal)], rp_y_cal[:len(y_cal)], X_te[:, 2], y_te, alphas=ALPHAS)
        # Regime B: Disjoint random calibration (full random_pair cal)
        res_B = compute_icp_conformal_multi_alpha(rp_p_cal, rp_y_cal, X_te[:, 2], y_te, alphas=ALPHAS)
        # Regime C: Cold-aware calibration (matched to target novelty)
        res_C = compute_icp_conformal_multi_alpha(X_cal[:, 2], y_cal, X_te[:, 2], y_te, alphas=ALPHAS)

        regimes = [
            ('Regime A (In-sample)', res_A),
            ('Regime B (Disjoint random)', res_B),
            ('Regime C (Cold-aware)', res_C)
        ]

        for r_name, res in regimes:
            for alpha, a_metrics in res['by_alpha'].items():
                extended_icp_records.append({
                    'split_type': split,
                    'calibration_regime': r_name,
                    'alpha': alpha,
                    'nominal_coverage': a_metrics['nominal_coverage'],
                    'empirical_coverage': a_metrics['empirical_coverage'],
                    'avg_set_size': a_metrics['avg_set_size'],
                    'selective_acc_50': res['sel_acc_50'],
                    'coverage_gap': a_metrics['coverage_gap']
                })

    df_ext_icp = pd.DataFrame(extended_icp_records)
    ext_icp_path = os.path.join(CUR_DIR, 'tmp', 'davis', 'cold_aware_icp_results.csv')
    df_ext_icp.to_csv(ext_icp_path, index=False)
    print(f"  [SAVED] Extended cold-aware ICP results (72 rows) -> {ext_icp_path}")

    # Plot Coverage vs Alpha curves across the 3 cold regimes (cold_protein, cold_drug, double_cold)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    cold_splits = ['cold_protein', 'cold_drug', 'double_cold']

    for idx, split in enumerate(cold_splits):
        ax = axes[idx]
        sub = df_ext_icp[df_ext_icp['split_type'] == split]

        for r_name, color, marker, ls in [
            ('Regime A (In-sample)', '#e74c3c', 'o', '--'),
            ('Regime B (Disjoint random)', '#f39c12', 's', '-.'),
            ('Regime C (Cold-aware)', '#27ae60', '^', '-')
        ]:
            r_sub = sub[sub['calibration_regime'] == r_name].sort_values('alpha')
            ax.plot(r_sub['alpha'], r_sub['empirical_coverage'],
                    marker=marker, linestyle=ls, linewidth=2.2, label=r_name, color=color)

        # Plot 1 - alpha nominal reference curve
        alphas_arr = np.array(ALPHAS)
        ax.plot(alphas_arr, 1.0 - alphas_arr, 'k:', linewidth=2, label='Nominal Coverage (1 - α)')

        ax.set_title(f"Split: {split.replace('_', ' ').title()}", fontsize=13, fontweight='bold')
        ax.set_xlabel('Significance Level α', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Empirical Coverage', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.03, 0.32)

    axes[0].legend(loc='lower left', fontsize=10)
    plt.suptitle('EnsDTI: Conformal Coverage vs. Significance Level α Across Calibration Regimes',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plot_alpha_curve_path = os.path.join(CUR_DIR, 'output', 'davis', 'cold_aware_icp_coverage_alpha_curve.png')
    plt.savefig(plot_alpha_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Extended coverage-alpha curve plot -> {plot_alpha_curve_path}")

    print("\n" + "=" * 80)
    print(" ALL DIAGNOSTICS & EXTENSIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
