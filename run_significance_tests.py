"""
run_significance_tests.py:
Executes:
Phase 2: Statistical Significance Testing of Gating Improvements.
Compares:
  1. Original MLP vs Novelty-Aware Gating
  2. Original MLP vs Adaptive Fallback Gating
  3. Original MLP vs Regularized MLP Gating
Across all 4 regimes: random_pair, cold_drug, cold_protein, double_cold.
Runs:
  - Paired Wilcoxon signed-rank test (per-sample absolute error & squared error)
  - Paired Student's t-test (per-sample absolute error & squared error)
  - Bootstrap AUROC hypothesis test (B=2000 resamples for ranking discrimination)
Outputs:
  - tmp/davis/gating_significance_tests.csv
    with columns: comparison, split_type, metric, statistic, p_value, effect_size, significant
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(CUR_DIR, 'tmp', 'davis')
SPLITS = ['random_pair', 'cold_drug', 'cold_protein', 'double_cold']
SEED = 42

COMPARISONS = [
    ('original_vs_novelty_aware', 'original_mlp', 'novelty_aware'),
    ('original_vs_adaptive_fallback', 'original_mlp', 'adaptive_fallback'),
    ('original_vs_regularized', 'original_mlp', 'regularized_mlp')
]


def bootstrap_auroc_test(y_true, p_orig, p_new, n_boot=2000, seed=42):
    """
    Bootstrap paired hypothesis test for H0: AUROC_new <= AUROC_orig vs H1: AUROC_new > AUROC_orig.
    Returns: observed_delta, p_value, 95% CI.
    """
    np.random.seed(seed)
    n = len(y_true)
    obs_orig = roc_auc_score(y_true, p_orig)
    obs_new = roc_auc_score(y_true, p_new)
    obs_delta = float(obs_new - obs_orig)

    deltas = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) > 1:
            auc_o = roc_auc_score(y_true[idx], p_orig[idx])
            auc_n = roc_auc_score(y_true[idx], p_new[idx])
            deltas.append(auc_n - auc_o)
    deltas = np.array(deltas)
    # p-value is fraction of bootstrap replications where delta <= 0
    pval = float(np.mean(deltas <= 0))
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    return obs_delta, pval, (ci_low, ci_high)


def main():
    print("=" * 80)
    print(" PHASE 2: STATISTICAL SIGNIFICANCE TESTING OF GATING IMPROVEMENTS")
    print("=" * 80)

    records = []

    for comp_name, var_orig, var_new in COMPARISONS:
        print(f"\n---> Testing Comparison: {comp_name}")
        for split in SPLITS:
            # Load test predictions from Phase 1 multi-seed directory for Seed 42
            orig_path = os.path.join(TMP_DIR, 'gating_multiseed', var_orig, f'seed_{SEED}', f'{split}_results.csv')
            new_path = os.path.join(TMP_DIR, 'gating_multiseed', var_new, f'seed_{SEED}', f'{split}_results.csv')

            df_orig = pd.read_csv(orig_path)
            df_new = pd.read_csv(new_path)

            y_true = df_orig['true_label'].values.astype(int)
            p_orig = df_orig['predicted_probability'].values.astype(float)
            p_new = df_new['predicted_probability'].values.astype(float)

            # 1. Absolute Error Metrics
            e_orig = np.abs(y_true - p_orig)
            e_new = np.abs(y_true - p_new)
            diff_abs = e_orig - e_new

            # Wilcoxon signed-rank test (testing e_orig > e_new)
            nonzero_abs = diff_abs[diff_abs != 0]
            if len(nonzero_abs) > 0:
                w_stat_abs, w_pval_abs = stats.wilcoxon(e_orig, e_new, alternative='greater')
                # Rank-biserial correlation
                rank_biserial_abs = float((np.sum(diff_abs > 0) - np.sum(diff_abs < 0)) / len(nonzero_abs))
            else:
                w_stat_abs, w_pval_abs, rank_biserial_abs = 0.0, 1.0, 0.0

            # Paired t-test
            t_stat_abs, t_pval_abs = stats.ttest_rel(e_orig, e_new, alternative='greater')
            cohen_d_abs = float(np.mean(diff_abs) / np.std(diff_abs)) if np.std(diff_abs) > 0 else 0.0

            records.append({
                'comparison': comp_name,
                'split_type': split,
                'metric': 'absolute_error_wilcoxon',
                'statistic': float(w_stat_abs),
                'p_value': float(w_pval_abs),
                'effect_size': rank_biserial_abs,
                'significant': bool(w_pval_abs < 0.05)
            })
            records.append({
                'comparison': comp_name,
                'split_type': split,
                'metric': 'absolute_error_ttest',
                'statistic': float(t_stat_abs),
                'p_value': float(t_pval_abs),
                'effect_size': cohen_d_abs,
                'significant': bool(t_pval_abs < 0.05)
            })

            # 2. Squared Error Metrics (Brier Score)
            se_orig = (y_true - p_orig) ** 2
            se_new = (y_true - p_new) ** 2
            diff_sq = se_orig - se_new

            nonzero_sq = diff_sq[diff_sq != 0]
            if len(nonzero_sq) > 0:
                w_stat_sq, w_pval_sq = stats.wilcoxon(se_orig, se_new, alternative='greater')
                rank_biserial_sq = float((np.sum(diff_sq > 0) - np.sum(diff_sq < 0)) / len(nonzero_sq))
            else:
                w_stat_sq, w_pval_sq, rank_biserial_sq = 0.0, 1.0, 0.0

            t_stat_sq, t_pval_sq = stats.ttest_rel(se_orig, se_new, alternative='greater')
            cohen_d_sq = float(np.mean(diff_sq) / np.std(diff_sq)) if np.std(diff_sq) > 0 else 0.0

            records.append({
                'comparison': comp_name,
                'split_type': split,
                'metric': 'squared_error_wilcoxon',
                'statistic': float(w_stat_sq),
                'p_value': float(w_pval_sq),
                'effect_size': rank_biserial_sq,
                'significant': bool(w_pval_sq < 0.05)
            })
            records.append({
                'comparison': comp_name,
                'split_type': split,
                'metric': 'squared_error_ttest',
                'statistic': float(t_stat_sq),
                'p_value': float(t_pval_sq),
                'effect_size': cohen_d_sq,
                'significant': bool(t_pval_sq < 0.05)
            })

            # 3. Bootstrap AUROC Discrimination Test
            delta_auc, pval_boot, (ci_l, ci_h) = bootstrap_auroc_test(y_true, p_orig, p_new, n_boot=2000, seed=42)
            records.append({
                'comparison': comp_name,
                'split_type': split,
                'metric': 'auroc_bootstrap_delta',
                'statistic': float(delta_auc),
                'p_value': float(pval_boot),
                'effect_size': float(delta_auc),
                'significant': bool(pval_boot < 0.05)
            })

            print(f"  [{split}] AUROC Delta: {delta_auc:+.4f} (p={pval_boot:.4f}, sig={pval_boot < 0.05}) | "
                  f"MAE Wilcoxon p={w_pval_abs:.4e} | t-test p={t_pval_abs:.4e}")

    df_sig = pd.DataFrame(records)
    sig_path = os.path.join(TMP_DIR, 'gating_significance_tests.csv')
    df_sig.to_csv(sig_path, index=False)
    print(f"\n[SAVED] Significance test results ({len(df_sig)} tests) -> {sig_path}")

    # Summary of double_cold AUROC recovery tests
    print("\n" + "=" * 80)
    print(" DOUBLE_COLD RECOVERY SIGNIFICANCE HIGHLIGHT")
    print("=" * 80)
    dc_sub = df_sig[(df_sig['split_type'] == 'double_cold') & (df_sig['metric'] == 'auroc_bootstrap_delta')]
    print(dc_sub[['comparison', 'statistic', 'p_value', 'significant']].to_string(index=False))


if __name__ == '__main__':
    main()
