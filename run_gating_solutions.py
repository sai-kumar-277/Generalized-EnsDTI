"""
run_gating_solutions.py — Complete implementation, evaluation, and comparison of:
  - Phase 1: Adaptive Fallback Gating (Option C)
  - Phase 2: Novelty-Aware Gating (Option B)
  - Phase 3: Regularized Gating (Option A)
  - Phase 4: Cross-Method Gating Comparison Table
  - Phase 5: Re-running Cold-Aware ICP with Winning Gating Architecture
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
TMP_DIR = os.path.join(CUR_DIR, 'tmp', 'davis')
OUT_DIR = os.path.join(CUR_DIR, 'output', 'davis')
SPLITS = ['random_pair', 'cold_drug', 'cold_protein', 'double_cold']
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


def load_split_data(split, seed=42, experts=('dpdta', 'dcdti', 'dacmf')):
    """Load train, calibration, and test expert predictions for a split."""
    data_dir = os.path.join(TMP_DIR, split)
    seed_dir = os.path.join(data_dir, f'seed_{seed}')
    
    def load_part(part):
        dfs = []
        for e in experts:
            p = os.path.join(data_dir, f'{e}_predictions_{part}.csv')
            if not os.path.exists(p) and os.path.exists(seed_dir):
                candidate = os.path.join(seed_dir, f'{e}_predictions_{part}.csv')
                if os.path.exists(candidate):
                    p = candidate
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing {p}")
            df = pd.read_csv(p).rename(columns={'predicted_probability': f'prob_{e}'})
            dfs.append(df.set_index('original_row_id'))
        
        merged = pd.concat([d[[f'prob_{e}']] for d, e in zip(dfs, experts)], axis=1)
        label_df = dfs[0]
        merged['label'] = label_df['label']
        merged['drug_id'] = label_df['drug_id']
        merged['protein_id'] = label_df['protein_id']
        merged['split'] = part
        return merged
    
    train_df = load_part('train')
    cal_df = load_part('calibration')
    test_df = load_part('test')
    return train_df, cal_df, test_df


def main():
    print("=" * 80)
    print(" EXECUTING MoE GATING SOLUTIONS (PHASES 1-5)")
    print("=" * 80)

    results_adaptive = []
    results_novelty = []
    results_regularized = []
    results_comparison = []

    winning_predictions = {}

    for split in SPLITS:
        print(f"\n---> Evaluating Split: {split} (Seed 42)")
        for cfg in ['2-expert', '3-expert']:
            experts = ['dpdta', 'dcdti'] if cfg == '2-expert' else ['dpdta', 'dcdti', 'dacmf']
            feature_cols = [f'prob_{e}' for e in experts]
            
            train_df, cal_df, test_df = load_split_data(split, seed=42, experts=experts)

            X_train = train_df[feature_cols].values
            y_train = train_df['label'].values.astype(int)
            X_cal = cal_df[feature_cols].values
            y_cal = cal_df['label'].values.astype(int)
            X_test = test_df[feature_cols].values
            y_test = test_df['label'].values.astype(int)

            train_drugs = set(train_df['drug_id'].values)
            train_prots = set(train_df['protein_id'].values)

            # Novelty features
            is_d_cal = (~cal_df['drug_id'].isin(train_drugs)).astype(float).values
            is_p_cal = (~cal_df['protein_id'].isin(train_prots)).astype(float).values
            is_inter_cal = is_d_cal * is_p_cal
            X_cal_nov = np.column_stack([X_cal, is_d_cal, is_p_cal, is_inter_cal])

            is_d_test = (~test_df['drug_id'].isin(train_drugs)).astype(float).values
            is_p_test = (~test_df['protein_id'].isin(train_prots)).astype(float).values
            is_inter_test = is_d_test * is_p_test
            X_test_nov = np.column_stack([X_test, is_d_test, is_p_test, is_inter_test])

            is_shift_test = (is_d_test > 0) | (is_p_test > 0)
            is_shift_cal = (is_d_cal > 0) | (is_p_cal > 0)

            # -------------------------------------------------------------
            # 1. Original MLP Gating (exact benchmark baseline)
            # -------------------------------------------------------------
            clf_orig = MLPClassifier(
                hidden_layer_sizes=(15,),
                activation='logistic',
                solver='adam',
                learning_rate_init=0.01,
                max_iter=35,
                random_state=42,
                early_stopping=True
            )
            clf_orig.fit(X_train, y_train)
            p_test_orig = clf_orig.predict_proba(X_test)[:, 1]
            m_orig = compute_metrics(y_test, p_test_orig)

            # -------------------------------------------------------------
            # 2. Regularized MLP Gating (Phase 3)
            # -------------------------------------------------------------
            clf_reg = MLPClassifier(
                hidden_layer_sizes=(10,),
                activation='logistic',
                alpha=1.0,
                max_iter=100,
                random_state=42
            )
            clf_reg.fit(X_train, y_train)
            p_test_reg = clf_reg.predict_proba(X_test)[:, 1]
            m_reg = compute_metrics(y_test, p_test_reg)

            results_regularized.append({
                'split_type': split,
                'moe_config': cfg,
                'gating_mode': 'regularized_mlp',
                'auroc': m_reg['auroc'],
                'auprc': m_reg['auprc'],
                'accuracy': m_reg['accuracy'],
                'aurc': m_reg['aurc'],
                'selective_acc_50': m_reg['selective_acc_50']
            })

            # -------------------------------------------------------------
            # 3. Adaptive Fallback Gating (Phase 1)
            # -------------------------------------------------------------
            # Fallback Option A: Simple Average under shift
            p_test_avg = np.mean(X_test, axis=1)
            p_test_adapt_avg = np.where(is_shift_test, p_test_avg, p_test_orig)
            m_adapt_avg = compute_metrics(y_test, p_test_adapt_avg)

            # Fallback Option B: Best single expert on shift
            if cfg == '3-expert':
                if split in ['cold_drug', 'double_cold']:
                    p_best_test = X_test[:, 2] # dacmf
                elif split == 'cold_protein':
                    p_best_test = X_test[:, 1] # dcdti
                else:
                    p_best_test = p_test_orig
            else: # 2-expert
                p_best_test = X_test[:, 1] if split == 'cold_protein' else X_test[:, 0]
            
            p_test_adapt_best = np.where(is_shift_test, p_best_test, p_test_orig)
            m_adapt_best = compute_metrics(y_test, p_test_adapt_best)

            # Log Phase 1 results
            results_adaptive.append({
                'split_type': split,
                'moe_config': cfg,
                'gating_mode': 'original_mlp',
                'auroc': m_orig['auroc'],
                'auprc': m_orig['auprc'],
                'accuracy': m_orig['accuracy'],
                'aurc': m_orig['aurc'],
                'selective_acc_50': m_orig['selective_acc_50']
            })
            results_adaptive.append({
                'split_type': split,
                'moe_config': cfg,
                'gating_mode': 'adaptive_fallback_average',
                'auroc': m_adapt_avg['auroc'],
                'auprc': m_adapt_avg['auprc'],
                'accuracy': m_adapt_avg['accuracy'],
                'aurc': m_adapt_avg['aurc'],
                'selective_acc_50': m_adapt_avg['selective_acc_50']
            })
            results_adaptive.append({
                'split_type': split,
                'moe_config': cfg,
                'gating_mode': 'adaptive_fallback_best_expert',
                'auroc': m_adapt_best['auroc'],
                'auprc': m_adapt_best['auprc'],
                'accuracy': m_adapt_best['accuracy'],
                'aurc': m_adapt_best['aurc'],
                'selective_acc_50': m_adapt_best['selective_acc_50']
            })

            # -------------------------------------------------------------
            # 4. Novelty-Aware Gating (Phase 2)
            # -------------------------------------------------------------
            np.random.seed(42)
            train_drugs_list = list(train_drugs)
            train_prots_list = list(train_prots)
            sim_novel_drugs = set(np.random.choice(train_drugs_list, size=int(0.25 * len(train_drugs_list)), replace=False))
            sim_novel_prots = set(np.random.choice(train_prots_list, size=int(0.25 * len(train_prots_list)), replace=False))

            is_d_sim = train_df['drug_id'].isin(sim_novel_drugs).astype(float).values
            is_p_sim = train_df['protein_id'].isin(sim_novel_prots).astype(float).values
            is_inter_sim = is_d_sim * is_p_sim
            X_train_nov = np.column_stack([X_train, is_d_sim, is_p_sim, is_inter_sim])

            clf_nov = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', alpha=1.0, max_iter=100, random_state=42)
            clf_nov.fit(X_train_nov, y_train)
            p_test_nov = clf_nov.predict_proba(X_test_nov)[:, 1]
            m_nov = compute_metrics(y_test, p_test_nov)

            results_novelty.append({
                'split_type': split,
                'moe_config': cfg,
                'gating_mode': 'novelty_aware_mlp',
                'auroc': m_nov['auroc'],
                'auprc': m_nov['auprc'],
                'accuracy': m_nov['accuracy'],
                'aurc': m_nov['aurc'],
                'selective_acc_50': m_nov['selective_acc_50']
            })

            # -------------------------------------------------------------
            # 5. Cross-Method Comparison (Phase 4)
            # -------------------------------------------------------------
            results_comparison.append({
                'gating_variant': 'original_mlp',
                'split_type': split,
                'moe_config': cfg,
                'auroc': m_orig['auroc'],
                'auprc': m_orig['auprc'],
                'accuracy': m_orig['accuracy'],
                'aurc': m_orig['aurc'],
                'selective_acc_50': m_orig['selective_acc_50']
            })
            results_comparison.append({
                'gating_variant': 'regularized_mlp',
                'split_type': split,
                'moe_config': cfg,
                'auroc': m_reg['auroc'],
                'auprc': m_reg['auprc'],
                'accuracy': m_reg['accuracy'],
                'aurc': m_reg['aurc'],
                'selective_acc_50': m_reg['selective_acc_50']
            })
            results_comparison.append({
                'gating_variant': 'adaptive_fallback',
                'split_type': split,
                'moe_config': cfg,
                'auroc': m_adapt_avg['auroc'],
                'auprc': m_adapt_avg['auprc'],
                'accuracy': m_adapt_avg['accuracy'],
                'aurc': m_adapt_avg['aurc'],
                'selective_acc_50': m_adapt_avg['selective_acc_50']
            })
            results_comparison.append({
                'gating_variant': 'novelty_aware',
                'split_type': split,
                'moe_config': cfg,
                'auroc': m_nov['auroc'],
                'auprc': m_nov['auprc'],
                'accuracy': m_nov['accuracy'],
                'aurc': m_nov['aurc'],
                'selective_acc_50': m_nov['selective_acc_50']
            })

            # For Phase 5 ICP, save 3-expert winning gating predictions (Novelty-Aware)
            if cfg == '3-expert':
                p_cal_nov = clf_nov.predict_proba(X_cal_nov)[:, 1]
                winning_predictions[split] = {
                    'p_cal': p_cal_nov,
                    'y_cal': y_cal,
                    'p_test': p_test_nov,
                    'y_test': y_test
                }

    # Save CSVs
    df_adapt = pd.DataFrame(results_adaptive)
    adapt_path = os.path.join(TMP_DIR, 'adaptive_gating_results.csv')
    df_adapt.to_csv(adapt_path, index=False)
    print(f"\n[SAVED] Adaptive gating results -> {adapt_path}")

    df_nov = pd.DataFrame(results_novelty)
    nov_path = os.path.join(TMP_DIR, 'novelty_aware_gating_results.csv')
    df_nov.to_csv(nov_path, index=False)
    print(f"[SAVED] Novelty-aware gating results -> {nov_path}")

    df_reg = pd.DataFrame(results_regularized)
    reg_path = os.path.join(TMP_DIR, 'regularized_gating_results.csv')
    df_reg.to_csv(reg_path, index=False)
    print(f"[SAVED] Regularized gating results -> {reg_path}")

    df_comp = pd.DataFrame(results_comparison)
    comp_path = os.path.join(TMP_DIR, 'gating_comparison.csv')
    df_comp.to_csv(comp_path, index=False)
    print(f"[SAVED] Gating comparison table -> {comp_path}")

    # -------------------------------------------------------------
    # PHASE 5: RE-RUN COLD-AWARE ICP WITH WINNING GATING ARCHITECTURE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [PHASE 5] Re-running Cold-Aware ICP with Winning Gating Architecture")
    print("=" * 80)

    rp_cal_p = winning_predictions['random_pair']['p_cal']
    rp_cal_y = winning_predictions['random_pair']['y_cal']

    icp_updated_records = []

    for split in SPLITS:
        win = winning_predictions[split]
        p_te = win['p_test']
        y_te = win['y_test']
        p_cal = win['p_cal']
        y_cal = win['y_cal']

        # Regime A: In-sample calibration (matched length)
        res_A = compute_icp_conformal_multi_alpha(rp_cal_p[:len(y_cal)], rp_cal_y[:len(y_cal)], p_te, y_te, alphas=ALPHAS)
        # Regime B: Disjoint random calibration (full random_pair cal)
        res_B = compute_icp_conformal_multi_alpha(rp_cal_p, rp_cal_y, p_te, y_te, alphas=ALPHAS)
        # Regime C: Cold-aware calibration (matched to entity shift)
        res_C = compute_icp_conformal_multi_alpha(p_cal, y_cal, p_te, y_te, alphas=ALPHAS)

        regimes = [
            ('Regime A (In-sample)', res_A),
            ('Regime B (Disjoint random)', res_B),
            ('Regime C (Cold-aware)', res_C)
        ]

        for r_name, res in regimes:
            for alpha, a_metrics in res['by_alpha'].items():
                icp_updated_records.append({
                    'split_type': split,
                    'calibration_regime': r_name,
                    'alpha': alpha,
                    'nominal_coverage': a_metrics['nominal_coverage'],
                    'empirical_coverage': a_metrics['empirical_coverage'],
                    'avg_set_size': a_metrics['avg_set_size'],
                    'selective_acc_50': res['sel_acc_50'],
                    'coverage_gap': a_metrics['coverage_gap']
                })

    df_updated_icp = pd.DataFrame(icp_updated_records)
    updated_icp_path = os.path.join(TMP_DIR, 'cold_aware_icp_results.csv')
    df_updated_icp.to_csv(updated_icp_path, index=False)
    print(f"[SAVED] Updated cold-aware ICP results (72 rows) -> {updated_icp_path}")

    # Plot Coverage vs Alpha curves across cold splits
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    cold_splits = ['cold_protein', 'cold_drug', 'double_cold']

    regime_colors = {
        'Regime A (In-sample)': '#e74c3c',
        'Regime B (Disjoint random)': '#f39c12',
        'Regime C (Cold-aware)': '#27ae60'
    }
    regime_markers = {
        'Regime A (In-sample)': 'x',
        'Regime B (Disjoint random)': 's',
        'Regime C (Cold-aware)': 'o'
    }

    for idx, split in enumerate(cold_splits):
        ax = axes[idx]
        sub = df_updated_icp[df_updated_icp['split_type'] == split]

        # Theoretical line
        ax.plot(ALPHAS, [1.0 - a for a in ALPHAS], 'k--', label='Theoretical Coverage (1 - α)', linewidth=2.0)

        for r_name in ['Regime A (In-sample)', 'Regime B (Disjoint random)', 'Regime C (Cold-aware)']:
            r_sub = sub[sub['calibration_regime'] == r_name].sort_values('alpha')
            ax.plot(r_sub['alpha'], r_sub['empirical_coverage'],
                    color=regime_colors[r_name], marker=regime_markers[r_name],
                    linewidth=2.2, markersize=7, label=r_name)

        ax.set_title(f"Split: {split.replace('_', ' ').title()}", fontsize=13, fontweight='bold')
        ax.set_xlabel('Significance Level (α)', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Empirical Coverage', fontsize=12, fontweight='bold')
        ax.set_xticks(ALPHAS)
        ax.set_xticklabels([f"{a:.2f}" for a in ALPHAS], fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
        ax.set_ylim(0.40, 1.02)

    plt.suptitle("Cold-Aware ICP Calibration Curves (Winning Novelty-Aware MoE Architecture)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    curve_plot_path = os.path.join(OUT_DIR, 'cold_aware_icp_coverage_alpha_curve.png')
    plt.savefig(curve_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Updated ICP Coverage vs Alpha plot -> {curve_plot_path}")

    # Summary table output
    print("\n" + "=" * 80)
    print(" SUMMARY OF GATING COMPARISON (3-EXPERT MoE, SEED 42)")
    print("=" * 80)
    df_3exp = df_comp[df_comp['moe_config'] == '3-expert']
    pivot_auc = df_3exp.pivot(index='gating_variant', columns='split_type', values='auroc')[SPLITS]
    print(pivot_auc.round(4).to_string())

    print("\n" + "=" * 80)
    print(" SUMMARY OF GATING COMPARISON (2-EXPERT MoE, SEED 42)")
    print("=" * 80)
    df_2exp = df_comp[df_comp['moe_config'] == '2-expert']
    pivot_2exp = df_2exp.pivot(index='gating_variant', columns='split_type', values='auroc')[SPLITS]
    print(pivot_2exp.round(4).to_string())


if __name__ == '__main__':
    main()
