"""
compare_moe_experts.py:
Evaluates and compares:
- 2-Expert MoE: DeepDTA + DeepConv-DTI
- 3-Expert MoE: DeepDTA + DeepConv-DTI + DACMF-DTI

Metrics:
- AUROC
- AUPRC
- Accuracy
- AURC (Area Under Risk-Coverage Curve)
- Selective Accuracy @ 50% Retention
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS = ['random_pair', 'cold_drug', 'cold_protein', 'double_cold']

def compute_metrics_for_df(df: pd.DataFrame):
    """
    Expects df with columns:
    'true_label', 'moe_predicted_probability', 'predicted_label', 'confidence'
    """
    y_true = df['true_label'].values.astype(int)
    y_prob = df['moe_predicted_probability'].values.astype(float)
    y_pred = df['predicted_label'].values.astype(int)
    conf = df['confidence'].values.astype(float)

    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    acc = accuracy_score(y_true, y_pred)

    # Selective Accuracy @ 50%
    n = len(df)
    n_ret = max(1, int(round(n * 0.5)))
    df_sorted = df.sort_values('confidence', ascending=False)
    ret_50 = df_sorted.iloc[:n_ret]
    sel_acc_50 = accuracy_score(ret_50['true_label'], ret_50['predicted_label'])

    # AURC
    thresholds = np.linspace(0, 1, 50)
    covs, risks = [], []
    for thr in thresholds:
        sub = df[df['confidence'] >= thr]
        if len(sub) == 0:
            continue
        c = len(sub) / n
        r = (sub['predicted_label'] != sub['true_label']).mean()
        covs.append(c)
        risks.append(r)
    if len(covs) > 1:
        # Sort coverages ascending for integration
        order = np.argsort(covs)
        covs_s = np.array(covs)[order]
        risks_s = np.array(risks)[order]
        aurc = float(np.trapz(risks_s, covs_s))
    else:
        aurc = float('nan')

    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': acc,
        'aurc': aurc,
        'sel_acc_50': sel_acc_50
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='davis')
    args = parser.parse_args()

    print("=" * 80)
    print(" COMPARISON: 2-EXPERT vs 3-EXPERT MoE ENSEMBLE ON DAVIS")
    print("=" * 80)

    rows = []
    for split in SPLITS:
        out_dir = os.path.join(CUR_DIR, 'output', args.dataset, split)
        icp_path_2 = os.path.join(out_dir, 'icp_results_2expert.csv')
        icp_path_3 = os.path.join(out_dir, 'icp_results_3expert.csv')
        standard_icp = os.path.join(out_dir, 'icp_results.csv')

        # Check for 2-expert results
        target_2 = icp_path_2 if os.path.exists(icp_path_2) else standard_icp
        if os.path.exists(target_2):
            df2 = pd.read_csv(target_2)
            m2 = compute_metrics_for_df(df2)
        else:
            m2 = {'auroc': np.nan, 'auprc': np.nan, 'accuracy': np.nan, 'aurc': np.nan, 'sel_acc_50': np.nan}

        # Check for 3-expert results
        if os.path.exists(icp_path_3):
            df3 = pd.read_csv(icp_path_3)
            m3 = compute_metrics_for_df(df3)
        else:
            m3 = {'auroc': np.nan, 'auprc': np.nan, 'accuracy': np.nan, 'aurc': np.nan, 'sel_acc_50': np.nan}

        rows.append({
            'Split': split,
            '2-Exp AUROC': f"{m2['auroc']:.4f}" if not np.isnan(m2['auroc']) else "N/A",
            '3-Exp AUROC': f"{m3['auroc']:.4f}" if not np.isnan(m3['auroc']) else "Pending Colab",
            '2-Exp AUPRC': f"{m2['auprc']:.4f}" if not np.isnan(m2['auprc']) else "N/A",
            '3-Exp AUPRC': f"{m3['auprc']:.4f}" if not np.isnan(m3['auprc']) else "Pending Colab",
            '2-Exp Acc': f"{m2['accuracy']*100:.2f}%" if not np.isnan(m2['accuracy']) else "N/A",
            '3-Exp Acc': f"{m3['accuracy']*100:.2f}%" if not np.isnan(m3['accuracy']) else "Pending Colab",
            '2-Exp AURC': f"{m2['aurc']:.4f}" if not np.isnan(m2['aurc']) else "N/A",
            '3-Exp AURC': f"{m3['aurc']:.4f}" if not np.isnan(m3['aurc']) else "Pending Colab",
            '2-Exp SelAcc@50%': f"{m2['sel_acc_50']*100:.2f}%" if not np.isnan(m2['sel_acc_50']) else "N/A",
            '3-Exp SelAcc@50%': f"{m3['sel_acc_50']*100:.2f}%" if not np.isnan(m3['sel_acc_50']) else "Pending Colab"
        })

    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    # Save summary table
    out_csv = os.path.join(CUR_DIR, 'output', args.dataset, 'MOE_2EXPERT_VS_3EXPERT_COMPARISON.csv')
    comp_df.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Comparison summary saved to {out_csv}")

if __name__ == '__main__':
    main()
