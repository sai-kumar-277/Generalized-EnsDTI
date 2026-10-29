"""
evaluate_results.py — Comprehensive evaluation with metrics, confidence analysis,
and risk-coverage curves.

Usage
-----
  python evaluate_results.py --dataset davis --split-type random_pair
  python evaluate_results.py --dataset davis  # evaluates all available split types
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows sandbox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, roc_curve, precision_recall_curve
)

warnings.filterwarnings('ignore')

SPLIT_TYPES = ['random_pair', 'cold_drug', 'cold_protein', 'double_cold']
RETAIN_LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_prediction_metrics(df: pd.DataFrame) -> dict:
    """Compute classification metrics from ICP results."""
    y_true = df['true_label'].values
    y_prob = df['moe_predicted_probability'].values
    y_pred = df['predicted_label'].values

    if len(np.unique(y_true)) < 2:
        return {'note': 'Only one class present — metrics not computable'}

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'auroc': float(roc_auc_score(y_true, y_prob)),
        'auprc': float(average_precision_score(y_true, y_prob)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'n_test': int(len(df)),
        'n_positive': int(y_true.sum()),
        'n_negative': int((1 - y_true).sum()),
    }


def compute_selective_metrics(df: pd.DataFrame, retain_frac: float) -> dict:
    """Compute selective accuracy at a given retention fraction."""
    n = len(df)
    n_retain = max(1, int(round(n * retain_frac)))
    # Retain examples with highest confidence
    df_sorted = df.sort_values('confidence', ascending=False)
    df_ret = df_sorted.iloc[:n_retain]
    y_true = df_ret['true_label'].values
    y_pred = df_ret['predicted_label'].values
    error_rate = (y_pred != y_true).mean()
    return {
        'retain_frac': float(retain_frac),
        'n_retained': int(n_retain),
        'coverage': float(n_retain / n),
        'selective_risk': float(error_rate),
        'selective_accuracy': float(1 - error_rate),
    }


def compute_risk_coverage(df: pd.DataFrame, n_points: int = 50):
    """Compute risk-coverage curve by sweeping confidence threshold."""
    thresholds = np.linspace(0, 1, n_points)
    coverages, risks = [], []
    n = len(df)
    for thr in thresholds:
        retained = df[df['confidence'] >= thr]
        if len(retained) == 0:
            continue
        risk = (retained['predicted_label'] != retained['true_label']).mean()
        coverage = len(retained) / n
        coverages.append(coverage)
        risks.append(risk)
    return np.array(coverages), np.array(risks)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confidence_distribution(df: pd.DataFrame, out_path: str):
    """Histogram of confidence scores."""
    fig, ax = plt.subplots(figsize=(7, 4))
    correct = df[df['predicted_label'] == df['true_label']]['confidence']
    wrong = df[df['predicted_label'] != df['true_label']]['confidence']
    ax.hist(correct, bins=30, alpha=0.6, label='Correct', color='steelblue', density=True)
    ax.hist(wrong, bins=30, alpha=0.6, label='Incorrect', color='tomato', density=True)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_error_by_bin(df: pd.DataFrame, out_path: str, n_bins: int = 10):
    """Error rate by confidence bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, error_rates, counts = [], [], []
    for i in range(n_bins):
        mask = (df['confidence'] >= bins[i]) & (df['confidence'] < bins[i + 1])
        sub = df[mask]
        if len(sub) == 0:
            continue
        err = (sub['predicted_label'] != sub['true_label']).mean()
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        error_rates.append(err)
        counts.append(len(sub))

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.bar(bin_centers, error_rates, width=0.08, color='tomato', alpha=0.7, label='Error rate')
    ax2.plot(bin_centers, counts, 'o--', color='navy', label='Count')
    ax1.set_xlabel('Confidence bin')
    ax1.set_ylabel('Error Rate', color='tomato')
    ax2.set_ylabel('Count', color='navy')
    ax1.set_title('Error Rate by Confidence Bin')
    ax1.axhline(0.05, ls='--', color='gray', alpha=0.5, label='5% error')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_error_vs_threshold(df: pd.DataFrame, out_path: str):
    """Error rate vs confidence threshold."""
    thresholds = np.linspace(0, 0.99, 50)
    errors, coverages = [], []
    n = len(df)
    for thr in thresholds:
        retained = df[df['confidence'] > thr]
        if len(retained) == 0:
            errors.append(np.nan)
            coverages.append(0.0)
        else:
            err = (retained['predicted_label'] != retained['true_label']).mean()
            errors.append(err)
            coverages.append(len(retained) / n)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(thresholds, errors, color='tomato', label='Error rate')
    ax2.plot(thresholds, coverages, color='steelblue', ls='--', label='Coverage')
    ax1.set_xlabel('Confidence threshold')
    ax1.set_ylabel('Error Rate', color='tomato')
    ax2.set_ylabel('Coverage', color='steelblue')
    ax1.set_title('Error Rate vs Confidence Threshold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_risk_coverage(coverages_risks_dict: dict, out_path: str):
    """Risk-coverage curves for multiple split types."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['steelblue', 'tomato', 'forestgreen', 'darkorange']
    for i, (name, (covs, risks)) in enumerate(coverages_risks_dict.items()):
        if len(covs) > 0:
            ax.plot(covs, risks, label=name, color=colors[i % len(colors)], lw=2)
    ax.set_xlabel('Coverage (fraction retained)')
    ax.set_ylabel('Selective Risk (error rate)')
    ax.set_title('Risk–Coverage Curves')
    ax.legend()
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_pr_curves(df: pd.DataFrame, out_path: str, split_type: str):
    """ROC and Precision-Recall curves."""
    y_true = df['true_label'].values
    y_prob = df['moe_predicted_probability'].values
    if len(np.unique(y_true)) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, lw=2, color='steelblue', label=f'AUC={auroc:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve ({split_type})')
    ax1.legend()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    ax2.plot(rec, prec, lw=2, color='tomato', label=f'AP={auprc:.3f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve ({split_type})')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Overlap analysis
# ---------------------------------------------------------------------------

def compute_overlap_analysis(manifest: pd.DataFrame, split_type: str) -> dict:
    """Compute entity overlap statistics from split manifest."""
    train = manifest[manifest['split'] == 'train']
    cal = manifest[manifest['split'] == 'calibration']
    test = manifest[manifest['split'] == 'test']

    train_pairs = set(zip(train['drug_id'], train['protein_id']))
    cal_pairs = set(zip(cal['drug_id'], cal['protein_id']))
    test_pairs = set(zip(test['drug_id'], test['protein_id']))

    train_drugs = set(train['drug_id'])
    cal_drugs = set(cal['drug_id'])
    test_drugs = set(test['drug_id'])

    train_proteins = set(train['protein_id'])
    cal_proteins = set(cal['protein_id'])
    test_proteins = set(test['protein_id'])

    return {
        'calibration_pair_overlap_with_training': len(cal_pairs & train_pairs),
        'calibration_drug_overlap_with_training': len(cal_drugs & train_drugs),
        'calibration_protein_overlap_with_training': len(cal_proteins & train_proteins),
        'test_drug_overlap_with_training': len(test_drugs & train_drugs),
        'test_protein_overlap_with_training': len(test_proteins & train_proteins),
        'test_pair_overlap_with_calibration': len(test_pairs & cal_pairs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_split(dataset: str, split_type: str) -> dict:
    """Evaluate results for one (dataset, split_type) combination."""
    icp_path = os.path.join('output', dataset, split_type, 'icp_results.csv')
    manifest_path = os.path.join('output', dataset, split_type, 'split_manifest.csv')
    out_dir = os.path.join('output', dataset, split_type)

    if not os.path.exists(icp_path):
        print(f"  [SKIP] {icp_path} not found")
        return {}

    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset} / {split_type}")
    print(f"{'='*60}")

    df = pd.read_csv(icp_path)

    # Prediction metrics
    metrics = compute_prediction_metrics(df)
    print(f"\n  Prediction Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:30s}: {v:.4f}")
        else:
            print(f"    {k:30s}: {v}")

    # Selective metrics
    selective = []
    print(f"\n  Selective Metrics (by retention level):")
    print(f"  {'Retain%':>8}  {'Coverage':>10}  {'Risk':>8}  {'Accuracy':>10}")
    for retain in RETAIN_LEVELS:
        sm = compute_selective_metrics(df, retain)
        selective.append(sm)
        print(f"  {retain*100:>7.0f}%  {sm['coverage']:>10.4f}  "
              f"{sm['selective_risk']:>8.4f}  {sm['selective_accuracy']:>10.4f}")

    # Risk-coverage curve
    covs, risks = compute_risk_coverage(df)
    if len(covs) > 1:
        sort_idx = np.argsort(covs)
        try:
            aurc = float(np.trapezoid(risks[sort_idx], covs[sort_idx]))
        except AttributeError:
            aurc = float(np.trapz(risks[sort_idx], covs[sort_idx]))
    else:
        aurc = np.nan
    print(f"\n  Area under Risk-Coverage curve (AURC): {aurc:.4f}")

    # Overlap analysis
    overlap = {}
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        overlap = compute_overlap_analysis(manifest, split_type)
        print(f"\n  Calibration/Test Overlap Analysis:")
        for k, v in overlap.items():
            print(f"    {k:50s}: {v}")

    # Plots
    plot_confidence_distribution(df, os.path.join(out_dir, 'confidence_distribution.png'))
    plot_error_by_bin(df, os.path.join(out_dir, 'error_by_confidence_bin.png'))
    plot_error_vs_threshold(df, os.path.join(out_dir, 'error_vs_threshold.png'))
    plot_roc_pr_curves(df, os.path.join(out_dir, 'roc_pr_curves.png'), split_type)

    print(f"\n  Plots saved to {out_dir}/")

    # Save metrics JSON
    result = {
        'dataset': dataset,
        'split_type': split_type,
        'metrics': metrics,
        'selective_metrics': selective,
        'aurc': float(aurc) if not np.isnan(aurc) else None,
        'overlap_analysis': overlap,
    }
    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Metrics saved -> {metrics_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate EnsDTI results')
    parser.add_argument('--dataset', '-d', default='davis')
    parser.add_argument('--split-type', '-s', default=None,
                        help='Specific split type (default: all available)')
    args = parser.parse_args()

    split_types = [args.split_type] if args.split_type else SPLIT_TYPES
    all_results = []
    risk_coverage_data = {}

    for split_type in split_types:
        result = evaluate_split(args.dataset, split_type)
        if result:
            all_results.append(result)
            icp_path = os.path.join('output', args.dataset, split_type, 'icp_results.csv')
            if os.path.exists(icp_path):
                df = pd.read_csv(icp_path)
                covs, risks = compute_risk_coverage(df)
                risk_coverage_data[split_type] = (covs, risks)

    # Combined risk-coverage plot
    if risk_coverage_data:
        out_path = os.path.join('output', args.dataset, 'risk_coverage_all.png')
        plot_risk_coverage(risk_coverage_data, out_path)
        print(f"\nCombined risk-coverage plot -> {out_path}")

    # Summary table
    if all_results:
        summary_rows = []
        for r in all_results:
            m = r.get('metrics', {})
            summary_rows.append({
                'split_type': r['split_type'],
                'accuracy': m.get('accuracy'),
                'auroc': m.get('auroc'),
                'auprc': m.get('auprc'),
                'f1': m.get('f1'),
                'n_test': m.get('n_test'),
                'aurc': r.get('aurc'),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join('output', args.dataset, 'evaluation_summary.csv')
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False, float_format='%.4f'))
        print(f"\nSummary saved -> {summary_path}")


if __name__ == '__main__':
    main()
