"""
run_pipeline.py — End-to-end leakage-controlled EnsDTI evaluation pipeline.

Stages:
  1. Load data and generate split
  2. Save split manifest
  3. Run leakage audit
  4. Train expert models (sklearn substitutes) on training split
  5. Generate expert predictions for calibration and test splits
  6. Train MoE gating MLP on training predictions only
  7. Generate MoE predictions for calibration and test splits
  8. Run ICP using calibration predictions only
  9. Save ICP results for test predictions
  10. Run evaluation and save metrics/plots

Usage
-----
  python run_pipeline.py \\
      --dataset davis \\
      --split-type random_pair \\
      --seed 42 \\
      --test-frac 0.2 \\
      --calibration-frac 0.1 \\
      --use-wandb false

Supported split types: random_pair, cold_drug, cold_protein, double_cold
"""

import os
import sys
import json
import time
import argparse
import subprocess
import pandas as pd


def run_step(step_name: str, cmd: list, cwd: str = '.'):
    """Run a pipeline step and exit if it fails."""
    print(f"\n{'#'*60}")
    print(f"# STEP: {step_name}")
    print(f"{'#'*60}")
    print(f"  Command: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, text=True,
                            capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAIL] Step '{step_name}' FAILED (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"  [OK] Completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end EnsDTI leakage-controlled pipeline')
    parser.add_argument('--dataset', '-d', default='davis',
                        help='Dataset name (e.g. davis, human, kiba)')
    parser.add_argument('--split-type', '-s', default='random_pair',
                        choices=['random_pair', 'cold_drug', 'cold_protein', 'double_cold'],
                        help='Split strategy')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for all splits and models')
    parser.add_argument('--test-frac', type=float, default=0.2,
                        help='Fraction of data/entities for test set')
    parser.add_argument('--calibration-frac', type=float, default=0.1,
                        help='Fraction of data/entities for calibration set')
    parser.add_argument('--use-wandb', default='false',
                        help='Enable wandb logging (true/false)')
    parser.add_argument('--skip-experts', action='store_true',
                        help='Skip expert model training (requires precomputed predictions)')
    parser.add_argument('--skip-moe', action='store_true',
                        help='Skip MoE (requires precomputed moe_predictions.csv)')
    parser.add_argument('--skip-icp', action='store_true',
                        help='Skip ICP (requires precomputed icp_results.csv)')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation step')
    args = parser.parse_args()

    python = sys.executable

    print(f"\n{'='*70}")
    print(f"EnsDTI Leakage-Controlled Pipeline")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Split type   : {args.split_type}")
    print(f"  Seed         : {args.seed}")
    print(f"  Test frac    : {args.test_frac}")
    print(f"  Cal frac     : {args.calibration_frac}")
    print(f"{'='*70}")

    # Create output directories
    os.makedirs(os.path.join('tmp', args.dataset, args.split_type), exist_ok=True)
    os.makedirs(os.path.join('output', args.dataset, args.split_type), exist_ok=True)

    # -----------------------------------------------------------------
    # STAGE 1+2: Split data and save manifest
    # -----------------------------------------------------------------
    run_step(
        "Split data & save manifest",
        [python, 'generate_split.py',
         '--dataset', args.dataset,
         '--split-type', args.split_type,
         '--seed', str(args.seed),
         '--test-frac', str(args.test_frac),
         '--calibration-frac', str(args.calibration_frac)]
    )

    # -----------------------------------------------------------------
    # STAGE 3: Leakage audit
    # -----------------------------------------------------------------
    run_step(
        "Leakage audit",
        [python, 'audit_splits.py',
         '--dataset', args.dataset,
         '--split-types', args.split_type]
    )

    # -----------------------------------------------------------------
    # STAGES 4+5: Train expert models and generate predictions
    # -----------------------------------------------------------------
    if not args.skip_experts:
        run_step(
            "Expert model training & prediction",
            [python, 'expert_predict.py',
             '--dataset', args.dataset,
             '--split-type', args.split_type,
             '--seed', str(args.seed)]
        )
    else:
        print("\n[SKIP] Expert prediction step (--skip-experts)")

    # -----------------------------------------------------------------
    # STAGES 6+7: MoE training and prediction
    # -----------------------------------------------------------------
    if not args.skip_moe:
        run_step(
            "MoE gating training & prediction",
            [python, 'moe_predict_v2.py',
             '--dataset', args.dataset,
             '--split-type', args.split_type,
             '--seed', str(args.seed)]
        )
    else:
        print("\n[SKIP] MoE step (--skip-moe)")

    # -----------------------------------------------------------------
    # STAGES 8+9: ICP calibration and test prediction
    # -----------------------------------------------------------------
    if not args.skip_icp:
        run_step(
            "ICP calibration & prediction",
            [python, 'icp_predict_v2.py',
             '--dataset', args.dataset,
             '--split-type', args.split_type]
        )
    else:
        print("\n[SKIP] ICP step (--skip-icp)")

    # -----------------------------------------------------------------
    # STAGE 10: Evaluation
    # -----------------------------------------------------------------
    if not args.skip_eval:
        run_step(
            "Evaluation (metrics + plots)",
            [python, 'evaluate_results.py',
             '--dataset', args.dataset,
             '--split-type', args.split_type]
        )
    else:
        print("\n[SKIP] Evaluation step (--skip-eval)")

    # -----------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"[OK] Pipeline complete: {args.dataset} / {args.split_type}")
    print(f"   Results in: output/{args.dataset}/{args.split_type}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
