"""
generate_split.py — Load dataset and create leakage-controlled 3-way split.

Reads dataset/<dataset>/data.csv, applies the requested split strategy,
and saves the split manifest to output/<dataset>/<split_type>/split_manifest.csv.

Called by run_pipeline.py — can also be run standalone.
"""

import os
import sys
import argparse
import pandas as pd

# Add EnsDTI root to path so we can import data.data_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.data_split import (
    split_random_pair, split_cold_drug, split_cold_protein,
    split_double_cold, save_split_manifest
)


SPLIT_FNS = {
    'random_pair': split_random_pair,
    'cold_drug': split_cold_drug,
    'cold_protein': split_cold_protein,
    'double_cold': split_double_cold,
}


def main():
    parser = argparse.ArgumentParser(description='Generate split manifest')
    parser.add_argument('--dataset', '-d', default='davis')
    parser.add_argument('--split-type', '-s', default='random_pair',
                        choices=list(SPLIT_FNS.keys()))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-frac', type=float, default=0.2)
    parser.add_argument('--calibration-frac', type=float, default=0.1)
    parser.add_argument('--drug-col', default='lig')
    parser.add_argument('--protein-col', default='pro')
    parser.add_argument('--label-col', default='lab')
    args = parser.parse_args()

    data_path = os.path.join('dataset', args.dataset, 'data.csv')
    if not os.path.exists(data_path):
        sys.exit(f"Dataset not found: {data_path}")

    print(f"\nLoading {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Rows: {len(df)}, Columns: {df.columns.tolist()}")
    print(f"  Unique drugs: {df[args.drug_col].nunique()}")
    print(f"  Unique proteins: {df[args.protein_col].nunique()}")
    print(f"  Label distribution: {df[args.label_col].value_counts().to_dict()}")

    print(f"\nApplying split: {args.split_type} (seed={args.seed}, "
          f"test_frac={args.test_frac}, cal_frac={args.calibration_frac})")

    split_fn = SPLIT_FNS[args.split_type]

    common_kwargs = dict(test_frac=args.test_frac,
                         calibration_frac=args.calibration_frac,
                         seed=args.seed)

    if args.split_type == 'random_pair':
        train_df, cal_df, test_df = split_fn(df, **common_kwargs)
    elif args.split_type == 'cold_drug':
        train_df, cal_df, test_df = split_fn(df, drug_col=args.drug_col, **common_kwargs)
    elif args.split_type == 'cold_protein':
        train_df, cal_df, test_df = split_fn(df, protein_col=args.protein_col, **common_kwargs)
    elif args.split_type == 'double_cold':
        train_df, cal_df, test_df = split_fn(
            df, drug_col=args.drug_col, protein_col=args.protein_col, **common_kwargs)

    print(f"\nSplit sizes:")
    print(f"  Train      : {len(train_df)} rows")
    print(f"  Calibration: {len(cal_df)} rows")
    print(f"  Test       : {len(test_df)} rows")
    print(f"  Total      : {len(train_df)+len(cal_df)+len(test_df)} / {len(df)} original")

    manifest_path = save_split_manifest(
        train_df, cal_df, test_df,
        dataset=args.dataset,
        split_type=args.split_type,
        drug_col=args.drug_col,
        protein_col=args.protein_col,
        label_col=args.label_col,
    )

    # Save split data for later use
    out_dir = os.path.join('output', args.dataset, args.split_type)
    train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=True)
    cal_df.to_csv(os.path.join(out_dir, 'calibration.csv'), index=True)
    test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=True)
    print(f"\nSplit CSVs saved to {out_dir}/")


if __name__ == '__main__':
    main()
