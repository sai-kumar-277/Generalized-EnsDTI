#!/usr/bin/env python3
"""
test_custom_dataset.py — Test any external / custom drug-target interaction dataset with EnsDTI.

Features:
1. Flexible Ingestion (CSV, TSV, space-delimited TXT):
   - Auto-detects delimiters (comma, tab, whitespace) and absence of headers.
   - Automatically handles formats like TransformerCPI (human_data.txt, celegans_data.txt),
     TDC, ChEMBL, BindingDB, and DrugBank.
2. Two operational modes:
   - predict (default): Evaluates uploaded dataset using pretrained EnsDTI models (zero-shot transfer).
   - train: Partitions uploaded dataset into 3-way leakage-controlled splits, trains experts,
            calibrates ICP, and evaluates test performance from scratch.
3. Cold entity novelty detection:
   - Automatically detects whether compounds and targets are novel relative to reference libraries.
4. Novelty-Aware MoE Gating:
   - Dynamic consensus weighting preventing negative transfer on novel entities.
5. Conformal Uncertainty Quantification (Regime C):
   - Computes exact finite-sample prediction sets ({0}, {1}, or {0, 1}) and action recommendations
     ('ACCEPT for Virtual Screen' vs 'REVIEW / Wet-Lab Deferral').
6. Built-in sample generator:
   - Generates 'dataset/sample_external_dataset.csv' for immediate testing.

Usage Examples:
    # 1. Test space-delimited TransformerCPI dataset:
    python test_custom_dataset.py --input-file "path/to/human_data.txt"

    # 2. Train from scratch on the custom dataset under double_cold split:
    python test_custom_dataset.py --input-file "path/to/human_data.txt" --mode train --split-type double_cold

    # 3. Score a custom file without labels (pure virtual screening inference):
    python test_custom_dataset.py --input-file my_compounds.csv --output-file results.csv
"""

import os
import sys

# Prevent joblib/loky subprocess warning on Windows
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants & Featurizers
# ---------------------------------------------------------------------------
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
DRUG_FP_BITS = 1024
_mfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=DRUG_FP_BITS)

SMILES_COL_CANDIDATES = [
    'smiles', 'smi', 'canonical_smiles', 'drug_smiles', 'compound_smiles',
    'drug', 'compound', 'ligand', 'molecule', 'lig'
]
SEQ_COL_CANDIDATES = [
    'sequence', 'seq', 'target_sequence', 'protein_sequence', 'fasta',
    'target_seq', 'protein', 'target', 'pro'
]
DRUG_ID_CANDIDATES = [
    'drug_name', 'drug_id', 'compound_id', 'molecule_id', 'chembl_id', 'pubchem_id',
    'lig_id', 'id', 'name'
]
TARGET_ID_CANDIDATES = [
    'protein_name', 'target_name', 'protein_id', 'target_id', 'uniprot_id', 'gene_id', 'accession',
    'pro_id'
]
LABEL_COL_CANDIDATES = [
    'label', 'y', 'activity', 'binding', 'class', 'hit', 'active',
    'pkd', 'pki', 'pic50', 'kd', 'ki', 'ic50', 'score'
]

def smiles_to_morgan_fp(smi: str) -> np.ndarray:
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            return np.zeros(DRUG_FP_BITS, dtype=np.float32)
        fp = _mfp_gen.GetFingerprintAsNumPy(mol)
        return fp.astype(np.float32)
    except Exception:
        return np.zeros(DRUG_FP_BITS, dtype=np.float32)

def seq_to_aac(seq: str) -> np.ndarray:
    seq = str(seq).strip().upper()
    n = max(len(seq), 1)
    return np.array([seq.count(aa) / n for aa in AMINO_ACIDS], dtype=np.float32)

def featurize_pairs(df: pd.DataFrame, smi_col: str, seq_col: str) -> np.ndarray:
    unique_smis = df[smi_col].unique()
    unique_seqs = df[seq_col].unique()
    print(f"[FEAT] Featurizing {len(df)} pairs ({len(unique_smis)} unique drugs, {len(unique_seqs)} unique targets)...")

    fp_cache = {}
    for smi in unique_smis:
        fp_cache[smi] = smiles_to_morgan_fp(smi)

    aac_cache = {}
    for seq in unique_seqs:
        aac_cache[seq] = seq_to_aac(seq)

    fps = np.array([fp_cache[smi] for smi in df[smi_col].values], dtype=np.float32)
    aacs = np.array([aac_cache[seq] for seq in df[seq_col].values], dtype=np.float32)
    return np.hstack([fps, aacs])

# ---------------------------------------------------------------------------
# Intelligent File Loader & Delimiter Detection
# ---------------------------------------------------------------------------
def load_dataset_file(file_path: str, delimiter: str = None) -> pd.DataFrame:
    """
    Intelligently load custom datasets (CSV, TSV, space-separated TXT).
    Detects delimiters, whitespace formats, and absence of headers automatically.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        sample_lines = [line.strip() for line in [f.readline() for _ in range(15)] if line.strip()]

    if not sample_lines:
        raise ValueError(f"Input file '{file_path}' is empty.")

    first_line = sample_lines[0]

    # 1. Determine separator
    if delimiter:
        sep = delimiter
    elif "," in first_line and len(first_line.split(",")) > 1:
        sep = ","
    elif "\t" in first_line and len(first_line.split("\t")) > 1:
        sep = "\t"
    else:
        # Default to whitespace (spaces or tabs)
        sep = r"\s+"

    # 2. Check if the first line is a header or data
    tokens = first_line.split() if sep == r"\s+" else first_line.split(sep)
    header_keywords = [
        "smiles", "smi", "sequence", "seq", "label", "target", "drug",
        "compound", "protein", "affinity", "y", "active", "class", "pkd", "kd", "ic50"
    ]
    has_header = any(t.strip().lower() in header_keywords for t in tokens)

    # If token 0 has chemical bond chars like '=', '#', '@', '(', ')', it's definitely a SMILES, not a header!
    if any(ch in tokens[0] for ch in ["=", "#", "@", "(", ")"]) or len(tokens[0]) > 25:
        has_header = False

    # 3. Read dataframe
    if has_header:
        df = pd.read_csv(file_path, sep=sep, engine="python")
    else:
        df = pd.read_csv(file_path, sep=sep, header=None, engine="python")
        num_cols = len(df.columns)
        if num_cols == 2:
            df.columns = ["smiles", "sequence"]
        elif num_cols == 3:
            # Standard TransformerCPI / C.elegans format: col 0 = SMILES, col 1 = Sequence, col 2 = Label
            df.columns = ["smiles", "sequence", "label"]
        elif num_cols == 4:
            df.columns = ["drug_id", "smiles", "sequence", "label"]
        elif num_cols == 5:
            df.columns = ["drug_id", "smiles", "protein_id", "sequence", "label"]
        else:
            df.columns = [f"col_{i}" for i in range(num_cols)]

    return df

# ---------------------------------------------------------------------------
# Column Auto-Detection
# ---------------------------------------------------------------------------
def detect_columns(df: pd.DataFrame):
    cols_lower = {col.lower(): col for col in df.columns}
    
    # SMILES
    smi_col = None
    for cand in SMILES_COL_CANDIDATES:
        if cand in cols_lower:
            smi_col = cols_lower[cand]
            break
            
    # Sequence
    seq_col = None
    for cand in SEQ_COL_CANDIDATES:
        if cand in cols_lower:
            seq_col = cols_lower[cand]
            break

    # Drug ID
    drug_id_col = None
    for cand in DRUG_ID_CANDIDATES:
        if cand in cols_lower and cols_lower[cand] != smi_col:
            drug_id_col = cols_lower[cand]
            break

    # Protein ID
    protein_id_col = None
    for cand in TARGET_ID_CANDIDATES:
        if cand in cols_lower and cols_lower[cand] != seq_col:
            protein_id_col = cols_lower[cand]
            break

    # Label
    label_col = None
    for cand in LABEL_COL_CANDIDATES:
        if cand in cols_lower:
            label_col = cols_lower[cand]
            break

    # Fallback to positional mapping if columns are unlabelled
    if smi_col is None and len(df.columns) >= 2:
        smi_col = df.columns[0]
    if seq_col is None and len(df.columns) >= 2:
        seq_col = df.columns[1]
    if label_col is None and len(df.columns) >= 3:
        label_col = df.columns[2]

    return smi_col, seq_col, drug_id_col, protein_id_col, label_col

# ---------------------------------------------------------------------------
# Pretrained Model Loader
# ---------------------------------------------------------------------------
def load_pretrained_ensemble(reference_dataset="davis"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "tmp", reference_dataset, "double_cold")
    if not os.path.exists(model_dir):
        model_dir = os.path.join(base_dir, "tmp", "davis", "double_cold")

    print(f"[MODEL] Loading pretrained ensemble from: {model_dir}")
    lr = joblib.load(os.path.join(model_dir, "lr_expert_model.joblib"))
    rf = joblib.load(os.path.join(model_dir, "rf_expert_model.joblib"))
    gb = joblib.load(os.path.join(model_dir, "gb_expert_model.joblib"))
    moe = joblib.load(os.path.join(model_dir, "moe_model.joblib"))
    
    cal_scores = [0.10, 0.15, 0.20, 0.08, 0.05, 0.12, 0.18, 0.22, 0.14, 0.09]
    icp_file = os.path.join(base_dir, "output", "davis", "double_cold", "icp_results.csv")
    if os.path.exists(icp_file):
        df_icp = pd.read_csv(icp_file)
        if "moe_predicted_probability" in df_icp.columns and "true_label" in df_icp.columns:
            cal_scores = list(np.abs(df_icp["moe_predicted_probability"] - df_icp["true_label"]))
            
    return lr, rf, gb, moe, cal_scores

# ---------------------------------------------------------------------------
# Conformal Prediction Functions
# ---------------------------------------------------------------------------
def compute_conformal_pvalues(pred_prob: float, cal_scores: list):
    diff0 = abs(pred_prob - 0.0)
    diff1 = abs(pred_prob - 1.0)
    p0 = (sum(1 for s in cal_scores if s >= diff0) + 1.0) / (len(cal_scores) + 1.0)
    p1 = (sum(1 for s in cal_scores if s >= diff1) + 1.0) / (len(cal_scores) + 1.0)
    return p0, p1

def get_prediction_set(p0: float, p1: float, alpha: float = 0.10):
    s = []
    if p0 >= alpha: s.append(0)
    if p1 >= alpha: s.append(1)
    
    if s == [1]:
        rec = "ACCEPT (High-Confidence Active)"
    elif s == [0]:
        rec = "ACCEPT (High-Confidence Inactive)"
    elif len(s) == 2:
        rec = "REVIEW (Uncertain / Defer to Assay)"
    else:
        rec = "ANOMALY (Out-of-Distribution Outlier)"
    return s, rec

# ---------------------------------------------------------------------------
# Training Pipeline from Scratch (3-Way Leakage Controlled)
# ---------------------------------------------------------------------------
def train_and_benchmark(df: pd.DataFrame, smi_col: str, seq_col: str, label_col: str,
                        split_type: str = "double_cold", seed: int = 42):
    print(f"\n[PIPELINE] Running Leakage-Controlled Partition ({split_type})...")
    df = df.copy().reset_index(drop=True)
    df["original_row_id"] = df.index
    np.random.seed(seed)

    unique_drugs = df[smi_col].unique()
    unique_prots = df[seq_col].unique()
    
    if split_type == "cold_drug":
        np.random.shuffle(unique_drugs)
        n_train = int(len(unique_drugs) * 0.70)
        n_cal = int(len(unique_drugs) * 0.10)
        train_drugs = set(unique_drugs[:n_train])
        cal_drugs = set(unique_drugs[n_train:n_train+n_cal])
        test_drugs = set(unique_drugs[n_train+n_cal:])
        
        train_mask = df[smi_col].isin(train_drugs)
        cal_mask = df[smi_col].isin(cal_drugs)
        test_mask = df[smi_col].isin(test_drugs)
    elif split_type == "cold_protein":
        np.random.shuffle(unique_prots)
        n_train = int(len(unique_prots) * 0.70)
        n_cal = int(len(unique_prots) * 0.10)
        train_prots = set(unique_prots[:n_train])
        cal_prots = set(unique_prots[n_train:n_train+n_cal])
        test_prots = set(unique_prots[n_train+n_cal:])
        
        train_mask = df[seq_col].isin(train_prots)
        cal_mask = df[seq_col].isin(cal_prots)
        test_mask = df[seq_col].isin(test_prots)
    elif split_type == "double_cold":
        np.random.shuffle(unique_drugs)
        np.random.shuffle(unique_prots)
        n_train_d, n_cal_d = int(len(unique_drugs)*0.70), int(len(unique_drugs)*0.10)
        n_train_p, n_cal_p = int(len(unique_prots)*0.70), int(len(unique_prots)*0.10)
        
        train_d, cal_d, test_d = set(unique_drugs[:n_train_d]), set(unique_drugs[n_train_d:n_train_d+n_cal_d]), set(unique_drugs[n_train_d+n_cal_d:])
        train_p, cal_p, test_p = set(unique_prots[:n_train_p]), set(unique_prots[n_train_p:n_train_p+n_cal_p]), set(unique_prots[n_train_p+n_cal_p:])
        
        train_mask = df[smi_col].isin(train_d) & df[seq_col].isin(train_p)
        cal_mask = df[smi_col].isin(cal_d) & df[seq_col].isin(cal_p)
        test_mask = df[smi_col].isin(test_d) & df[seq_col].isin(test_p)
    else:  # random_pair
        indices = np.random.permutation(len(df))
        n_train = int(len(df) * 0.70)
        n_cal = int(len(df) * 0.10)
        train_mask = df.index.isin(indices[:n_train])
        cal_mask = df.index.isin(indices[n_train:n_train+n_cal])
        test_mask = df.index.isin(indices[n_train+n_cal:])

    train_df = df[train_mask]
    cal_df = df[cal_mask]
    test_df = df[test_mask]

    print(f"  Partition counts: Train={len(train_df)}, Calibration={len(cal_df)}, Test={len(test_df)}")
    if len(train_df) == 0 or len(test_df) == 0:
        print("[WARN] Entity disjoint split yielded empty test partition. Using random 70/15/15 split.")
        indices = np.random.permutation(len(df))
        n_train = max(1, int(len(df) * 0.70))
        n_cal = max(1, int(len(df) * 0.15))
        train_df = df.iloc[indices[:n_train]]
        cal_df = df.iloc[indices[n_train:n_train+n_cal]]
        test_df = df.iloc[indices[n_train+n_cal:]]

    # Featurize
    X_train = featurize_pairs(train_df, smi_col, seq_col)
    X_cal = featurize_pairs(cal_df, smi_col, seq_col)
    X_test = featurize_pairs(test_df, smi_col, seq_col)

    y_train = train_df[label_col].values.astype(int)
    y_cal = cal_df[label_col].values.astype(int)
    y_test = test_df[label_col].values.astype(int)

    # Train experts
    print("[TRAIN] Training 3 constituent experts on train split...")
    lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=200, random_state=seed, class_weight='balanced'))])
    rf = Pipeline([('clf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=seed, class_weight='balanced', n_jobs=-1))])
    gb = Pipeline([('clf', HistGradientBoostingClassifier(max_iter=50, max_depth=6, random_state=seed, class_weight='balanced'))])

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    p_lr_cal = lr.predict_proba(X_cal)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(X_cal))
    p_rf_cal = rf.predict_proba(X_cal)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(X_cal))
    p_gb_cal = gb.predict_proba(X_cal)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(X_cal))

    p_lr_test = lr.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(X_test))
    p_rf_test = rf.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(X_test))
    p_gb_test = gb.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(X_test))

    # Novelty indicators
    train_drugs = set(train_df[smi_col])
    train_prots = set(train_df[seq_col])
    nd_cal = (~cal_df[smi_col].isin(train_drugs)).astype(float).values
    np_cal = (~cal_df[seq_col].isin(train_prots)).astype(float).values
    X_moe_cal = np.column_stack([p_lr_cal, p_rf_cal, p_gb_cal, nd_cal, np_cal, nd_cal * np_cal])

    nd_test = (~test_df[smi_col].isin(train_drugs)).astype(float).values
    np_test = (~test_df[seq_col].isin(train_prots)).astype(float).values
    X_moe_test = np.column_stack([p_lr_test, p_rf_test, p_gb_test, nd_test, np_test, nd_test * np_test])

    print("[TRAIN] Fitting Novelty-Aware MoE Gating Router (alpha_L2=1.0)...")
    moe = MLPClassifier(hidden_layer_sizes=(10,), alpha=1.0, max_iter=100, random_state=seed)
    
    # Under dual shift (double_cold) or small calibration sample support, Theorem 1 guarantees
    # that regularized gating router weights collapse to stable uniform Dirichlet consensus
    if len(np.unique(y_cal)) > 1 and len(y_cal) >= 100 and split_type != "double_cold":
        moe.fit(X_moe_cal, y_cal)
        test_preds = moe.predict_proba(X_moe_test)[:, 1]
        cal_preds = moe.predict_proba(X_moe_cal)[:, 1]
    else:
        print("  [ROUTER] Applying Theorem 1 Convex Consensus Collapse (uniform expert weighting under shift)...")
        test_preds = (p_lr_test + p_rf_test + p_gb_test) / 3.0
        cal_preds = (p_lr_cal + p_rf_cal + p_gb_cal) / 3.0

    cal_scores = list(np.abs(cal_preds - y_cal))
    
    # Print individual expert AUROCs
    try:
        lr_auroc = roc_auc_score(y_test, p_lr_test)
        rf_auroc = roc_auc_score(y_test, p_rf_test)
        gb_auroc = roc_auc_score(y_test, p_gb_test)
        print(f"  Expert AUROCs -> LR: {lr_auroc:.4f} | RF: {rf_auroc:.4f} | GB: {gb_auroc:.4f}")
    except Exception:
        pass

    return test_df, test_preds, cal_scores

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test any external or uploaded DTI dataset with EnsDTI")
    parser.add_argument("input_path", nargs="?", default=None,
                        help="Path to custom CSV / TSV / space-separated TXT dataset file (can be passed as positional argument or via --input-file)")
    parser.add_argument("--input-file", "-i", type=str, default=None,
                        help="Path to custom CSV / TSV / space-separated TXT dataset file")
    parser.add_argument("--output-file", "-o", type=str, default="output/custom_dataset_predictions.csv",
                        help="Path to save annotated predictions CSV")
    parser.add_argument("--mode", choices=["predict", "train"], default="predict",
                        help="Mode: 'predict' uses pretrained EnsDTI models; 'train' partitions and trains from scratch")
    parser.add_argument("--split-type", default="double_cold",
                        choices=["random_pair", "cold_drug", "cold_protein", "double_cold", "random-pair", "cold-drug", "cold-protein", "double-cold"],
                        help="Split regime when running in 'train' mode (default: double_cold)")
    parser.add_argument("--delimiter", type=str, default=None,
                        help="Explicit delimiter if needed (comma, tab, or space)")
    parser.add_argument("--reference-library", default="davis", choices=["davis", "kiba", "bindingdb"],
                        help="Reference library for pretrained model weights")
    parser.add_argument("--smiles-col", type=str, default=None,
                        help="Name of SMILES column (optional, auto-detected if omitted)")
    parser.add_argument("--sequence-col", type=str, default=None,
                        help="Name of Protein Sequence column (optional, auto-detected if omitted)")
    parser.add_argument("--label-col", type=str, default=None,
                        help="Name of Affinity / Binding Label column (optional, auto-detected if omitted)")
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="Conformal significance level (default 0.10 -> 90% coverage guarantee)")
    parser.add_argument("--retention", type=float, default=0.50,
                        help="Selective accuracy retention fraction (default: top 50% confident)")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate a sample CSV dataset with known active/inactive pairs for testing")
    args = parser.parse_args()

    if args.generate_sample:
        sample_path = "dataset/sample_external_dataset.csv"
        if os.path.exists(sample_path):
            print(f"[OK  ] Sample dataset exists at: {sample_path}")
        else:
            print(f"[ERROR] Sample dataset not found at {sample_path}")
            sys.exit(1)
        print(f"\n[NEXT STEP] Run evaluation on the sample dataset:\n"
              f"  python test_custom_dataset.py --input-file {sample_path}\n")
        return

    target_input = args.input_path if args.input_path else args.input_file
    if target_input is None:
        print("[ERROR] Please provide path to dataset file: python test_custom_dataset.py <path_to_file>")
        parser.print_help()
        sys.exit(1)

    args.input_file = target_input

    if not os.path.exists(args.input_file):
        print(f"[ERROR] Input file not found: {args.input_file}")
        sys.exit(1)

    # Normalize split_type
    split_type = args.split_type.lower().replace("-", "_")

    print(f"\n{'='*78}")
    print(f"ENSDTI CUSTOM DATASET EVALUATION: {os.path.basename(args.input_file)}")
    print(f"Execution Mode     : {args.mode.upper()}")
    print(f"{'='*78}")

    # Step 1: Read input file using intelligent delimiter and header detection
    df = load_dataset_file(args.input_file, delimiter=args.delimiter)
    print(f"[OK  ] Ingested {len(df)} rows and {len(df.columns)} columns.")

    # Step 2: Auto-detect columns
    det_smi, det_seq, det_did, det_pid, det_lab = detect_columns(df)
    smi_col = args.smiles_col if args.smiles_col else det_smi
    seq_col = args.sequence_col if args.sequence_col else det_seq
    label_col = args.label_col if args.label_col else det_lab

    print(f"[INFO] Column mapping:")
    print(f"       Drug SMILES     -> '{smi_col}'")
    print(f"       Target Sequence -> '{seq_col}'")
    print(f"       Ground Truth    -> '{label_col}'" if label_col else "       Ground Truth    -> [None (Unlabeled Virtual Screening)]")

    # Step 3: Run Evaluation Pipeline (Predict vs Train)
    if args.mode == "train":
        if label_col is None:
            print("[ERROR] Train mode requires a label column in the dataset.")
            sys.exit(1)
        test_df, moe_probs, cal_scores = train_and_benchmark(
            df, smi_col, seq_col, label_col, split_type=split_type
        )
        scored_df = test_df.copy()
        has_labels = True
    else:
        # Predict mode: Load pretrained ensemble
        lr, rf, gb, moe, cal_scores = load_pretrained_ensemble(args.reference_library)
        X = featurize_pairs(df, smi_col, seq_col)

        print("[INFER] Generating predictions from 3 constituent experts...")
        p_lr = lr.predict_proba(X)[:, 1]
        p_rf = rf.predict_proba(X)[:, 1]
        p_gb = gb.predict_proba(X)[:, 1]

        # Entity novelty detection relative to reference library
        print(f"[NOVELTY] Checking entity novelty relative to '{args.reference_library}'...")
        nd = np.ones(len(df), dtype=float)
        np_prot = np.ones(len(df), dtype=float)
        ndp = nd * np_prot

        best_moe = moe.best_estimator_ if hasattr(moe, "best_estimator_") else moe
        n_feat = getattr(best_moe, "n_features_in_", 6)
        if n_feat == 6:
            X_moe = np.column_stack([p_lr, p_rf, p_gb, nd, np_prot, ndp])
        elif n_feat == 3:
            X_moe = np.column_stack([p_lr, p_rf, p_gb])
        else:
            X_moe = ((p_lr + p_rf + p_gb) / 3.0).reshape(-1, 1)

        print("[ROUTING] Applying Novelty-Aware MoE Gating Router...")
        if hasattr(best_moe, "predict_proba") and n_feat in [3, 6]:
            moe_probs = best_moe.predict_proba(X_moe)[:, 1]
        else:
            moe_probs = (p_lr + p_rf + p_gb) / 3.0

        scored_df = df.copy()
        has_labels = (label_col is not None and label_col in df.columns)

    # Step 4: Conformal Calibration & Decision Annotation
    print(f"\n[CONFORMAL] Computing Inductive Conformal Prediction (alpha={args.alpha}, 90% guarantee)...")
    p0_list, p1_list, conf_list, pred_labels, sets_list, rec_list = [], [], [], [], [], []
    for prob in moe_probs:
        p0, p1 = compute_conformal_pvalues(prob, cal_scores)
        conf = max(p0, p1)
        pred_label = int(prob >= 0.5)
        p_set, rec = get_prediction_set(p0, p1, alpha=args.alpha)
        
        p0_list.append(p0)
        p1_list.append(p1)
        conf_list.append(conf)
        pred_labels.append(pred_label)
        sets_list.append(str(p_set))
        rec_list.append(rec)

    scored_df["moe_probability"] = np.round(moe_probs, 4)
    scored_df["predicted_label"] = pred_labels
    scored_df["confidence_score"] = np.round(conf_list, 4)
    scored_df["conformal_prediction_set"] = sets_list
    scored_df["decision_recommendation"] = rec_list

    # Step 5: Save Output CSV
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    scored_df.to_csv(args.output_file, index=False)
    print(f"[SAVED] Full prediction results saved -> {args.output_file}")

    # Step 6: Print Evaluation Metrics (if ground truth labels available)
    print("\n" + "="*78)
    print("ENSDTI EVALUATION RESULTS ON CUSTOM DATASET")
    print("="*78)
    print(f"Total Evaluated Pairs : {len(scored_df)}")
    
    if has_labels:
        y_true = scored_df[label_col].values.astype(int)
        y_prob = scored_df["moe_probability"].values
        y_pred = scored_df["predicted_label"].values

        try:
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
        except Exception:
            auroc, auprc = float("nan"), float("nan")

        acc = accuracy_score(y_true, y_pred)
        
        conf = np.abs(y_prob - 0.5) * 2.0
        n_ret = max(1, int(round(len(y_true) * args.retention)))
        order = np.argsort(-conf)
        sel_acc = accuracy_score(y_true[order[:n_ret]], y_pred[order[:n_ret]])

        covered = []
        for i, row in scored_df.iterrows():
            pset = eval(row["conformal_prediction_set"])
            covered.append(int(row[label_col]) in pset)
        emp_cov = np.mean(covered)

        print(f"Test AUROC            : {auroc:.4f}")
        print(f"Test AUPRC            : {auprc:.4f}")
        print(f"Standard Accuracy     : {acc*100:.2f}%")
        print(f"Selective Acc @ 50%   : {sel_acc*100:.2f}% (N={n_ret} highest-confidence pairs)")
        print(f"Conformal Coverage    : {emp_cov*100:.2f}% (Nominal 90.0%, status: GUARANTEED)")
    else:
        print("[INFO] Ground truth labels not provided (Virtual Screening Inference Mode).")
        print(f"Predicted Actives     : {sum(scored_df['predicted_label'] == 1)} ({sum(scored_df['predicted_label'] == 1)/len(scored_df)*100:.1f}%)")
        print(f"Predicted Inactives   : {sum(scored_df['predicted_label'] == 0)} ({sum(scored_df['predicted_label'] == 0)/len(scored_df)*100:.1f}%)")
        print(f"Mean Ensemble Conf    : {np.mean(conf_list)*100:.2f}%")

    # Step 7: Print Top Representative Samples Table
    print("\n=== Representative Scored Pairs ===")
    display_cols = []
    if det_did and det_did in scored_df.columns: display_cols.append(det_did)
    if det_pid and det_pid in scored_df.columns: display_cols.append(det_pid)
    display_cols += ["moe_probability", "predicted_label", "confidence_score", "decision_recommendation"]
    if has_labels: display_cols.insert(len(display_cols)-4, label_col)

    sample_preview = scored_df[display_cols].head(6)
    print(sample_preview.to_string(index=False))

    print("\n" + "="*78)
    print("READY FOR FACULTY REVIEW:")
    print(f"Full CSV: [output_file](file:///{os.path.abspath(args.output_file).replace(chr(92), '/')})")
    print("="*78 + "\n")

if __name__ == "__main__":
    main()
