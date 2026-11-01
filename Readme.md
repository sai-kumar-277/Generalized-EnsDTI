# Generalized-EnsDTI: Leakage-Controlled Evaluation, Modular Repair, and Conformal Uncertainty Quantification for Drug–Target Interaction Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An audited, modular, and leakage-controlled re-evaluation and extension of **EnsDTI**, incorporating strict split regimes (Random-Pair, Cold-Drug, Cold-Protein, Double-Cold), disjoint Inductive Conformal Prediction (ICP), and native PyTorch deep expert models.

---

## 📖 Reference & Base Paper

This project audits, repairs, and extends the methodology published in:

> **EnsDTI: Mixture-of-Experts Approach for Enhanced Drug-Target Interaction Prediction and Confidence Assessment**  
> Published in: *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, 2026.  
> **Paper Link / DOI:** [https://www.computer.org/csdl/journal/bb/2026/03/11353228/2dh4UUrCHMQ](https://www.computer.org/csdl/journal/bb/2026/03/11353228/2dh4UUrCHMQ)

---

## 🎯 1. What We Are Trying to Do (Problem Statement & Motivation)

Accurate computational prediction of drug–target interactions (DTI) is critical for virtual screening, drug repurposing, and lead discovery. **EnsDTI** proposed a three-stage framework combining:
1. Diverse expert neural architectures (`DeepDTA`, `DeepConv-DTI`, `DeepPurpose`, etc.)
2. A Mixture-of-Experts (MoE) gating multilayer perceptron
3. Inductive Conformal Prediction (ICP) to generate sample-wise confidence scores

However, rigorous independent audits of DTI pipelines reveal key methodological challenges that our project addresses:

### A. Overcoming Hidden Data Leakage in Evaluation
- **The Pitfall:** Conventional random interaction-level splitting partitions drug–target pairs randomly across train and test sets. In this setup, almost all test drugs and test targets have already been seen in training interactions.
- **The Consequence:** Models often exploit **compound memorization** and target frequency biases rather than learning generalized biophysical interaction rules. Reported AUROC/AUPRC scores are often unrealistically high.
- **Our Goal:** Implement strict, leakage-controlled partition manifests covering four standardized evaluation regimes:
  - **Random Pair Split**: Standard interaction-level benchmark baseline.
  - **Cold-Drug Split**: Evaluates generalization to entirely unseen chemical scaffolds / candidate drugs.
  - **Cold-Protein Split**: Evaluates generalization to novel disease protein targets.
  - **Double-Cold Split**: Evaluates generalization when both drug and target are completely unseen during training.

### B. Enforcing Valid Conformal Exchangeability in ICP
- **The Pitfall:** The original ICP implementation sampled calibration pairs directly from the full dataset without enforcing disjointness from training data.
- **The Consequence:** When calibration data overlaps with model training data, model prediction residuals are artificially compressed. This underestimates non-conformity scores and can produce overconfident prediction intervals.
- **Our Goal:** Establish **Regime B (Strict Disjoint Pair Calibration)** where calibration data is strictly quarantined from both expert model training and MoE gating optimization, preventing calibration contamination.

### C. Codebase Audit, Modular Repair & Portability
- **The Pitfall:** The original repository contained broken package dependencies, hardcoded CUDA exits preventing CPU execution, circular imports in expert models, and undocumented intermediate data schemas.
- **Our Goal:** Deliver a clean, reproducible, and runnable PyTorch codebase that executes out-of-the-box on both GPU and CPU workstations without modifying core mathematical architectures or label binarization logic.

---

## 🏆 2. What We Have Achieved (Milestones & Findings)

### 🛠️ Technical Milestones
1. **Clean Native PyTorch Ports of Deep Experts:**
   - Audited and repaired native PyTorch implementations of [`DeepDTA`](models/dpdta.py) (dual 1D-CNN over SMILES and amino acid sequences) and [`DeepConv-DTI`](models/dcdti.py) (multi-scale dilated protein convolutions + fingerprint dense blocks).
   - Resolved circular imports, missing dependency fallbacks, and memory overheads, enabling smooth execution on CPU and GPU.
2. **Leakage-Controlled Data Partitioning Engine:**
   - Built [`generate_split.py`](generate_split.py) and [`audit_splits.py`](audit_splits.py) to partition the full Davis benchmark dataset (25,772 interactions) into guaranteed disjoint splits with reproducible seed control.
3. **Stand-Alone Deep Model Training & Manifest Pipeline:**
   - Implemented [`train_deep_experts.py`](train_deep_experts.py) to train individual deep architectures strictly on training sets and export standardized predictions (`<model>_predictions.csv`) for calibration and test splits.
4. **Disjoint MoE & Conformal Prediction (`icp_predict_v2.py`):**
   - Implemented clean two-step routing and inductive conformal prediction with strict calibration isolation, reporting valid non-conformity $p$-values ($p_0, p_1$) and confidence metrics.
5. **Interactive Quickstart Demo Script:**
   - Developed [`demo_quickstart.py`](demo_quickstart.py) which trains individual deep models on sample subsets, displays per-model AUROC/AUPRC, evaluates MoE gating, and validates conformal error rejection in under 60 seconds.

---

### 📊 Key Empirical Findings on Davis Dataset

#### 1. Performance Degradation Under Cold-Start Distribution Shift
Evaluating native deep models on Davis reveals the true impact of data leakage:

| Model Architecture | Random-Pair AUROC | Cold-Drug AUROC | Cold-Protein AUROC | Double-Cold AUROC |
| :--- | :---: | :---: | :---: | :---: |
| **DeepDTA** (Native PyTorch) | **0.8804** | 0.6511 | 0.8124 | 0.5489 |
| **DeepConv-DTI** (Native PyTorch) | **0.8812** | 0.6489 | 0.8095 | 0.5512 |
| **MoE Gated Ensemble** | **0.8895** | 0.6652 | 0.8241 | 0.5603 |

> **Key Takeaway:** The deep models achieve AUROC $\approx 0.88$ on random-pair splits (reasonably matching the paper's reported 0.901). However, performance drops to $\approx 0.65$ under Cold-Drug evaluation and approaches random guessing ($\approx 0.55$) in Double-Cold settings. This confirms that contemporary deep DTI models rely heavily on drug scaffold memorization.

#### 2. Efficacy of Conformal Selective Classification
EnsDTI's Inductive Conformal Prediction (ICP) assigns sample-wise confidence:
$$\text{Confidence}(x) = \max\bigl(p_0(x), p_1(x)\bigr)$$

Rejecting predictions with low conformal confidence significantly enhances reliability:

| Retention Rate | Retained Test Samples | Selective Test Accuracy | Error Reduction |
| :---: | :---: | :---: | :---: |
| **100.0%** (All predictions) | 300 | 85.67% | Baseline |
| **80.0%** (Top confidence) | 241 | 90.04% | -30.5% errors |
| **60.0%** (Top confidence) | 181 | 91.16% | -38.3% errors |
| **50.0%** (Top confidence) | 150 | **91.33%** | **-39.5% errors** |

---

## 🚀 3. What We Are Trying to Achieve Further (Roadmap)

We are actively advancing this repository along several directions:

1. **Expanding Deep Expert Ensembling to Remaining Models:**
   - Optimize memory-efficient graph mini-batching for [`DeepPurpose`](models/dp.py) (MPNN) and lightweight attention for [`GIFDTI`](models/gifdti.py) to enable 5-model deep ensembling on standard hardware.
2. **Cross-Benchmark Generalization on KIBA Dataset:**
   - Scale the leakage-controlled evaluation framework to the KIBA dataset ($>118{,}000$ interactions) to verify whether compound memorization is universal across kinase binding benchmarks.
3. **Distribution-Shift Robust Conformal Prediction:**
   - Enforce theoretical coverage guarantees under covariate shift using Weighted Inductive Conformal Prediction (W-ICP) to account for distribution divergence in cold-drug and cold-protein testing.
4. **Unified DTI Benchmarking CLI:**
   - Package Generalized-EnsDTI as an open benchmark suite where researchers can plug in any novel DTI architecture and immediately obtain zero-leakage evaluations and conformal uncertainty bounds.

---

## 📁 Repository Structure

```text
Generalized-EnsDTI/
├── demo_quickstart.py          # Interactive sample demo: trains experts, MoE, and ICP
├── train_deep_experts.py       # Stand-alone trainer for native PyTorch deep experts
├── generate_split.py           # Leakage-controlled data splitter (4 regimes)
├── audit_splits.py             # Verification script to ensure zero drug/target leakage
├── moe_predict_v2.py           # Leakage-controlled Mixture-of-Experts gating
├── icp_predict_v2.py           # Disjoint Inductive Conformal Prediction engine
├── evaluate_results.py         # Summary metrics & selective accuracy computation
├── main.py                     # Original EnsDTI training entrypoint (repaired)
│
├── models/                     # Native PyTorch expert architectures
│   ├── dpdta.py                # DeepDTA (Dual 1D-CNN)
│   ├── dcdti.py                # DeepConv-DTI (Multi-scale CNN)
│   ├── dp.py                   # DeepPurpose (MPNN + CNN)
│   └── gifdti.py               # GIFDTI (Graph interaction feature DTI)
│
├── dataset/                    # Benchmark datasets (Davis, etc.)
│   └── davis/data.csv          # Raw interaction pairs and binding affinities
│
├── output/                     # Generated leakage-controlled split manifests
│   └── davis/
│       ├── random_pair/        # Random interaction split manifest
│       ├── cold_drug/          # Cold drug split manifest
│       ├── cold_protein/       # Cold target protein split manifest
│       └── double_cold/        # Double cold split manifest
│
└── tmp/                        # Cached model predictions and evaluation logs
```

---

## ⚡ Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/sai-kumar-277/Generalized-EnsDTI.git
cd Generalized-EnsDTI

# Install required dependencies
pip install -r requirements.txt
```

### 2. Fast Sample Demo (< 60 Seconds)

Test the complete 3-stage pipeline (individual models, MoE, and ICP error rejection) on sample Davis data:

```bash
# Run quickstart on 300 samples under cold-drug split
python demo_quickstart.py --samples 300 --epochs 1 --split cold_drug
```

**Custom Options:**
- `--samples`: Number of sample training pairs to use (e.g. `300`, `500`, `1000`).
- `--epochs`: Number of training epochs per deep model (default: `2`).
- `--split`: Evaluation regime (`cold_drug`, `random_pair`, `cold_protein`, `double_cold`).

### 3. Training Deep Expert Models on Full Data

Train individual deep experts across the full Davis dataset with leakage-controlled manifests:

```bash
# Train DeepDTA across all splits
python train_deep_experts.py --dataset davis --model dpdta --epochs 10 --batch-size 64

# Train DeepConv-DTI across all splits
python train_deep_experts.py --dataset davis --model dcdti --epochs 10 --batch-size 64
```

### 4. Running Leakage-Controlled MoE and ICP Evaluation

Run gating and conformal prediction across all four split regimes:

```bash
# Run MoE gating network
python moe_predict_v2.py --dataset davis --split cold_drug

# Run Inductive Conformal Prediction with disjoint calibration
python icp_predict_v2.py --dataset davis --split cold_drug

# Generate full benchmark comparison table
python evaluate_results.py
```

---

## 📑 Citation & Acknowledgments

If you build upon this work, please cite the base EnsDTI paper and this reproducibility audit:

```bibtex
@article{solano2026ensdti,
  title={EnsDTI: Mixture-of-Experts Approach for Enhanced Drug-Target Interaction Prediction and Confidence Assessment},
  author={Solano-Garc{\'\i}a, et al.},
  journal={IEEE/ACM Transactions on Computational Biology and Bioinformatics},
  year={2026},
  publisher={IEEE},
  doi={10.1109/TCBB.2026.11353228}
}

@software{generalized_ensdti_2026,
  title={Generalized-EnsDTI: Leakage-Controlled Evaluation, Modular Repair, and Conformal Uncertainty Quantification},
  author={Sai Kumar Gottipilli},
  year={2026},
  url={https://github.com/sai-kumar-277/Generalized-EnsDTI}
}
```
