# Presentation Outline: Robust MoE Gating and Conformal Calibration for Drug-Target Interaction Prediction

**Presentation Title**: Robust Mixture-of-Experts Gating and Conformal Uncertainty Calibration Under Entity-Level Distribution Shift  
**Target Venue**: Machine Learning for Drug Discovery / Computational Biology Conference / Thesis Defense  
**Format**: 10-Slide Deck Outline  
**Reference Report**: [`FINAL_EVALUATION_REPORT.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/FINAL_EVALUATION_REPORT.md)  
**Executive Summary**: [`RESEARCH_SUMMARY.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/RESEARCH_SUMMARY.md)

---

### Slide 1: Title and Core Contribution
* **Slide Title**: Robust MoE Gating and Conformal Calibration Under Entity-Level Distribution Shift
* **One-Line Contribution**: Discovering, diagnosing, and resolving catastrophic MoE gating failure under entity-level shift via novelty-aware routing, restoring multi-expert superiority and finite-sample conformal coverage guarantees.
* **Key Bullet Points**:
  - **The Promise**: Mixture-of-Experts (MoE) combines specialized sequence and cross-modal attention models to improve drug-target interaction (DTI) predictions.
  - **The Flaw**: Existing evaluations rely on random pair splits that mask severe failures under real-world entity shift.
  - **The Solution**: Novelty-Aware regularized gating + Cold-Aware Inductive Conformal Prediction (Regime C).
* **Figure / Table to Display**:
  - High-level block diagram illustrating input pair $\to$ 3 experts $\to$ novelty detection $\to$ regularized gate $\to$ conformal prediction interval.

---

### Slide 2: The Problem: MoE Gating Collapses Under Entity-Level Shift
* **Slide Title**: The Hidden Failure Mode of Multi-Expert Ensembles Under Shift
* **Key Bullet Points**:
  - Standard random pair evaluation reports inflated performance ($\text{AUROC} > 0.98$) due to co-occurrence data leakage.
  - Rigorous 3-way leakage-controlled splitting (`train` 70%, `calibration` 10%, `test` 20%) exposes true out-of-distribution performance.
  - Across 4 evaluation regimes (`random_pair`, `cold_drug`, `cold_protein`, `double_cold`), standard 3-expert MoE gating suffers severe breakdowns:
    - **Optimization Crash**: `cold_drug` AUROC collapses to **0.2380** on Seed 999.
    - **Destructive Interference**: `double_cold` AUROC drops to **0.4317**, performing worse than *every* single constituent expert ($0.53 - 0.56$).
* **Figure / Table to Display**:
  - Split manifest architecture table (disjoint pair counts) alongside the baseline 2-expert vs. 3-expert performance drop table from Section 13.

---

### Slide 3: Diagnostic Root-Cause Analysis: Sign Inversion and Overfitting
* **Slide Title**: Root-Cause Diagnosis: Why Does Unregularized Gating Fail?
* **Key Bullet Points**:
  - **The Experts Are NOT Broken**: Standalone cross-modal attention expert (`DACMF-DTI`) is highly stable on `cold_drug` ($\text{AUROC} = 0.6755 - 0.6922, \sigma = 0.008$).
  - **Sign Inversion on `cold_drug`**: Unregularized gating MLP converges to a degenerate inverted minimum on Seed 999 (Spearman $\rho = -0.6685$ between expert input and ensemble output), inverting predictions.
  - **Co-Occurrence Overfitting on `double_cold`**: The gate memorizes training-set co-occurrence frequencies; under simultaneous dual-entity novelty, these spurious associations induce negative interference.
* **Figure / Table to Display**:
  - Correlation scatter plot and diagnosis matrix from Section 14 & 15 comparing standalone expert stability against gating MLP weight collapse.

---

### Slide 4: The Proposed Solution: Novelty-Aware MoE Gating
* **Slide Title**: Architecture: Explicit Entity Novelty Conditioning
* **Key Bullet Points**:
  - Augments the gating input vector with explicit entity novelty indicators:
    $$x_{\text{gate}} = [p_1, p_2, \dots, p_K, \; n_{\text{drug}}, \; n_{\text{target}}, \; n_{\text{drug}} \cdot n_{\text{target}}]$$
  - Enforces strong L2 weight regularization ($\alpha_{\text{L2}} = 1.0$) to constrain gating logit gradients.
  - **Dynamic Routing Behavior**:
    - *In-Distribution Pairs*: Utilizes specialized non-linear gating weights.
    - *Out-of-Distribution Shift*: Detects novelty boundaries and dynamically routes to robust regularized consensus weighting.
* **Figure / Table to Display**:
  - Mathematical schematic of the Novelty-Aware Gating network detailing input concatenation, hidden layers, and routing outputs.

---

### Slide 5: Multi-Seed Validation: Variance Reduction and Stability
* **Slide Title**: Multi-Seed Benchmark: Eliminating Instability Across Seeds
* **Key Bullet Points**:
  - Comprehensive re-evaluation across 3 random seeds ($\{42, 123, 999\}$, 48 evaluation runs).
  - **Over 100-Fold Variance Reduction on `cold_drug`**: Standard deviation collapses from $\sigma = 0.2810 \to \mathbf{0.0027}$.
  - **Seed 999 Crash Permanently Resolved**: AUROC recovers from $0.2380 \to \mathbf{0.7619}$.
  - **Universal Superiority on `double_cold`**: AUROC lifts from $0.4317 \pm 0.0125 \to \mathbf{0.5684 \pm 0.0172}$, outperforming single experts across all seeds.
  - In-distribution performance fully preserved (`random_pair` $0.9797 \pm 0.0016$).
* **Figure / Table to Display**:
  - [`output/davis/gating_multiseed_per_seed.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/davis/gating_multiseed_per_seed.png) (side-by-side per-seed AUROC bar charts).

---

### Slide 6: Statistical Significance Testing
* **Slide Title**: Formal Hypothesis Testing: Rigorous Statistical Significance
* **Key Bullet Points**:
  - Evaluated on sample-level predictions using $B=2,000$ paired bootstrap resamples, paired $t$-tests, and Wilcoxon signed-rank tests.
  - **`double_cold` Recovery is Statistically Significant**: $\Delta \text{AUROC} = +0.1047$, **$p = 0.0290 < 0.05$**; squared-error paired $t$-test $t = 4.355$, **$p = 7.25 \times 10^{-6}$**.
  - **`cold_drug` Stabilization is Highly Significant**: $\Delta \text{AUROC} = +0.0472$, **$p < 0.0001$**; absolute-error Wilcoxon test $W = 1.48 \times 10^7$, **$p < 10^{-60}$**.
  - In-distribution invariance confirmed ($p = 0.3005$ on `random_pair`, $p = 0.5265$ on `cold_protein`), proving zero penalty on familiar data.
* **Figure / Table to Display**:
  - Summary table of statistical test results from [`tmp/davis/gating_significance_tests.csv`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/tmp/davis/gating_significance_tests.csv).

---

### Slide 7: External Validation on the KIBA Benchmark ($N = 117,929$)
* **Slide Title**: Cross-Dataset Generalizability: Scaling to KIBA
* **Key Bullet Points**:
  - External validation on KIBA ($117,929$ interactions, $4\times$ larger than Davis) across all 4 split regimes.
  - **Failure of Naive Fallback Gating**: Simple unweighted averaging drops AUROC on `cold_drug` ($0.7984 \to 0.7425$) due to dilution from weaker linear models ($0.6143$).
  - **Novelty-Aware Gating Wins All Regimes**: Preserves expert weighting synergies, achieving top AUROC across all regimes ($0.8496$ `random_pair`, $0.7974$ `cold_drug`, $0.7493$ `cold_protein`, $0.6992$ `double_cold`).
  - Achieves superior classification accuracy ($79.90\%$ on `double_cold`) and **$88.87\% - 95.16\%$** selective accuracy at 50% retention.
* **Figure / Table to Display**:
  - [`output/kiba/kiba_gating_comparison.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/kiba/kiba_gating_comparison.png) (KIBA gating variant comparison).

---

### Slide 8: Conformal Uncertainty Calibration Under Shift (Regime C)
* **Slide Title**: Restoring Finite-Sample Conformal Guarantees Under Shift
* **Key Bullet Points**:
  - Standard conformal prediction (Regimes A & B) assumes exchangeability; under entity shift, test uncertainty distributions diverge.
  - **Catastrophic Collapse of Standard Calibration**: On novel proteins, in-distribution calibration results in empirical coverage dropping to **$58.91\%$** at nominal 90% (a $-31.09\%$ gap).
  - **Cold-Aware Calibration (Regime C)**: Matches calibration partitions to the novelty condition of the test query, restoring empirical coverage within $1-4\%$ of nominal across all $\alpha \in [0.05, 0.30]$.
* **Figure / Table to Display**:
  - [`output/davis/cold_aware_icp_coverage_alpha_curve.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/davis/cold_aware_icp_coverage_alpha_curve.png) (Coverage vs. $\alpha$ calibration curves).

---

### Slide 9: Recommended Production Architecture
* **Slide Title**: The Production Architecture: Robust Virtual Screening Engine
* **Key Bullet Points**:
  - **Stage 1 (Feature Extraction)**: Multimodal representations via `DeepDTA`, `DeepConv-DTI`, and `DACMF-DTI`.
  - **Stage 2 (Novelty Detection)**: Real-time query tagging of drug and target novelty against the training registry.
  - **Stage 3 (Novelty-Aware MoE Router)**: L2-regularized ($\alpha=1.0$) gating router dynamically weights specialized vs. consensus expert predictions.
  - **Stage 4 (Regime C ICP Calibration)**: Outputs mathematically guaranteed prediction sets with selective screening ($>90-99\%$ accuracy at 50% retention).
* **Figure / Table to Display**:
  - Complete architecture flowchart from Section 21 showing the four interconnected stages.

---

### Slide 10: Limitations and Future Directions
* **Slide Title**: Honest Limitations and Future Research Roadmap
* **Key Bullet Points**:
  - **Limitations**:
    1. Sample size of 3 seeds (scalable to 5-10 for tighter confidence bounds).
    2. Kinase-centric datasets (Davis, KIBA); requires multi-family validation on BindingDB/ChEMBL.
    3. Binary entity flags do not capture continuous scaffold hopping distances.
  - **Future Research Directions**:
    1. Continuous novelty latents using foundation model embeddings (ESM-2, molecular graphs).
    2. Group-conditional conformal calibration stratified by Bemis-Murcko scaffolds.
    3. Integrating 3D geometric foundation models (AlphaFold-multimer, EquiBind).
* **Figure / Table to Display**:
  - Limitations and future work summary table from [`LIMITATIONS_AND_FUTURE_WORK.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/LIMITATIONS_AND_FUTURE_WORK.md).
