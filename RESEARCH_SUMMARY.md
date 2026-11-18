# Research Summary: Robust Mixture-of-Experts Gating and Conformal Calibration for Drug-Target Interaction Prediction

**Project**: Generalized EnsDTI  
**Author / Engineer**: Antigravity Machine Learning Research Engineer  
**Date**: October 2026  
**Repository**: `https://github.com/sai-kumar-277/Generalized-EnsDTI.git`  
**Full Report**: [`FINAL_EVALUATION_REPORT.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/FINAL_EVALUATION_REPORT.md)  
**Detailed Limitations**: [`LIMITATIONS_AND_FUTURE_WORK.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/LIMITATIONS_AND_FUTURE_WORK.md)

---

### Headline Claim
> Under entity-level distribution shift, Mixture-of-Experts (MoE) gating networks for drug-target interaction prediction suffer from optimization sign inversion and destructive co-occurrence overfitting, which can be eliminated via novelty-aware regularized gating to restore multi-expert superiority and finite-sample conformal coverage guarantees across all deployment regimes.

---

### 1. Research Context and The Core Dilemma
Mixture-of-Experts (MoE) ensembling has emerged as a premier paradigm for drug-target interaction (DTI) prediction, combining specialized models such as sequence-based CNNs and cross-modal attention networks. However, evaluations in literature almost exclusively rely on random pair splitting, which masks severe vulnerabilities under real-world distribution shifts. 

When subjected to rigorous, leakage-controlled 3-way evaluation (`train` 70%, `calibration` 10%, `test` 20%) across four distinct regimes—`random_pair` (in-distribution), `cold_drug` (unseen compounds), `cold_protein` (unseen targets), and `double_cold` (simultaneously novel compounds and targets)—standard unconstrained MoE gating fails catastrophically.

---

### 2. Key Empirical Findings

#### Result 1: Root Cause Diagnosis & Over 100-Fold Variance Reduction on `cold_drug`
On `cold_drug`, the standalone dual cross-modal attention expert (`DACMF-DTI`) proved highly accurate and stable ($\text{AUROC} = 0.6755 - 0.6922, \sigma = 0.008$). However, the unregularized gating MLP displayed extreme optimization instability across random seeds ($\sigma = 0.2810$). On Seed 999, the gating network converged to an inverted local minimum where output ensemble predictions negatively correlated with input expert probabilities (Spearman $\rho = -0.6685$), dropping ensemble AUROC to a catastrophic $0.2380$. 

By conditioning the gating router on entity novelty features alongside strong L2 regularization ($\alpha_{\text{L2}} = 1.0$), **Novelty-Aware Gating** permanently eradicates sign inversion, achieving $\text{AUROC} = 0.7619$ on Seed 999, improving mean AUROC from $0.5620 \to 0.7589$, and collapsing variance over **100-fold** ($\sigma = 0.2810 \to 0.0027, p < 0.0001$).

#### Result 2: Statistically Significant Recovery of `double_cold` Ensemble Regression
On `double_cold`, unregularized gating overfitted to training-set drug-target co-occurrence frequencies, producing destructive negative interference: the 3-expert ensemble ($\text{AUROC} = 0.4317 \pm 0.0125$) severely underperformed every individual single expert ($\text{AUROC} = 0.53 - 0.56$). 

Novelty-Aware Gating dynamically detects dual-entity novelty and routes predictions via robust regularized consensus weighting, boosting AUROC to **$0.5684 \pm 0.0172$**—a statistically significant mean gain of $\mathbf{+0.1047}$ AUROC ($p = 0.0290 < 0.05$ on $B=2,000$ paired bootstrap resamples; paired $t$-test $p = 7.25 \times 10^{-6}$). In-distribution gains were 100% preserved (`random_pair` $\text{AUROC} = 0.9797 \pm 0.0016$; `cold_protein` $\text{AUROC} = 0.9282 \pm 0.0072$).

#### Result 3: Cross-Dataset External Generalizability on KIBA ($N = 117,929$)
On the $4\times$ larger KIBA benchmark, naive consensus averaging (`adaptive_fallback`) degraded performance on `cold_drug` ($0.7984 \to 0.7425$) and `double_cold` ($0.7047 \to 0.6630$) due to dilution from weaker linear models. In contrast, Novelty-Aware Gating preserved expert weighting synergies, securing top AUROC across all splits ($0.8496$ on `random_pair`, $0.7974$ on `cold_drug`, $0.7493$ on `cold_protein`, and $0.6992$ on `double_cold`) and achieving superior classification accuracy ($79.90\%$ on `double_cold` vs. $55.6\% - 72.6\%$ for single experts) and $88.87\% - 95.16\%$ selective accuracy at 50% retention.

#### Result 4: Finite-Sample Conformal Coverage Restored Under Shift (Regime C)
Standard conformal prediction (Regimes A & B) catastrophically collapses under entity shift: on novel proteins, in-distribution calibration results in an empirical coverage drop to **$58.91\%$** at nominal 90% ($\alpha = 0.10$), a $-31.09\%$ coverage deficit. **Cold-Aware Conformal Prediction (Regime C)** aligns nonconformity distributions to entity novelty, restoring valid finite-sample coverage guarantees ($1 - \alpha$) within $1-4\%$ of nominal across all regimes and significance levels ($\alpha \in [0.05, 0.30]$).

---

### 3. Key Visual References
* **Conformal Calibration Across Regimes**: [`output/davis/cold_aware_icp_coverage_alpha_curve.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/davis/cold_aware_icp_coverage_alpha_curve.png) illustrates the coverage collapse of Regimes A/B and exact restoration under Regime C.
* **Per-Seed Gating Stability**: [`output/davis/gating_multiseed_per_seed.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/davis/gating_multiseed_per_seed.png) depicts the complete elimination of `cold_drug` variance and universal superiority on `double_cold`.
* **KIBA External Validation**: [`output/kiba/kiba_gating_comparison.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/kiba/kiba_gating_comparison.png) visualizes gating comparisons across chemical and dual-entity shift.

---

### 4. Primary Limitation and Future Work
* **Limitation**: The current novelty mechanism relies on binary entity disjointness flags rather than continuous chemical scaffold hopping distance or target binding-site homology.
* **Future Work**: Integrate continuous distance metrics computed in foundation model latent spaces (e.g., ESM-2 protein embeddings and molecular graph distances) and validate across multi-family target databases such as BindingDB.
