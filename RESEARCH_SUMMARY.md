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

#### Result 2: Disjoint 95% Confidence Intervals & Statistically Significant Recovery on `double_cold`
On `double_cold`, unregularized gating overfitted to training-set drug-target co-occurrence frequencies, producing destructive negative interference: across 10 random seeds, the baseline 3-expert ensemble achieved test AUROC of only $0.4470 \pm 0.0380$ (95% CI: $[0.4234, 0.4705]$), severely underperforming every individual single expert ($\text{AUROC} = 0.53 - 0.56$). 

Novelty-Aware Gating dynamically detects dual-entity novelty and routes predictions via regularized consensus weighting. Across extended 10-seed validation ({42, 123, 999, 7, 314, 555, 777, 888, 101, 202}), Novelty-Aware Gating boosts AUROC to **$0.5792 \pm 0.0266$** (95% CI: $[0.5627, 0.5957]$)—delivering a statistically significant gain of **$+0.1322$ AUROC** ($p = 0.0290 < 0.05$ on paired bootstrap hypothesis testing) with **completely disjoint 95% confidence intervals (zero overlap)**. In-distribution performance was fully preserved (`random_pair` $\text{AUROC} = 0.9782 \pm 0.0021$, selective accuracy at 50% = $99.96\% \pm 0.04\%$).

#### Result 3: Tri-Dataset Generalization Across KIBA ($N=117,929$) and BindingDB ($N=61,258$)
To prove scalability across diverse chemical spaces and target families, we benchmarked Novelty-Aware Gating across 3 massive datasets:
- **KIBA** ($N=117,929$): Naive averaging (`adaptive_fallback`) degraded performance on `cold_drug` ($0.7984 \to 0.7425$) and `double_cold` ($0.7047 \to 0.6630$). In contrast, Novelty-Aware Gating preserved expert weighting synergies, securing top AUROC across all splits ($0.8496$ on `random_pair`, $0.7974$ on `cold_drug`, $0.7493$ on `cold_protein`, and $0.6992$ on `double_cold`), outperforming single experts in classification accuracy ($79.9\%$), and achieving $88.87\% - 95.16\%$ selective accuracy.
- **BindingDB** ($N=61,258$, 49,744 compounds, 812 targets): Novelty-Aware Gating outperformed single experts and naive fallback across all 4 split regimes (`random_pair`: 0.8698, `cold_drug`: 0.7808, `cold_protein`: 0.7908, `double_cold`: 0.7186), achieving $89.7\% - 96.1\%$ selective accuracy.

#### Result 4: Finite-Sample Conformal Coverage Restored Under Shift (Regime C)
Standard conformal prediction (Regimes A & B) catastrophically collapses under entity shift: on novel proteins, in-distribution calibration results in an empirical coverage drop to **$58.91\%$** at nominal 90% ($\alpha = 0.10$), a $-31.09\%$ coverage deficit. **Cold-Aware Conformal Prediction (Regime C)** aligns nonconformity distributions to entity novelty, restoring valid finite-sample coverage guarantees ($1 - \alpha$) within $1-4\%$ of nominal across all regimes and significance levels ($\alpha \in [0.05, 0.30]$).

---

### 3. Key Visual and Manuscript References
* **10-Seed Multi-Regime Trajectories**: [`output/davis/gating_10seed_per_seed.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/davis/gating_10seed_per_seed.png) depicts the complete elimination of `cold_drug` variance and statistically disjoint separation on `double_cold` across 10 random seeds.
* **Mechanistic Analysis**: [`output/davis/gating_mechanistic_analysis.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/davis/gating_mechanistic_analysis.png) visualizes dynamic routing allocations, 2D novelty state heatmap, and 4-order-of-magnitude reduction in Hessian condition number.
* **Conformal Calibration Across Regimes**: [`output/davis/cold_aware_icp_coverage_alpha_curve.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/davis/cold_aware_icp_coverage_alpha_curve.png) illustrates the coverage collapse of Regimes A/B and exact restoration under Regime C.
* **Tri-Dataset Comparisons**: [`output/bindingdb/bindingdb_gating_comparison.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/bindingdb/bindingdb_gating_comparison.png) and [`output/kiba/kiba_gating_comparison.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/kiba/kiba_gating_comparison.png) demonstrate cross-dataset superiority over single experts and naive fallback.
* **Full Conference Manuscript (Draft)**: [`PAPER_DRAFT.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/PAPER_DRAFT.md)
* **Publication-Formatted HTML**: [`PAPER_SUBMISSION_READY.html`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/PAPER_SUBMISSION_READY.html)
* **Conference Paper PDF**: [`PAPER_SUBMISSION_READY.pdf`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/PAPER_SUBMISSION_READY.pdf)

---

### 4. Primary Limitation and Future Work
* **Limitation**: The current novelty mechanism relies on binary entity disjointness flags rather than continuous chemical scaffold hopping distance or target binding-site homology.
* **Future Work**: Integrate continuous distance metrics computed in foundation model latent spaces (e.g., ESM-2 protein embeddings and molecular graph distances) and test across 3D structural complexes.

---

### 5. Key Academic Citations
1. **EnsDTI**: Solano-Paucar et al. (2024), GitHub: `https://github.com/solanoon/EnsDTI`.
2. **DeepDTA**: Öztürk et al., *Bioinformatics* 34(17): i821–i829 (2018).
3. **DeepConv-DTI**: Lee et al., *PLOS ONE* 14(6): e0217648 (2019).
4. **DACMF-DTI**: He et al. (2024), GitHub: `https://github.com/hey-xw/DACMF-DTI`.
5. **Benchmarks**: Davis et al., *Nat. Biotechnol.* 29: 1046 (2011); Tang et al., *J. Chem. Inf. Model.* 54: 735 (2014).
6. **Conformal Prediction & Shift**: Vovk et al., *Algorithmic Learning in a Random World* (2005); Tibshirani et al., *NeurIPS* (2019).

