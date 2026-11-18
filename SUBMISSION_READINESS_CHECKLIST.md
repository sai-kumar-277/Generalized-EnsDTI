# Submission Readiness Checklist: EnsDTI Research Arc

**Project**: Generalized EnsDTI (Robust MoE Gating & Cold-Aware Conformal Prediction)  
**Verification Date**: October 2026  
**Status**: **ALL ITEMS VERIFIED — READY FOR SUBMISSION**  
**Remote Git Repository**: [`https://github.com/sai-kumar-277/Generalized-EnsDTI.git`](https://github.com/sai-kumar-277/Generalized-EnsDTI.git)  
**Primary Working Branch**: `leakage-controlled-evaluation` (mirrored to `main`)  
**Commit Hash**: `a834e04`

---

## 1. Submission Package Checklist

| Category | Item Description | Location / Reference | Verification Status |
| :--- | :--- | :--- | :---: |
| **Comprehensive Report** | All 25 sections complete, including Executive Summary, diagnostics, repairs, statistical testing, KIBA validation, limitations, and references. | [`FINAL_EVALUATION_REPORT.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/FINAL_EVALUATION_REPORT.md) | **VERIFIED** |
| **Standalone Summary** | 1–2 page submission summary (<600 words), containing 1-sentence headline claim, 4 quantitative findings, figure references, and citations. | [`RESEARCH_SUMMARY.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/RESEARCH_SUMMARY.md) | **VERIFIED** |
| **Presentation Deck** | 10-slide conference/defense deck outline detailing slide titles, 3–5 bullet points, and exact figure/table references. | [`PRESENTATION_OUTLINE.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/PRESENTATION_OUTLINE.md) | **VERIFIED** |
| **Limitations & Roadmap** | Thorough analysis of 5 honest scientific limitations and 5 concrete future research directions. | [`LIMITATIONS_AND_FUTURE_WORK.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/LIMITATIONS_AND_FUTURE_WORK.md) | **VERIFIED** |
| **Consistency Audit** | Meticulous cross-check of 22 empirical claims, statistics, and sample sizes between summary and full report (0 discrepancies). | [`CONSISTENCY_CHECK.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/CONSISTENCY_CHECK.md) | **VERIFIED** |
| **QA Audit Report** | Systematic QA audit report confirming markdown table alignment, link integrity, terminology consistency, and academic citations. | [`QA_REPORT.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/QA_REPORT.md) | **VERIFIED** |

---

## 2. Key Visual Figures Verification

| Figure Identifier | Target Phenomena Visualized | Disk Path | Dimensions & Integrity |
| :--- | :--- | :--- | :---: |
| **Figure 1 (Multi-Seed Gating)** | Per-seed AUROC comparison across seeds $\{42, 123, 999\}$ for `cold_drug` and `double_cold`, illustrating sign inversion elimination and recovery. | [`gating_multiseed_per_seed.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/davis/gating_multiseed_per_seed.png) | 4170 $\times$ 1698 (PNG) — **VERIFIED** |
| **Figure 2 (Conformal Calibration)** | Empirical coverage vs. significance level $\alpha \in [0.05, 0.30]$, illustrating catastrophic collapse of Regimes A/B and exact restoration under Regime C. | [`cold_aware_icp_coverage_alpha_curve.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/davis/cold_aware_icp_coverage_alpha_curve.png) | 5364 $\times$ 1692 (PNG) — **VERIFIED** |
| **Figure 3 (KIBA External Validation)** | Gating comparison on KIBA ($N=117,929$) under chemical and dual-entity shift, showing dilution failure of naive averaging and superiority of Novelty-Aware Gating. | [`kiba_gating_comparison.png`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/kiba/kiba_gating_comparison.png) | 4200 $\times$ 1500 (PNG) — **VERIFIED** |
| **Secondary Figures (x7)** | Risk-coverage curves and per-regime ROC/PR curves for Davis benchmarks. | [`output/davis/`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/output/davis/) | All 7 exist on disk — **VERIFIED** |

---

## 3. Methodological and Experimental Rigor Checklist

* **[x] Strict Leakage Control Enforced**: 
  - All 4 split regimes (`random_pair`, `cold_drug`, `cold_protein`, `double_cold`) partitioned into 3 strictly disjoint sets (`train` 70%, `calibration` 10%, `test` 20%).
  - Zero target overlap verified via set intersection audits across Davis and KIBA.
* **[x] Multi-Seed Validation**:
  - Re-evaluated across random seeds $\{42, 123, 999\}$ across 4 gating variants $\times$ 4 split regimes (48 per-seed result files in `tmp/davis/gating_multiseed/`).
  - Variance on `cold_drug` reduced by two orders of magnitude ($\sigma = 0.2810 \to 0.0027$).
* **[x] Formal Statistical Hypothesis Testing**:
  - $B=2,000$ paired bootstrap hypothesis tests confirm `double_cold` recovery ($\Delta = +0.1047, p = 0.0290 < 0.05$) and `cold_drug` stabilization ($p < 0.0001$).
  - Paired squared-error $t$-test confirms significant error reduction ($t = 4.355, p = 7.25 \times 10^{-6}$).
  - Paired Wilcoxon signed-rank test confirms residual reduction on `cold_drug` ($p < 10^{-60}$).
* **[x] Cross-Dataset Generalization on KIBA**:
  - Tested on 117,929 kinase interactions. Novelty-Aware Gating secured top AUROC across all 4 regimes and achieved 79.90% classification accuracy and 88.87%–95.16% selective screening accuracy at 50% retention.
* **[x] Conformal Uncertainty Calibration (Regime C)**:
  - Finite-sample coverage guarantees verified across all 4 regimes and 6 significance levels ($\alpha \in [0.05, 0.30]$), maintaining empirical coverage within $1-4\%$ of nominal.

---

## 4. Citation and Attribution Checklist

* **[x] Core Model & Architecture Citations**: EnsDTI (Solano-Paucar et al., 2024), DeepDTA (Öztürk et al., 2018), DeepConv-DTI (Lee et al., 2019), DACMF-DTI (He et al., 2024).
* **[x] Benchmark Dataset Citations**: Davis (Davis et al., 2011), KIBA (Tang et al., 2014).
* **[x] Conformal Prediction & Shift Citations**: Vovk et al. (2005), Angelopoulos & Bates (2021), Tibshirani et al. (2019), Barber et al. (2023).
* **[x] Statistical Resampling Citations**: Efron (1979), Wilcoxon (1945).
* **[x] Bibliographic Section**: Complete Section 25 added to [`FINAL_EVALUATION_REPORT.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/FINAL_EVALUATION_REPORT.md) and Section 5 added to [`RESEARCH_SUMMARY.md`](file:///c:/Users/saiku/OneDrive/Desktop/UROP/EnsDTI/RESEARCH_SUMMARY.md).

---

## 5. Remaining Gaps or Blockers

* **Identified Blockers**: **NONE (0 Blockers)**
* **Identified Discrepancies**: **NONE (0 Discrepancies)**
* **Identified Broken Links**: **NONE (0 Broken Links across 52 audited URLs)**
* **Identified Table Misalignments**: **NONE (0 Table Errors remaining)**

---

## 6. Final Recommendation & Sign-Off

The research arc is complete, robustly defended with multi-seed and cross-dataset statistical evidence, fully audited for quality, and packaged for manuscript submission, thesis presentation, or industrial virtual screening deployment.
