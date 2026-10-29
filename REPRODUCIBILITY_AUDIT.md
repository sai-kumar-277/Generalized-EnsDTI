# EnsDTI Reproducibility Audit

Generated: 2026-10-29  
Auditor: Automated Reproducibility Check

---

## 1. Repository Commit Hash

```
be6ea5dd4afec8c236bdb8ed7e5cf51c8290ee94
```

Branch: `master` (audit branch: `leakage-controlled-evaluation`)

---

## 2. Python and Package Versions

| Package | Version |
|---------|---------|
| Python | 3.13.7 |
| torch | 2.11.0 |
| numpy | 2.3.3 |
| pandas | (installed) |
| scikit-learn | 1.7.2 |
| joblib | 1.5.2 |
| bidict | 0.24.1 (**not pre-installed**) |
| rdkit | 2026.9.1 (**not pre-installed**) |
| matplotlib | (installed) |
| wandb | **NOT installed** |
| torch_geometric | **NOT installed** |
| ogb | **NOT installed** |

---

## 3. Broken Imports

| File | Line | Broken Import | Reason |
|------|------|---------------|--------|
| `data/dataset.py` | 7 | `from bidict import bidict` | `bidict` not installed by default |
| `data/dataset.py` | 20 | `from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector` | `ogb` not installed |
| `data/dataset.py` | 21 | `from utils import *` | relative import error; should be `from .utils import *` |
| `data/dataset.py` | 24 | `from torch_geometric.data import Data, InMemoryDataset` | `torch_geometric` not installed |
| `data/data_loader.py` | 5 | `from datasets import CustomCPIDataset` | wrong module name; should be `from data.dataset import CustomCPIDataset` |
| `data/data_utils.py` | 6 | `from data_loader import generate_dataset_by_model` | missing package prefix; should be `from data.data_loader import ...` |
| `data/data_utils.py` | 7 | `from data_split import split_dataset, split_dataset_by_fold` | missing package prefix |
| `data/__init__.py` | 4 | `from .utils import prepare_data, return_dataloader, split_dataset_by_fold` | conflicts with line 6 which also imports these from `.data_utils` |
| `main.py` | 4 | `import wandb` | `wandb` not installed |
| `main.py` | 9 | `from models import declare_model` | `declare_model` not defined in `models/__init__.py` (only `build_model` exists) |
| `models/__init__.py` | 13 | `from .cpi import CPIPrediction` | file `models/cpi.py` does not exist |
| `models/dcdti.py` | 4 | `from drug_target_interaction.end.models.GlobalMaxPooling1D import GlobalMaxPooling1D` | non-existent package |
| `models/dpdta.py` | 5 | `from drug_target_interaction.end.models.utils import GlobalMaxPooling1D` | non-existent package |
| `models/dp.py` | 7 | `from drug_target_interaction.end.models.utils import get_kernel_size` | non-existent package |
| `engine/trainer.py` | 1 | `import wandb` | `wandb` not installed |
| `data/utils.py` | 3 | `from torch_geometric.data import Data` | `torch_geometric` not installed |

---

## 4. Missing Files and Modules

| Item | Status |
|------|--------|
| `models/cpi.py` | **MISSING** — imported in `models/__init__.py` but file does not exist |
| `models/perceivercpi.py` | **MISSING** — referenced but commented out |
| `models/smtdta.py` | **MISSING** — referenced but commented out |
| `tmp/` directory | Not committed; must be created at runtime |
| `output/` directory | Not committed; must be created at runtime |
| `tmp/<dataset>/<model>_result.csv` | Precomputed expert prediction files **NOT included** |
| `tmp/<dataset>/train_index.csv` | Split index files **NOT included** |
| `tmp/<dataset>/test_index.csv` | Split index files **NOT included** |
| `results/new_ensdti_internal_validation_on_<dataset>.csv` | ICP input files **NOT included** |
| `requirements.txt` | **MISSING** |

---

## 5. Missing, Stubbed, or Non-Functional Models

| Model | Status | Reason |
|-------|--------|--------|
| `DeepConvDTI` (dcdti) | **Non-functional** | Imports from non-existent `drug_target_interaction` package |
| `DeepDTA` (dpdta) | **Non-functional** | Same broken import |
| `DeepPurpose` (dp) | **Non-functional** | Same broken import + `get_kernel_size` not implemented |
| `MDeePred` (mdprd) | **Functional architecture** | No broken imports; requires CUDA for training |
| `GIFDTI` (gifdti) | **Unknown** — file exists but not imported in `__init__.py` |
| `CPIPrediction` (cpi) | **MISSING** — file `models/cpi.py` does not exist |
| `PerceiverCPI` | **MISSING** — commented out |
| `SMTDTA` | **MISSING** — commented out |

---

## 6. Runtime Errors

| Script | Error | Location |
|--------|-------|----------|
| `main.py` | `ModuleNotFoundError: No module named 'wandb'` | Line 4 |
| `main.py` | `sys.exit('Do not support training model with cpu')` | Line 20 — hard blocks CPU execution |
| `main.py` | `ModuleNotFoundError: No module named 'drug_target_interaction'` | Via `models/dcdti.py` line 4 |
| `moe_predict.py` | `FileNotFoundError: tmp/davis/dcdti_result.csv` | Line 28 — precomputed files missing |
| `moe_predict.py` | **Bug: unfitted model used for prediction** | Line 52 uses `model.predict_proba()` instead of `clf.predict_proba()` |
| `icp_predict.py` | `FileNotFoundError: results/new_ensdti_internal_validation_on_davis.csv` | Line 39 |

---

## 7. Is `wandb` Mandatory or Optional?

**Mandatory in original code** — hard imported with no fallback. Running any script without `wandb` installed crashes immediately.

**Assessment**: `wandb` is functionally optional; it is used solely for logging. It can be made optional by wrapping in try/except with a no-op fallback.

---

## 8. CPU Execution Support

**NOT supported in original `main.py`**: Line 19–20 explicitly exits if device is not CUDA:
```python
if args.exp_mode != 'pure_test':
    sys.exit('Do not support training model with cpu')
```

Assessment: This is an artificial constraint. The model architectures are standard PyTorch and are compatible with CPU execution (with performance trade-offs for MDeePred due to 2D convolutions on protein matrices). **The CPU block must be removed**.

---

## 9. Pretrained Weights and Precomputed Prediction Files

**Required but not provided:**

- `tmp/<dataset>/<model>_result.csv` — expert model predictions needed by `moe_predict.py`
- `tmp/<dataset>/train_index.csv` / `test_index.csv` — split indices needed by `moe_predict.py`
- `results/new_ensdti_internal_validation_on_<dataset>.csv` — MoE output needed by `icp_predict.py`
- No pretrained model weights are included in the repository

The pipeline cannot run from scratch without either: (a) training all expert models (requires CUDA + broken deps), or (b) providing precomputed predictions.

---

## 10. Cold-Drug and Cold-Protein Splits

**NOT implemented in the codebase.**

`data/data_split.py` contains only:
- `split_dataset()` — random row shuffle, 90/10 train/val
- `split_dataset_by_fold()` — random k-fold by interaction row index

Both split at the **interaction-pair level**, not the **entity level**. In Davis (64 drugs × 442 proteins), a random pair split guarantees that every drug and every protein appears in both training and test sets, creating complete identity leakage.

The paper claims cold-drug/cold-protein results following protocol of [34], but no implementation exists in the repository.

---

## 11. ICP Calibration Overlap with Training Data

**YES — the original ICP calibration overlaps with training data.**

In `icp_predict.py` lines 43–45:
```python
cal_df_frac = 0.5
cal_df = df.sample(frac=cal_df_frac)           # 50% of ALL rows
test_df = pd.concat([df, cal_df]).drop_duplicates(keep=False)  # other 50%
```

The input `df` is read from `results/new_ensdti_internal_validation_on_{d}.csv`, which covers the full internal validation set. Calibration rows are sampled from the same pool as rows used to train the expert models and the MoE, so calibration scores are computed on **in-distribution** data that the model was trained on — not a held-out calibration set. This violates the core ICP exchangeability assumption.

---

## 12. Data Columns

| Column | Role |
|--------|------|
| `lig` | Drug identity (compound name/ID) |
| `pro` | Protein identity (gene/kinase name) |
| `smi` | Drug SMILES string |
| `seq` | Protein amino-acid sequence |
| `lab` | Binary interaction label (0 = non-interacting, 1 = interacting) |
| `affinity` | Raw Kd affinity value (float; in Davis: Kd in nM, threshold ≤30 nM → label=1) |

Label distribution (Davis, n=28,288): positive=2,293 (8.1%), negative=25,995 (91.9%).
Unique drugs: 64. Unique proteins: 442.
