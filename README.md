# CAGMF-Net-MA-CAGMF-Net-with-CV-Model-Averaging-for-HRD-Prediction
This repository provides the official implementation of **CAGMF-Net-MA** for predicting Homologous Recombination Deficiency (HRD) status from multi-omics data.
**CAGMF-Net** (Clinical-Anchored Gated Multi-modal Fusion Network) integrates clinical features, SNV (single nucleotide variants), CNV (copy number variations), and mRNA expression data. It uses the clinical modality as an anchor and employs gating mechanisms to selectively fuse complementary information from other modalities.

**CAGMF-Net-MA** extends CAGMF-Net with a cross-validation-optimized model averaging (MA) strategy:

1. Train 8 candidate models with different modality combinations (from clinical-only to full four-modality fusion).
2. Obtain out-of-fold (OOF) predictions for each candidate via 5-fold CV.
3. Optimize ensemble weights by minimizing a CV criterion, subject to sum-to-one and non-negativity constraints.
4. Average test-set predictions from constituent candidates using the optimized weights.



## Datasets

- **TCGA-BRCA**: Used for training and internal cross-validation (5-fold CV, 100 random splits).
- **METABRIC**: Used for independent external validation.

Each dataset includes:

| Modality | File                         |
| -------- | ---------------------------- |
| Clinical | `*_Clinical_HRD.csv`         |
| SNV      | `*_SNV.csv`                  |
| CNA      | `*_CNV_CX.csv` / `*_CNA.csv` |
| mRNA     | `*_mRNA.csv`                 |

## Environment

- Python >= 3.8
- PyTorch >= 1.10
- scikit-learn
- numpy, pandas, tqdm
- xgboost, lightgbm (optional, for ML baselines)

Install dependencies:

```bash
pip install torch numpy pandas scikit-learn tqdm xgboost lightgbm
```

## Usage

### 1. Train and evaluate CAGMF-Net-MA

```bash
python cagmf_net_ma_HRD.py \
    --data_dir ./data/tcga \
    --output_dir ./eval_results/tcga_hrd_cagmf_ma \
    --n_splits 100 \
    --smote \
    --youden
```

Key arguments:

- `--smote`: Apply SMOTE for class balancing
- `--oversample`: Apply random oversampling (alternative to SMOTE)
- `--youden`: Use Youden index for threshold selection
- `--loss {ce,focal}`: Loss function (cross-entropy or focal loss)
- `--cv_criterion {mse,ce,focal}`: Criterion for CV weight optimization

### 2. External validation on METABRIC

```bash
python cagmf_net_ma_HRD_external_validation.py \
    --metabric_dir ./data/metabric \
    --model_dir ./eval_results/tcga_hrd_cagmf_ma \
    --n_seeds 100
```

### 3. Baselines

```bash
# Single CAGMF-Net (no model averaging)
python baseline_compare/cagmf_net_baseline_HRD.py --data_dir ./data/tcga

# Traditional ML baselines (LR, RF, XGBoost, LightGBM, SVM, Lasso)
python baseline_compare/ml_baseline_HRD.py --data_dir ./data/tcga

# Equal-weight / SAIC / SBIC ensemble baselines
python baseline_compare/equal_weight_baseline_HRD.py --data_dir ./data/tcga

# SAIC/SBIC model selection from scratch
python baseline_compare/model_choose_baseline_HRD.py --data_dir ./data/tcga
```

## Project Structure

```
├── utils.py                          # Evaluation metrics and seed setting
├── cagmf_net_ma_HRD.py               # Main experiment: CAGMF-Net with model averaging
├── cagmf_net_ma_HRD_external_validation.py  # External validation on METABRIC
├── baseline_compare/
│   ├── cagmf_net_baseline_HRD.py     # Single-model CAGMF-Net baseline
│   ├── equal_weight_baseline_HRD.py  # Equal weight / SAIC / SBIC baselines
│   ├── ml_baseline_HRD.py            # Traditional ML baselines
│   └── model_choose_baseline_HRD.py  # SAIC/SBIC model selection
├── data/                             # Dataset directory
│   ├── tcga/
│   └── metabric/
└── results/                          # Saved models and evaluation results
    ├── inner/                        # CAGMF-Net-MA Internal results
    └── external/                     # CAGMF-Net-MA External validation results
```

## Citation

If you use this code in your research, please cite the corresponding paper.
