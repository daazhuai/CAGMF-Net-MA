# CAGMF-Net-MA
This repository provides the official implementation of **CAGMF-Net-MA** for predicting Homologous Recombination Deficiency (HRD) status from multi-omics data.
**CAGMF-Net** (Clinical-Anchored Gated Multi-modal Fusion Network) integrates clinical features, SNV (single nucleotide variants), CNV (copy number variations), and mRNA expression data. It uses the clinical modality as an anchor and employs gating mechanisms to selectively fuse complementary information from other modalities.

**CAGMF-Net-MA** extends CAGMF-Net with a cross-validation-optimized model averaging (MA) strategy:

1. Train 8 candidate models with different modality combinations (from clinical-only to full four-modality fusion).
2. Obtain out-of-fold (OOF) predictions for each candidate via 5-fold CV.
3. Optimize ensemble weights by minimizing a CV criterion, subject to sum-to-one and non-negativity constraints.
4. Average test-set predictions from constituent candidates using the optimized weights.



## Datasets

- **TCGA-BRCA**: Used for training and internal cross-validation (5-fold CV, 100 random splits).
- **METABRIC, MyBrCa, SCAN-B**: Used for independent external validation.

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
python cagmf_net_ma_cv_addeval_HRD.py
```

### 2. External validation on METABRIC

```bash
python cagmf_net_ma_external_train_TCGA.py

python cagmf_net_ma_external_validation_METABRIC.py
python cagmf_net_ma_external_validation_MyBrCa.py
ython cagmf_net_ma_external_validation_SCAN-B.py
```

### 3. Baselines

```bash
# Traditional ML baselines
python baseline_compare/ml_baseline_HRD.py \
      --data_dir ./data/tcga \
      --output_dir ./eval_results/tcga_hrd_ml_baselines \
      --n_splits 100 \
      --smote

# Single CAGMF-Net (no model averaging)
python baseline_compare/cagmf_net_baseline_HRD.py \
      --main_exp_dir ./eval_results/tcga_hrd_cagmf_ma \
      --output_dir ./eval_results/tcga_hrd_cagmf_single

# AIC/BIC model selection from scratch
ython baseline_compare/model_choose_baseline_HRD.py \
      --main_exp_dir ./eval_results/tcga_hrd_cagmf_ma \
      --output_dir ./eval_results/tcga_hrd_model_choose \
      --method both

# Equal-weight / SAIC / SBIC ensemble baselines
python baseline_compare/model_average_baseline_HRD.py \
      --main_exp_dir ./eval_results/tcga_hrd_cagmf_ma \
      --output_dir ./eval_results/tcga_hrd_model_average

```

## Project Structure

```
├── utils.py                          # Base setting
├── cagmf_net_ma_HRD.py               # Main experiment: CAGMF-Net with model averaging
├── cagmf_net_ma_external_train_TCGA.py  # train on full TCGA
├── cagmf_net_ma_external_validation_METABRIC.py  # External validation on METABRIC
├── cagmf_net_ma_external_validation_MyBrCa.py  # External validation on MyBrCa
├── cagmf_net_ma_external_validation_SCAN-B.py  # External validation on SCAN-B
├── baseline_compare/
│   ├── ml_baseline_HRD.py            # Traditional ML baselines
│   ├── equal_weight_baseline_HRD.py  # Single-model CAGMF-Net baseline
│   ├── cagmf_net_baseline_HRD.py     # AIC/BIC model selection# Equal weight / SAIC / SBIC baselines
│   └── model_choose_baseline_HRD.py  # Equal weight / SAIC / SBIC baselines
├── data/                             # Dataset directory
└── eval_results/                     # Saved models and evaluation results
```

## Citation

If you use this code in your research, please cite the corresponding paper.
