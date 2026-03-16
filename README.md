# Multi-Omics Fusion Network for Breast Cancer Subtype Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Abstract

Breast cancer is a highly heterogeneous malignancy, and accurate molecular subtyping based on the PAM50 standard is crucial for clinical treatment decisions and prognosis prediction. While deep learning methods have been widely applied to biomedical data analysis tasks, existing approaches often rely on single data modalities, failing to capture the complex biological landscape involving multi-layered molecular mechanisms across genomics, transcriptomics, and other omics levels. The increasing availability of multi-omics data offers potential for comprehensive tumor characterization, yet effectively integrating multi-omics data with clinical information remains a core challenge.

To address these challenges, we propose **ClinicalAnchor-Gated Multi-Omics Fusion Network (CAGMF-Net)**, a deep learning framework with gated fusion mechanisms designed to integrate four modalities—clinical features, single nucleotide variants (SNV), copy number alterations (CNA), and gene expression (RNA)—for accurate PAM50 subtype prediction. The framework employs modality-specific encoders to learn latent representations and introduces a gated fusion mechanism anchored on clinical features for adaptive information integration.

Furthermore, to enhance model robustness and generalization, we introduce **CAGMF-Net with Model Averaging (CAGMF-Net-MA)**. By constructing candidate models with different modality combinations and optimizing weights through cross-validation, this strategy achieves superior predictive performance.

**Key Results:**

- CAGMF-Net significantly outperforms traditional machine learning methods and single-modality approaches
- The cross-validation-based model averaging strategy achieves optimal performance among various model selection and averaging methods
- Progressive addition of omics modalities (CNA → SNV → RNA) to clinical features steadily improves classification performance, confirming that each data type contributes unique information
- Strong generalization performance validated on external TCGA dataset

---

## Quick Start
```
import pandas as pd
import numpy as np
from predict import CancerPredictor

# Line 1: Load model
predictor = CancerPredictor("/root/METAtrain/saved_models_clin_cnv_snv_mrna")

# Line 2: Prepare data (Example with random data)
clin_data = pd.DataFrame(np.random.randn(5, 6))     # Clinical: 5 samples x 6 features
cnv_data = pd.DataFrame(np.random.randn(5, 7))      # CNV: 5 samples x 7 features
snv_data = pd.DataFrame(np.random.randint(0, 2, (5, 15)))  # SNV: 5 samples x 15 features (binary)
mrna_data = pd.DataFrame(np.random.randn(5, 50))    # mRNA: 5 samples x 50 features

# Line 3: Get predictions
labels, probabilities = predictor.predict(clin_data, cnv_data, snv_data, mrna_data)

# View results
print("Predicted labels:", labels)
print("Prediction probabilities:\n", probabilities)
```


## Code Structure and Documentation

### 1. `CAGMF-Net.py` - Core Model Architecture and Training

This script implements the complete CAGMF-Net framework, training the full four-modality model and saving parameters for subtype prediction.

**Key Features:**

- MultiOmicNet architecture with modality-specific MLP encoders
- Clinical-anchored gated fusion mechanism
- Complete data preprocessing pipeline for METABRIC dataset
- Model training, evaluation, and saving functionality

**Main Configurable Parameters:**

```python
hidden = 128           # Hidden dimension size for all MLP encoders
epochs = 50            # Number of training epochs
batch_size = 32        # Batch size for DataLoader
```

**Usage:**

```bash
python CAGMF-Net.py
```

---

### 2. `Train_Internal_Model.py` - Submodel Training for Internal Evaluation

This script performs 100 random train-validation splits and trains all possible submodel combinations (all non-empty subsets of the four modalities), saving all trained models for subsequent evaluation.

**Key Features:**

- Generates all 15 possible modality combinations (2⁴ - 1 = 15)
- Performs 100 independent train-validation splits
- Caches all trained models for reuse in multiple evaluation scripts
- Tracks training statistics and progress

**Main Configurable Parameters:**

```python
n_runs = 100           # Number of random data splits
cache_dir = "model_cache"  # Directory for saving trained models
epochs = 50            # Training epochs per model
```

**Usage:**

```bash
python Train_Internal_Model.py
```

---

### 3. `Internal_Eval_ML.py` - Machine Learning Baseline Evaluation

This script evaluates four traditional machine learning models across the 100 random splits, serving as baseline comparisons for the deep learning approaches.

**Evaluated Models:**

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

**Key Features:**

- Evaluates all 4 models × 8 modality combinations × 100 splits
- Computes 9 metrics including Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss, MSE, MAE
- Generates comprehensive summary tables and comparison files

**Main Configurable Parameters:**

```python
n_runs = 100           # Number of random data splits
model_names = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM']  # Models to evaluate
test_size = 0.2        # Validation set proportion
```

**Usage:**

```bash
python Internal_Eval_ML.py
```

---

### 4. `Internal_Eval_DL.py` - Base CAGMF-Net Evaluation

This script evaluates the base CAGMF-Net architecture (single network, no ensemble) across the 100 random splits for all 8 modality combinations.

**Key Features:**

- Loads pre-trained models from model_cache
- Evaluates all 8 modality combinations × 100 splits
- Computes comprehensive metrics including newly added Log Loss, MSE, MAE
- Saves detailed predictions and evaluation results

**Main Configurable Parameters:**

```python
n_runs = 100           # Number of random data splits
model_cache_dir = "/root/METAtrain/model_cache"  # Directory with pre-trained models
test_size = 0.2        # Validation set proportion
```

**Usage:**

```bash
python Internal_Eval_DL.py
```

---

### 5. `Internal_Eval_MC.py` - Model Selection Evaluation (AIC/BIC)

This script evaluates AIC and BIC-based model selection strategies, choosing the optimal submodel for each modality combination based on information criteria.

**Key Features:**

- For each modality combination, selects the best submodel using AIC or BIC
- Evaluates selected models on validation sets across 100 splits
- Records which submodels are selected most frequently
- Generates comprehensive comparison tables

**Main Configurable Parameters:**

```python
n_runs = 100           # Number of random data splits
model_cache_dir = "/root/METAtrain/model_cache"  # Directory with pre-trained models
test_size = 0.2        # Validation set proportion
```

**Usage:**

```bash
python Internal_Eval_MC.py
```

---

### 6. `Internal_Eval_MA.py` - Model Averaging Evaluation (Equal, SAIC, SBIC, CV)

This script evaluates four model averaging strategies for each modality combination, comparing their performance across 100 random splits.

**Weighting Methods:**

- **CV**: Cross-validation-based quadratic programming optimization
- **AIC**: Smoothed AIC weights based on training set performance
- **BIC**: Smoothed BIC weights based on training set performance  
- **Equal**: Simple average (1/M) as baseline

**Key Features:**

- Computes weights for all submodels within each modality combination
- Evaluates weighted ensemble predictions on validation sets
- Performs 5-fold cross-validation within training set for CV weights
- Comprehensive comparison across all four methods

**Main Configurable Parameters:**

```python
n_runs = 100           # Number of random data splits
K = 5                  # Number of CV folds for weight optimization
model_cache_dir = "/root/METAtrain/model_cache"  # Directory with pre-trained models
```

**Usage:**

```bash
python Internal_Eval_MA.py
```

---

### 7. `Train_External_Model.py` - Training Models for External Validation

This script trains CAGMF-Net-MA models on the complete METABRIC dataset for all 8 modality combinations, saving them for external validation on TCGA.

**Key Features:**

- Trains all submodels on the full dataset
- Computes optimal CV weights via 5-fold cross-validation
- Saves complete model ensembles with configurations, scalers, and encoders
- Includes clinical feature encoders for proper categorical variable handling

**Main Configurable Parameters:**

```python
K = 5                  # Number of CV folds for weight optimization
epochs = 50            # Training epochs on full dataset
hidden_dim = 128       # Hidden dimension size
```

**Usage:**

```bash
python Train_External_Model.py
```

**Output Structure:**

- `saved_models_clin/` - Clinical-only model
- `saved_models_clin_/` - Clinical + CNA ensemble
- `saved_models_clin_snv/` - Clinical + SNV ensemble
- `saved_models_clin_mrna/` - Clinical + RNA ensemble
- `saved_models_clin_cnv_snv/` - Clinical + CNA + SNV ensemble
- `saved_models_clin_cnv_mrna/` - Clinical + CNA + RNA ensemble
- `saved_models_clin_snv_mrna/` - Clinical + SNV + RNA ensemble
- `saved_models_clin_cnv_snv_mrna/` - Full four-modality ensemble

---

### 8. `External_Eval.py` - External Validation on TCGA

This script evaluates all 8 trained modality combinations on the external TCGA dataset, comparing their generalization performance.

**Key Features:**

- Loads pre-trained ensembles for all modality combinations
- Preprocesses TCGA data using training-derived encoders
- Handles feature dimension mismatches and unseen categories
- Computes comprehensive metrics including PR-AUC, ROC-AUC, F1, and newly added metrics
- Generates detailed prediction reports and confusion matrices

**Main Configurable Parameters:**

```python
ensemble_dirs = [      # Directories for all 8 modality combinations
    "saved_models_clin",
    "saved_models_clin_",
    ...
]
external_data_dir = "/root/METAtrain/data/external_validation/"  # TCGA data directory
```

**Usage:**

```bash
python External_Eval.py
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/CAGMF-Net.git
cd CAGMF-Net

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**

- Python 3.8+
- PyTorch 1.9+
- scikit-learn
- pandas
- numpy
- xgboost
- lightgbm
- tqdm
- scipy (optional, for quadratic programming)
- cvxopt (optional, for quadratic programming)

---

## Dataset Preparation

### METABRIC (Training)

Place the following files in `./dataset/METABRIC/`:

- `METABRIC_Clinical.csv`
- `METABRIC_CNA.csv`
- `METABRIC_SNV.csv`
- `METABRIC_RNA.csv`

### TCGA (External Validation)

Place the following files in your external validation directory (e.g., `./dataset/TCGA/`):

- `TCGA_Clinical.csv`
- `TCGA_CNA.csv`
- `TCGA_SNV.csv`
- `TCGA_RNA.csv`

---

## Complete Workflow

### Step 1: Train Internal Evaluation Models

```bash
python Train_Internal_Model.py
```

This generates all submodels for 100 random splits in `model_cache/`.

### Step 2: Run Internal Evaluations

```bash
python Internal_Eval_ML.py      # Machine learning baselines
python Internal_Eval_DL.py      # Base CAGMF-Net
python Internal_Eval_MC.py      # AIC/BIC model selection
python Internal_Eval_MA.py      # Model averaging (CV, AIC, BIC, Equal)
```

### Step 3: Train External Validation Models

```bash
python Train_External_Model.py
```

This generates full ensembles for all modality combinations.

### Step 4: Run External Validation

```bash
python External_Eval.py
```

---

## Results and Outputs

Each evaluation script generates structured outputs:

- **Predictions**: Detailed prediction probabilities and labels per run
- **Summaries**: CSV tables with mean±std [min,max] for all metrics
- **Detailed results**: JSON files with complete statistics
- **Comparison tables**: Best model comparisons across methods

All outputs are organized in dedicated directories:

- `./ml_predictions/`, `./ml_evaluations/`, `./ml_summary/` for ML baselines
- `./base_predictions/`, `./dl_evaluations/` for DL evaluations
- `./ma_weights/`, `./ma_predict_result/`, `./eva_result/` for model averaging
- `./aic_bic_predict_results/`, `./aic_bic_eva_results/` for model selection

---

## Citation

If you use this code in your research, please cite:

```
@article{cagmfnet2024,
  title={CAGMF-Net: ClinicalAnchor-Gated Multi-Omics Fusion Network for Breast Cancer Subtype Classification},
  author={Your Name},
  journal={},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
