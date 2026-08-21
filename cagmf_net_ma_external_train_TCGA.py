"""
CAGMF-Net 全量 TCGA 训练脚本
在 TCGA 全量数据上训练所有候选模型，保存模型及推理所需的所有组件。

输出：
  saved_models/{name}_model.pth        - 各候选模型的 state_dict
  saved_models/{name}_metadata.json     - 模型架构元数据
  saved_models/scalers.pkl              - 各模态 StandardScaler
  saved_models/cv_weights.pkl           - CV 集成权重 (dict: group_name → np.array)
  saved_models/thresholds.pkl           - 各组合组的阈值 (dict: group_name → (threshold, info))
  saved_models/tcga_oof_predictions.npz - TCGA OOF 预测
  saved_models/tcga_training_metadata.pkl - 训练元数据 (sample_ids, y, le_y, n_classes, dims)
  experiment_config.json                - 运行配置快照

使用方式：
  python cagmf_net_ma_external_train_TCGA.py \
      --tcga_data_dir ./data/tcga \
      --output_dir ./eval_results/tcga_hrd_trained \
      --random_seed 42 --smote
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve
import warnings
import argparse
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.append('.')
from utils import set_seed


# ======================== 模型定义 ========================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.fc(x)))


class Gate(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_k, z_ref):
        g = torch.sigmoid(self.fc(torch.cat([z_k, z_ref], dim=1)))
        return self.dropout(g * z_k)


class CAGMFNet(nn.Module):
    """以临床模态为锚点的门控多模态融合网络"""
    def __init__(self, dims, hidden, n_class, dropout=0.2):
        super().__init__()
        self.anchor = 'clin'
        self.mlp = nn.ModuleDict({k: MLP(dims[k], hidden, dropout) for k in dims})
        self.gate = nn.ModuleDict({k: Gate(hidden, dropout) for k in dims if k != self.anchor})
        self.classifier = nn.Linear(hidden, n_class)

    def forward(self, xs):
        z = {k: self.mlp[k](xs[k]) for k in xs}
        z_ref = z[self.anchor]
        fused = z_ref
        for k in z:
            if k != self.anchor:
                fused = fused + self.gate[k](z[k], z_ref)
        return self.classifier(fused)


class SingleModalMLP(nn.Module):
    """单模态退化为多层感知机"""
    def __init__(self, in_dim, hidden, n_class, dropout=0.2):
        super().__init__()
        self.mlp = MLP(in_dim, hidden, dropout)
        self.classifier = nn.Linear(hidden, n_class)

    def forward(self, x):
        z = self.mlp(x)
        return self.classifier(z)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


# ======================== 常量 ========================
ALL_CANDIDATE_MODELS = [
    ['clin'],
    ['clin', 'snv'],
    ['clin', 'cnv'],
    ['clin', 'mrna'],
    ['clin', 'snv', 'cnv'],
    ['clin', 'snv', 'mrna'],
    ['clin', 'cnv', 'mrna'],
    ['clin', 'snv', 'cnv', 'mrna'],
]

COMBINATION_GROUPS = {
    'Clinical': [0],
    'Clinical+SNV': [1],
    'Clinical+CNA': [2],
    'Clinical+RNA': [3],
    'Clinical+SNV+CNA': [1, 2, 4],
    'Clinical+SNV+RNA': [1, 3, 5],
    'Clinical+CNA+RNA': [2, 3, 6],
    'Clinical+SNV+CNA+RNA': [1, 2, 3, 4, 5, 6, 7],
}

MODALITY_NAME_MAP = {
    'clin': 'Clinical',
    'snv': 'SNV',
    'cnv': 'CNA',
    'mrna': 'RNA'
}


def get_combination_display_name(modalities):
    return '+'.join([MODALITY_NAME_MAP[mod] for mod in modalities])


# ======================== TCGA 数据加载 ========================
def standardize_sample_id(sample_id):
    """标准化样本ID：去除-01、-02等后缀（TCGA格式）"""
    if isinstance(sample_id, str):
        if sample_id.endswith(('-01', '-02', '-03')):
            sample_id = sample_id.rsplit('-', 1)[0]
    return sample_id


def load_tcga_hrd_data(data_dir, return_sample_ids=False):
    """
    加载TCGA HRD数据
    Clinical: AGE, ER, HER2, LN（仅4特征，显式排除STAGE/GRADE）
    """
    clinical_path = os.path.join(data_dir, "TCGA_Clinical_HRD.csv")
    snv_path = os.path.join(data_dir, "tcga_SNV.csv")
    cna_path = os.path.join(data_dir, "tcga_CNV_CX.csv")
    mrna_path = os.path.join(data_dir, "tcga_mRNA.csv")

    print(f"加载TCGA HRD数据:")
    print(f"  临床数据: {clinical_path}")

    clin = pd.read_csv(clinical_path)
    snv = pd.read_csv(snv_path)
    cna = pd.read_csv(cna_path)
    mrna = pd.read_csv(mrna_path)

    def set_index_from_column(df, id_col=None):
        df = df.copy()
        if id_col and id_col in df.columns:
            df = df.set_index(id_col)
        else:
            sample_id_columns = ['SAMPLE_ID', 'Sample_ID', 'sample_id', 'sample', 'Unnamed: 0']
            sample_col = None
            for col in sample_id_columns:
                if col in df.columns:
                    sample_col = col
                    break
            if sample_col is None:
                sample_col = df.columns[0]
            df = df.set_index(sample_col)
        df = df[~df.index.duplicated(keep='first')]
        df.index = df.index.map(standardize_sample_id)
        return df

    def clean_numeric(df):
        df = df.copy()
        bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        df = df.drop(columns=bad)
        return df.apply(pd.to_numeric, errors='coerce').fillna(0)

    clin = set_index_from_column(clin, id_col='patient')

    if 'HRD_label' in clin.columns:
        y = clin['HRD_label'].astype(int)
    else:
        raise ValueError("临床数据中未找到 HRD_label 列")

    clin_features = clin.drop(columns=['HRD_label']).copy()
    for id_col in ['Sample_ID', 'SAMPLE_ID', 'sample_id', 'patient']:
        if id_col in clin_features.columns:
            clin_features = clin_features.drop(columns=id_col)
    for col in clin_features.columns:
        if not pd.api.types.is_numeric_dtype(clin_features[col]):
            clin_features[col] = clin_features[col].fillna('NA')
            clin_features[col] = LabelEncoder().fit_transform(clin_features[col].astype(str))
    clin_feat = clean_numeric(clin_features)

    snv = set_index_from_column(snv)
    cna = set_index_from_column(cna)

    first_col_name = mrna.columns[0]
    if first_col_name in ['GeneSet', 'Unnamed: 0', ''] or 'HALLMARK' in str(mrna.iloc[0, 0]):
        print("  检测到Hallmark格式数据，进行转置...")
        mrna = mrna.set_index(mrna.columns[0])
        mrna = mrna.T
    mrna = set_index_from_column(mrna)

    snv = clean_numeric(snv)
    cna = clean_numeric(cna)
    mrna = clean_numeric(mrna)

    common_samples = sorted(list(
        set(snv.index) & set(cna.index) & set(mrna.index) &
        set(clin_feat.index) & set(y.index)
    ))
    print(f"  匹配样本数: {len(common_samples)}")

    if len(common_samples) == 0:
        raise ValueError("无法找到共同样本")

    snv = snv.loc[common_samples]
    cna = cna.loc[common_samples]
    mrna = mrna.loc[common_samples]
    clin_feat = clin_feat.loc[common_samples]
    y = y.loc[common_samples]

    X_data = {
        "clin": clin_feat.values.astype(np.float32),
        "snv": snv.values.astype(np.float32),
        "cnv": cna.values.astype(np.float32),
        "mrna": mrna.values.astype(np.float32),
    }

    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)
    n_classes = len(np.unique(y_enc))
    feature_dims = {mod: X_data[mod].shape[1] for mod in X_data}

    print(f"  HRD标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  类别数: {n_classes}")
    print(f"  维度: SNV={X_data['snv'].shape}, CNA={X_data['cnv'].shape}, "
          f"mRNA={X_data['mrna'].shape}, Clinical={X_data['clin'].shape}")

    if return_sample_ids:
        return X_data, y_enc, le_y, feature_dims, common_samples
    return X_data, y_enc, le_y, feature_dims


# ======================== 数据增强 ========================
def apply_data_augmentation(X_dict, y, modalities, method, random_state=42):
    """对训练数据进行SMOTE/过采样增强"""
    if method is None:
        return X_dict, y

    X_concat = np.concatenate([X_dict[mod] for mod in modalities], axis=1)

    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state, k_neighbors=min(5, np.sum(y == 1) - 1))
            X_res, y_res = smote.fit_resample(X_concat, y)
        except ImportError:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=random_state)
            X_res, y_res = ros.fit_resample(X_concat, y)
    elif method == 'oversample':
        try:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=random_state)
            X_res, y_res = ros.fit_resample(X_concat, y)
        except ImportError:
            classes, counts = np.unique(y, return_counts=True)
            max_count = counts.max()
            indices = []
            for cls in classes:
                cls_idx = np.where(y == cls)[0]
                if len(cls_idx) < max_count:
                    additional = np.random.choice(cls_idx, max_count - len(cls_idx), replace=True)
                    indices.extend(cls_idx.tolist())
                    indices.extend(additional.tolist())
                else:
                    indices.extend(cls_idx.tolist())
            X_res = X_concat[np.array(indices)]
            y_res = y[np.array(indices)]
    else:
        return X_dict, y

    offset = 0
    X_aug = {}
    for mod in modalities:
        dim = X_dict[mod].shape[1]
        X_aug[mod] = X_res[:, offset:offset + dim]
        offset += dim

    return X_aug, y_res


# ======================== 模型训练 ========================
def train_model_external(X_train_dict, y_train, modalities, dims, n_classes,
                         hidden=64, epochs=50, lr=0.001, device=None, seed=42,
                         val_ratio=0.1, patience=10, loss_type='ce',
                         data_augmentation=None):
    """训练单个CAGMF-Net模型"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    hidden = 32
    lr = 0.001
    max_epochs = 50
    epochs = max_epochs

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=val_ratio, random_state=seed,
        stratify=y_train if len(np.unique(y_train)) > 1 else None
    )

    if data_augmentation:
        X_train_inner = {mod: X_train_dict[mod][train_idx] for mod in modalities}
        y_train_inner = y_train[train_idx]
        X_train_aug, y_train_aug = apply_data_augmentation(
            X_train_inner, y_train_inner, modalities, data_augmentation, random_state=seed
        )
        X_train_use = X_train_aug
        y_train_use = y_train_aug
    else:
        X_train_use = {mod: X_train_dict[mod][train_idx] for mod in modalities}
        y_train_use = y_train[train_idx]

    if len(modalities) == 1:
        mod = modalities[0]
        model = SingleModalMLP(dims[mod], hidden, n_classes)
    else:
        model = CAGMFNet(dims, hidden, n_classes)

    model = model.to(device)

    train_tensors = [torch.tensor(X_train_use[mod], dtype=torch.float32) for mod in modalities]
    train_dataset = TensorDataset(*train_tensors, torch.tensor(y_train_use, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_tensors = [torch.tensor(X_train_dict[mod][val_idx], dtype=torch.float32) for mod in modalities]
    val_dataset = TensorDataset(*val_tensors, torch.tensor(y_train[val_idx], dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = FocalLoss(alpha=0.25, gamma=2.0) if loss_type == 'focal' else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_data in train_loader:
            batch_X_dict = {}
            for i_mod, mod in enumerate(modalities):
                batch_X_dict[mod] = batch_data[i_mod].to(device)
            batch_y = batch_data[-1].to(device)

            optimizer.zero_grad()
            if len(modalities) == 1:
                outputs = model(batch_X_dict[modalities[0]])
            else:
                outputs = model(batch_X_dict)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_X_dict = {}
                for i_mod, mod in enumerate(modalities):
                    batch_X_dict[mod] = batch_data[i_mod].to(device)
                batch_y = batch_data[-1].to(device)
                if len(modalities) == 1:
                    outputs = model(batch_X_dict[modalities[0]])
                else:
                    outputs = model(batch_X_dict)
                val_loss += criterion(outputs, batch_y).item()

        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


# ======================== 模型预测 ========================
def predict_model(model, X_dict, modalities, device=None):
    """用模型预测概率"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    tensors = [torch.tensor(X_dict[mod], dtype=torch.float32) for mod in modalities]
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch_data in loader:
            if len(modalities) == 1:
                batch_X = batch_data[0].to(device)
                outputs = model(batch_X)
            else:
                batch_X_dict = {}
                for i, mod in enumerate(modalities):
                    batch_X_dict[mod] = batch_data[i].to(device)
                outputs = model(batch_X_dict)
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


# ======================== CV 权重计算 ========================
def compute_cv_weights(oof_predictions, y_train, model_indices, n_classes, criterion='mse'):
    """为指定组的模型计算CV权重"""
    n_models_group = len(model_indices)
    n_samples = len(y_train)
    oof_group = oof_predictions[model_indices, :, :]

    if criterion == 'mse':
        y_onehot = np.eye(n_classes)[y_train]
        errors = oof_group - y_onehot
        errors_flat = errors.reshape(n_models_group, -1)
        Q = (errors_flat @ errors_flat.T) / (n_samples * n_classes)
        Q = Q + 1e-8 * np.eye(n_models_group)

        try:
            from cvxopt import matrix, solvers
            P = matrix(Q.astype(float))
            q = matrix(np.zeros(n_models_group))
            G = matrix(-np.eye(n_models_group))
            h = matrix(np.zeros(n_models_group))
            A = matrix(np.ones((1, n_models_group)).astype(float))
            b = matrix(1.0)
            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)
            if solution['status'] == 'optimal':
                weights = np.array(solution['x']).flatten()
            else:
                weights = np.ones(n_models_group) / n_models_group
        except:
            from scipy.optimize import minimize
            def objective(w):
                return 0.5 * w @ Q @ w
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in range(n_models_group)]
            result = minimize(objective, np.ones(n_models_group) / n_models_group,
                              method='SLSQP', bounds=bounds, constraints=constraints,
                              options={'maxiter': 1000, 'disp': False})
            weights = result.x
    else:
        from scipy.optimize import minimize
        def objective(w):
            ensemble_probs = np.average(oof_group, axis=0, weights=w)
            ensemble_probs = np.clip(ensemble_probs, 1e-15, 1 - 1e-15)
            if criterion == 'ce':
                return -np.mean(np.log(ensemble_probs[np.arange(n_samples), y_train]))
            else:
                pt = ensemble_probs[np.arange(n_samples), y_train]
                return np.mean(0.25 * (1 - pt) ** 2.0 * (-np.log(pt)))
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_models_group)]
        result = minimize(objective, np.ones(n_models_group) / n_models_group,
                          method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'disp': False})
        weights = result.x

    weights = np.maximum(weights, 0)
    weights = weights / np.sum(weights)
    return weights


# ======================== 阈值搜索 ========================
def find_youden_threshold(y_true, y_prob_class1):
    """寻找最大化Youden指数的分类阈值"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_class1)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    return thresholds[best_idx], youden_j[best_idx]


def find_f2_threshold(y_true, y_prob_class1, beta=2.0):
    """寻找最大化F-beta分数的分类阈值"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_class1)
    valid = (thresholds > 0.01) & (thresholds < 0.99)
    thresholds = thresholds[valid]
    tpr = tpr[valid]
    fpr = fpr[valid]

    if len(thresholds) == 0:
        return 0.5, 0.0

    n_neg = np.sum(y_true == 0)
    n_pos = np.sum(y_true == 1)

    best_threshold = 0.5
    best_fbeta = -1.0
    for i in range(len(thresholds)):
        tp = tpr[i] * n_pos
        fp = fpr[i] * n_neg
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr[i]
        beta2 = beta * beta
        if precision + recall > 0:
            fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        else:
            fbeta = 0
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = thresholds[i]

    return best_threshold, best_fbeta


def find_sensitivity_threshold(y_true, y_prob_class1, target_sensitivity=0.85):
    """找到使sensitivity >= target的最保守阈值"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_class1)
    valid = (thresholds > 0.0) & (thresholds < 1.0)
    thresholds = thresholds[valid]
    tpr = tpr[valid]
    fpr = fpr[valid]

    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0

    for i in range(len(thresholds)):
        if tpr[i] >= target_sensitivity:
            return thresholds[i], tpr[i], 1.0 - fpr[i]

    best_idx = np.argmax(tpr)
    return thresholds[best_idx], tpr[best_idx], 1.0 - fpr[best_idx]


def get_threshold(y_true, y_prob_class1, method='argmax', target_sensitivity=0.85):
    """统一阈值计算接口"""
    if method == 'argmax':
        return 0.5, {}
    elif method == 'youden':
        th, j = find_youden_threshold(y_true, y_prob_class1)
        return th, {'youden_j': float(j)}
    elif method == 'f2':
        th, f2 = find_f2_threshold(y_true, y_prob_class1)
        return th, {'f2_score': float(f2)}
    elif method == 'sensitivity':
        th, sens, spec = find_sensitivity_threshold(y_true, y_prob_class1, target_sensitivity)
        return th, {'actual_sensitivity': float(sens), 'actual_specificity': float(spec)}
    else:
        return 0.5, {}


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(description='CAGMF-Net 全量 TCGA 训练')
    parser.add_argument('--tcga_data_dir', type=str, default='./data/tcga',
                        help='TCGA 数据目录')
    parser.add_argument('--output_dir', type=str, default='./eval_results/tcga_hrd_trained',
                        help='模型保存目录')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备，默认自动选择')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='TCGA交叉验证折数')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal'],
                        help='损失函数: ce 或 focal')
    parser.add_argument('--cv_criterion', type=str, default='mse', choices=['mse', 'ce', 'focal'],
                        help='CV集成准则')
    parser.add_argument('--threshold_method', type=str, default='argmax',
                        choices=['argmax', 'youden', 'f2', 'sensitivity'],
                        help='阈值选取方法')
    parser.add_argument('--target_sensitivity', type=float, default=0.85,
                        help='目标灵敏度 (仅 sensitivity 阈值方法)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--smote', action='store_true',
                        help='使用SMOTE数据增强')
    parser.add_argument('--oversample', action='store_true',
                        help='使用随机过采样数据增强')
    args = parser.parse_args()

    data_augmentation = None
    if args.smote:
        data_augmentation = 'smote'
    elif args.oversample:
        data_augmentation = 'oversample'

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    print("=" * 80)
    print("CAGMF-Net 全量 TCGA 训练")
    print(f"TCGA 数据: {args.tcga_data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"随机种子: {args.random_seed}")
    print(f"损失函数: {'Focal Loss' if args.loss == 'focal' else '交叉熵'}")
    if data_augmentation:
        print(f"数据增强: {data_augmentation}")
    print(f"阈值方法: {args.threshold_method}", end='')
    if args.threshold_method == 'sensitivity':
        print(f" (目标灵敏度={args.target_sensitivity})")
    else:
        print()
    print("=" * 80)

    # ========== 1. 加载 TCGA 数据 ==========
    print("\n1. 加载 TCGA 数据...")
    X_tcga, y_tcga, le_y, tcga_feature_dims, tcga_sample_ids = load_tcga_hrd_data(
        args.tcga_data_dir, return_sample_ids=True
    )
    n_classes = len(np.unique(y_tcga))
    print(f"TCGA 总样本数: {len(y_tcga)}, 类别数: {n_classes}")

    # ========== 2. 全量标准化 TCGA ==========
    print("\n2. 全量标准化 TCGA 数据...")
    scalers = {}
    X_tcga_std = {}
    for mod in X_tcga.keys():
        scaler = StandardScaler()
        X_tcga_std[mod] = scaler.fit_transform(X_tcga[mod])
        scalers[mod] = scaler

    # ========== 3. K折CV → OOF预测 + CV权重 ==========
    print(f"\n3. {args.cv_folds}折交叉验证计算OOF预测...")
    K = args.cv_folds
    n_samples = len(y_tcga)
    n_models = len(ALL_CANDIDATE_MODELS)
    oof_predictions = np.zeros((n_models, n_samples, n_classes))

    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=args.random_seed)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(y_tcga, y_tcga)):
        print(f"  Fold {fold_idx + 1}/{K}")

        train_X_fold = {}
        val_X_fold = {}
        for mod in X_tcga_std.keys():
            fold_scaler = StandardScaler()
            X_train_fold_data = X_tcga_std[mod][train_idx]
            fold_scaler.fit(X_train_fold_data)
            train_X_fold[mod] = fold_scaler.transform(X_train_fold_data)
            val_X_fold[mod] = fold_scaler.transform(X_tcga_std[mod][val_idx])

        y_train_fold = y_tcga[train_idx]

        for m, modalities in enumerate(ALL_CANDIDATE_MODELS):
            dims = {mod: tcga_feature_dims[mod] for mod in modalities}
            X_train_sub = {mod: train_X_fold[mod] for mod in modalities}

            model = train_model_external(
                X_train_sub, y_train_fold, modalities, dims, n_classes,
                device=device, seed=args.random_seed + fold_idx * 10 + m,
                loss_type=args.loss, data_augmentation=data_augmentation
            )

            X_val_sub = {mod: val_X_fold[mod] for mod in modalities}
            probs = predict_model(model, X_val_sub, modalities, device)
            oof_predictions[m, val_idx, :] = probs

    print("  CV完成")

    # 计算CV权重
    print("  计算CV权重...")
    cv_weights = {}
    for group_name, model_indices in COMBINATION_GROUPS.items():
        if len(model_indices) == 0:
            continue
        weights = compute_cv_weights(oof_predictions, y_tcga, model_indices, n_classes,
                                    criterion=args.cv_criterion)
        cv_weights[group_name] = weights
        model_names = [get_combination_display_name(ALL_CANDIDATE_MODELS[i]) for i in model_indices]
        weight_str = ', '.join([f'{n}={w:.4f}' for n, w in zip(model_names, weights)])
        print(f"    {group_name}: {weight_str}")

    # ========== 4. 训练最终模型（全量TCGA） ==========
    print("\n4. 训练最终模型（全量TCGA）...")
    final_models = []
    for m, modalities in enumerate(ALL_CANDIDATE_MODELS):
        dims = {mod: tcga_feature_dims[mod] for mod in modalities}
        X_train_sub = {mod: X_tcga_std[mod] for mod in modalities}

        model = train_model_external(
            X_train_sub, y_tcga, modalities, dims, n_classes,
            device=device, seed=args.random_seed + m,
            loss_type=args.loss, data_augmentation=data_augmentation
        )
        final_models.append(model)
        model_name = get_combination_display_name(modalities)
        print(f"  {model_name} 训练完成")

    # ========== 5. TCGA预测 + 阈值计算 ==========
    print(f"\n5. 计算TCGA预测及阈值 (方法={args.threshold_method})...")
    tcga_candidate_probs = []
    for m, modalities in enumerate(ALL_CANDIDATE_MODELS):
        X_sub = {mod: X_tcga_std[mod] for mod in modalities}
        probs = predict_model(final_models[m], X_sub, modalities, device)
        tcga_candidate_probs.append(probs)
    tcga_candidate_probs = np.array(tcga_candidate_probs)

    thresholds = {}
    threshold_infos = {}
    for group_name, model_indices in COMBINATION_GROUPS.items():
        if len(model_indices) == 0:
            continue

        weights = cv_weights[group_name]
        oof_group = oof_predictions[model_indices, :, :]

        tcga_oof_ensemble = np.zeros((n_samples, n_classes))
        for w, probs in zip(weights, oof_group):
            tcga_oof_ensemble += w * probs

        th, th_info = get_threshold(
            y_tcga, tcga_oof_ensemble[:, 1],
            method=args.threshold_method,
            target_sensitivity=args.target_sensitivity
        )
        thresholds[group_name] = float(th)
        threshold_infos[group_name] = th_info

        if args.threshold_method == 'youden':
            print(f"  {group_name}: Youden阈值={th:.4f}, J={th_info.get('youden_j', 0):.4f}")
        elif args.threshold_method == 'f2':
            print(f"  {group_name}: F2阈值={th:.4f}, F2={th_info.get('f2_score', 0):.4f}")
        elif args.threshold_method == 'sensitivity':
            print(f"  {group_name}: Sens阈值={th:.4f}, Sens={th_info.get('actual_sensitivity', 0):.4f}, "
                  f"Spec={th_info.get('actual_specificity', 0):.4f}")
        else:
            print(f"  {group_name}: argmax阈值=0.5")

    # ========== 6. 保存所有训练产物 ==========
    print("\n6. 保存模型和训练产物...")
    models_dir = os.path.join(args.output_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)

    # 保存各模型 state_dict + 元数据
    for m, modalities in enumerate(ALL_CANDIDATE_MODELS):
        model_name = get_combination_display_name(modalities)
        safe_name = model_name.replace('+', '_')
        model_path = os.path.join(models_dir, f"{safe_name}_model.pth")
        torch.save(final_models[m].state_dict(), model_path)
        metadata = {
            'modalities': modalities,
            'dims': {mod: tcga_feature_dims[mod] for mod in modalities},
            'n_classes': n_classes,
            'hidden': 32,
            'model_type': 'SingleModalMLP' if len(modalities) == 1 else 'CAGMFNet',
        }
        with open(model_path.replace('.pth', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  {model_name}: {safe_name}_model.pth")

    # 保存 scalers
    scaler_path = os.path.join(models_dir, "scalers.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"  scalers.pkl")

    # 保存 CV 权重
    cv_weights_path = os.path.join(models_dir, "cv_weights.pkl")
    with open(cv_weights_path, 'wb') as f:
        pickle.dump(cv_weights, f)
    print(f"  cv_weights.pkl")

    # 保存阈值
    thresholds_path = os.path.join(models_dir, "thresholds.pkl")
    with open(thresholds_path, 'wb') as f:
        pickle.dump({'thresholds': thresholds, 'threshold_infos': threshold_infos,
                      'threshold_method': args.threshold_method}, f)
    print(f"  thresholds.pkl")

    # 保存 OOF 预测
    oof_path = os.path.join(models_dir, "tcga_oof_predictions.npz")
    np.savez(oof_path, oof_predictions=oof_predictions, y_tcga=y_tcga,
             candidate_models=np.array(ALL_CANDIDATE_MODELS, dtype=object),
             combination_groups=json.dumps({k: v for k, v in COMBINATION_GROUPS.items()}))
    print(f"  tcga_oof_predictions.npz")

    # 保存训练元数据
    metadata_path = os.path.join(models_dir, "tcga_training_metadata.pkl")
    training_metadata = {
        'tcga_sample_ids': tcga_sample_ids,
        'y_tcga': y_tcga,
        'le_y': le_y,
        'n_classes': n_classes,
        'feature_dims': tcga_feature_dims,
        'random_seed': args.random_seed,
        'loss_type': args.loss,
        'cv_criterion': args.cv_criterion,
        'data_augmentation': data_augmentation,
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(training_metadata, f)
    print(f"  tcga_training_metadata.pkl")

    # 保存运行配置
    config = {
        'tcga_data_dir': args.tcga_data_dir,
        'cv_folds': args.cv_folds,
        'loss_type': args.loss,
        'cv_criterion': args.cv_criterion,
        'threshold_method': args.threshold_method,
        'target_sensitivity': args.target_sensitivity,
        'random_seed': args.random_seed,
        'data_augmentation': data_augmentation,
        'n_classes': n_classes,
        'n_tcga_samples': len(y_tcga),
        'candidate_models': [get_combination_display_name(m) for m in ALL_CANDIDATE_MODELS],
        'combination_groups': {k: [get_combination_display_name(ALL_CANDIDATE_MODELS[i])
                                    for i in v] for k, v in COMBINATION_GROUPS.items()},
    }
    config_path = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  experiment_config.json")

    print(f"\n{'=' * 80}")
    print("TCGA 训练完成!")
    print(f"所有产物已保存至: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
