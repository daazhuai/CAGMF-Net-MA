"""
CAGMF-Net 模型平均组合评估 - TCGA HRD数据专用版
使用TCGA_Clinical_HRD.csv，以patient为ID，HRD_label为标签（二分类）
评估8种模型平均组合，支持SMOTE/Oversample数据增强和Youden阈值选取
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_curve, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import warnings
import json
import pickle
from tqdm import tqdm
import argparse

warnings.filterwarnings('ignore')

sys.path.append('.')
from utils import evaluate_predictions, set_seed


# ======================== 模型定义（CAGMF-Net 架构） ========================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden)

    def forward(self, x):
        return F.relu(self.fc(x))


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, z_k, z_ref):
        g = torch.sigmoid(self.fc(torch.cat([z_k, z_ref], dim=1)))
        return g * z_k


class CAGMFNet(nn.Module):
    """以临床模态为锚点的门控多模态融合网络"""
    def __init__(self, dims, hidden, n_class):
        super().__init__()
        # 始终以 clin 为锚点
        self.anchor = 'clin'
        self.mlp = nn.ModuleDict({k: MLP(dims[k], hidden) for k in dims})
        self.gate = nn.ModuleDict({k: Gate(hidden) for k in dims if k != self.anchor})
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
    def __init__(self, in_dim, hidden, n_class):
        super().__init__()
        self.mlp = MLP(in_dim, hidden)
        self.classifier = nn.Linear(hidden, n_class)

    def forward(self, x):
        z = self.mlp(x)
        return self.classifier(z)


# ======================== Focal Loss ========================
class FocalLoss(nn.Module):
    """多分类Focal Loss，适用于类别不均衡场景"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ======================== 常量定义 ========================
# 8个唯一的候选模型（由不同组学数据训练）
ALL_CANDIDATE_MODELS = [
    ['clin'],                      # 0: Clinical（模型退化为MLP）
    ['clin', 'snv'],              # 1: Clinical+SNV
    ['clin', 'cnv'],              # 2: Clinical+CNA
    ['clin', 'mrna'],             # 3: Clinical+RNA
    ['clin', 'snv', 'cnv'],       # 4: Clinical+SNV+CNA
    ['clin', 'snv', 'mrna'],      # 5: Clinical+SNV+RNA
    ['clin', 'cnv', 'mrna'],      # 6: Clinical+CNA+RNA
    ['clin', 'snv', 'cnv', 'mrna'],  # 7: Full（所有组学）
]

# 8个模型平均组合，每个组合包含候选模型的索引
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
    """根据模态列表生成显示名称"""
    return '+'.join([MODALITY_NAME_MAP[mod] for mod in modalities])


# ======================== 数据加载 ========================
def standardize_sample_id(sample_id):
    """标准化样本ID：去除-01、-02等后缀（TCGA格式）"""
    if isinstance(sample_id, str):
        if sample_id.endswith('-01') or sample_id.endswith('-02') or sample_id.endswith('-03'):
            sample_id = sample_id.rsplit('-', 1)[0]
    return sample_id


def load_hrd_data(data_dir, return_sample_ids=False):
    """
    加载TCGA HRD数据

    Clinical: ./data/tcga/TCGA_Clinical_HRD.csv
      - patient: 样本ID列
      - HRD_label: 二分类标签
      - 其他列（AGE, ER, Her2, LN, STAGE）: 临床特征

    Args:
        data_dir: 数据目录路径
        return_sample_ids: 是否返回样本ID列表
    """
    clinical_path = os.path.join(data_dir, "TCGA_Clinical_HRD.csv")
    snv_path = os.path.join(data_dir, "tcga_SNV.csv")
    cna_path = os.path.join(data_dir, "tcga_CNV_CX.csv")
    mrna_path = os.path.join(data_dir, "tcga_mRNA.csv")

    print(f"加载TCGA HRD数据:")
    print(f"  临床数据: {clinical_path}")
    print(f"  SNV数据: {snv_path}")
    print(f"  CNA数据: {cna_path}")
    print(f"  RNA数据: {mrna_path}")

    # 读取数据
    clin = pd.read_csv(clinical_path)
    snv = pd.read_csv(snv_path)
    cna = pd.read_csv(cna_path)
    mrna = pd.read_csv(mrna_path)

    def set_index_from_column(df, id_col=None):
        """将指定列设为索引并标准化"""
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

    # ========== 处理临床数据 ==========
    # 'patient' 是ID列，先保存用于设索引
    clin = set_index_from_column(clin, id_col='patient')

    # 提取标签
    if 'HRD_label' in clin.columns:
        y = clin['HRD_label'].astype(int)
    else:
        raise ValueError("临床数据中未找到 HRD_label 列")

    # 处理临床特征：除HRD_label外的所有列
    clin_features = clin.drop(columns=['HRD_label']).copy()
    # 对类别变量进行编码（填充NaN后编码）
    for col in clin_features.columns:
        if not pd.api.types.is_numeric_dtype(clin_features[col]):
            clin_features[col] = clin_features[col].fillna('NA')
            clin_features[col] = LabelEncoder().fit_transform(clin_features[col].astype(str))
    clin_feat = clean_numeric(clin_features)

    # ========== 处理组学数据 ==========
    snv = set_index_from_column(snv)
    cna = set_index_from_column(cna)

    # RNA数据：检测Hallmark格式并转置
    first_col_name = mrna.columns[0]
    if first_col_name in ['GeneSet', 'Unnamed: 0', ''] or 'HALLMARK' in str(mrna.iloc[0, 0]):
        print("检测到Hallmark格式数据，进行转置...")
        mrna = mrna.set_index(mrna.columns[0])
        mrna = mrna.T
        print(f"转置后形状: {mrna.shape}")
    mrna = set_index_from_column(mrna)

    # 清洗组学数值数据
    snv = clean_numeric(snv)
    cna = clean_numeric(cna)
    mrna = clean_numeric(mrna)

    # ========== 样本对齐 ==========
    common_samples = sorted(list(
        set(snv.index) &
        set(cna.index) &
        set(mrna.index) &
        set(clin_feat.index) &
        set(y.index)
    ))

    print(f"匹配样本数: {len(common_samples)}")

    if len(common_samples) == 0:
        raise ValueError("无法找到共同样本，请检查数据索引是否匹配")

    snv = snv.loc[common_samples]
    cna = cna.loc[common_samples]
    mrna = mrna.loc[common_samples]
    clin_feat = clin_feat.loc[common_samples]
    y = y.loc[common_samples]

    # 转换为numpy数组
    X_snv = snv.values.astype(np.float32)
    X_cna = cna.values.astype(np.float32)
    X_mrna = mrna.values.astype(np.float32)
    X_clin = clin_feat.values.astype(np.float32)

    # 确保标签从0开始编码
    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)
    n_classes = len(np.unique(y_enc))

    print(f"HRD标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"类别数: {n_classes}")
    print(f"数据维度: SNV={X_snv.shape}, CNA={X_cna.shape}, mRNA={X_mrna.shape}, Clinical={X_clin.shape}")

    X_data = {
        "clin": X_clin,
        "snv": X_snv,
        "cnv": X_cna,
        "mrna": X_mrna,
    }

    feature_dims = {mod: X_data[mod].shape[1] for mod in X_data}

    if return_sample_ids:
        return X_data, y_enc, le_y, feature_dims, common_samples
    else:
        return X_data, y_enc, le_y, feature_dims


# ======================== 数据增强 ========================
def apply_data_augmentation(X_dict, y, modalities, method, random_state=42):
    """
    对训练数据进行数据增强

    Args:
        X_dict: 各模态数据字典
        y: 标签
        modalities: 当前模型使用的模态列表
        method: 'smote' 或 'oversample'
        random_state: 随机种子

    Returns:
        增强后的 X_dict, y
    """
    if method is None:
        return X_dict, y

    # 拼接多模态特征
    X_concat = np.concatenate([X_dict[mod] for mod in modalities], axis=1)

    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state, k_neighbors=min(5, np.sum(y == 1) - 1))
            X_res, y_res = smote.fit_resample(X_concat, y)
            print(f"      SMOTE增强: {len(y)} -> {len(y_res)} 样本")
        except ImportError:
            print("      SMOTE不可用(需pip install imblearn)，使用随机过采样替代")
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=random_state)
            X_res, y_res = ros.fit_resample(X_concat, y)
            print(f"      RandomOverSampler增强: {len(y)} -> {len(y_res)} 样本")
    elif method == 'oversample':
        try:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=random_state)
            X_res, y_res = ros.fit_resample(X_concat, y)
            print(f"      RandomOverSampler增强: {len(y)} -> {len(y_res)} 样本")
        except ImportError:
            # 手动实现随机过采样
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
            print(f"      手动过采样增强: {len(y)} -> {len(y_res)} 样本")
    else:
        return X_dict, y

    # 将增强后的特征按模态拆分
    offset = 0
    X_aug = {}
    for mod in modalities:
        dim = X_dict[mod].shape[1]
        X_aug[mod] = X_res[:, offset:offset + dim]
        offset += dim

    return X_aug, y_res


# ======================== 模型训练与预测 ========================
def train_model(X_train_dict, y_train, modalities, dims, n_classes,
                hidden=64, epochs=50, lr=0.001, device=None, seed=42,
                val_ratio=0.1, patience=10, data_augmentation=None,
                loss_type='ce'):
    """训练单个CAGMF-Net模型，每个epoch输出验证集F1/PRAUC/ROCAUC"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    n_modalities = len(modalities)
    if n_modalities <= 2:
        hidden = 64
        lr = 0.001
        max_epochs = 50
    elif n_modalities == 3:
        hidden = 96
        lr = 0.0008
        max_epochs = 80
    else:
        hidden = 128
        lr = 0.0005
        max_epochs = 100

    epochs = max_epochs

    if data_augmentation:
        X_train_dict, y_train = apply_data_augmentation(
            X_train_dict, y_train, modalities, data_augmentation, random_state=seed
        )

    if len(modalities) == 1:
        mod = modalities[0]
        model = SingleModalMLP(dims[mod], hidden, n_classes)
    else:
        model = CAGMFNet(dims, hidden, n_classes)

    model = model.to(device)

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=val_ratio, random_state=seed,
        stratify=y_train if len(np.unique(y_train)) > 1 else None
    )

    train_tensors = [torch.tensor(X_train_dict[mod][train_idx], dtype=torch.float32) for mod in modalities]
    train_dataset = TensorDataset(*train_tensors, torch.tensor(y_train[train_idx], dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_tensors = [torch.tensor(X_train_dict[mod][val_idx], dtype=torch.float32) for mod in modalities]
    val_dataset = TensorDataset(*val_tensors, torch.tensor(y_train[val_idx], dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    if loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    model_name = get_combination_display_name(modalities)
    best_val_f1 = 0.0
    best_val_prauc = 0.0
    best_val_roc_auc = 0.0
    best_epoch = 0
    trained_epochs = 0

    for epoch in range(epochs):
        trained_epochs = epoch + 1
        model.train()
        train_loss = 0.0
        for batch_data in train_loader:
            batch_X_dict = {}
            for i, mod in enumerate(modalities):
                batch_X_dict[mod] = batch_data[i].to(device)
            batch_y = batch_data[-1].to(device)

            optimizer.zero_grad()
            if len(modalities) == 1:
                outputs = model(batch_X_dict[modalities[0]])
            else:
                outputs = model(batch_X_dict)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        all_val_probs = []
        all_val_targets = []
        with torch.no_grad():
            for batch_data in val_loader:
                batch_X_dict = {}
                for i, mod in enumerate(modalities):
                    batch_X_dict[mod] = batch_data[i].to(device)
                batch_y = batch_data[-1].to(device)
                if len(modalities) == 1:
                    outputs = model(batch_X_dict[modalities[0]])
                else:
                    outputs = model(batch_X_dict)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                probs = F.softmax(outputs, dim=1)
                all_val_probs.append(probs.cpu().numpy())
                all_val_targets.append(batch_y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        val_probs_np = np.concatenate(all_val_probs, axis=0)
        val_targets_np = np.concatenate(all_val_targets, axis=0)
        val_preds_np = np.argmax(val_probs_np, axis=1)

        val_f1 = f1_score(val_targets_np, val_preds_np, average='macro', zero_division=0)
        if n_classes == 2:
            val_roc_auc = roc_auc_score(val_targets_np, val_probs_np[:, 1])
            val_prauc = average_precision_score(val_targets_np, val_probs_np[:, 1])
        else:
            y_bin = label_binarize(val_targets_np, classes=range(n_classes))
            val_roc_auc = roc_auc_score(y_bin, val_probs_np, multi_class='ovr', average='macro')
            val_prauc = average_precision_score(y_bin, val_probs_np, average='macro')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_prauc = val_prauc
            best_val_roc_auc = val_roc_auc
            best_epoch = epoch + 1

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    print(f"    {model_name} ({best_epoch}/{epochs} epochs) | "
          f"Val F1: {best_val_f1:.4f} | PRAUC: {best_val_prauc:.4f} | ROCAUC: {best_val_roc_auc:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


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


# ======================== Youden阈值 ========================
def find_youden_threshold(y_true, y_prob_class1):
    """
    寻找最大化Youden指数的分类阈值（仅用于二分类）

    Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_class1)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    return best_threshold, youden_j[best_idx]


def predict_with_youden_threshold(probs, threshold):
    """使用Youden阈值将概率转换为二分类预测"""
    preds = np.zeros(len(probs), dtype=int)
    preds[probs[:, 1] >= threshold] = 1
    return preds


# ======================== CV权重计算 ========================
def _cv_criterion_loss(weights, oof_group, y_true, n_classes, criterion):
    """计算CV准则损失（用于scipy优化）"""
    ensemble_probs = np.average(oof_group, axis=0, weights=weights)
    ensemble_probs = np.clip(ensemble_probs, 1e-15, 1 - 1e-15)

    if criterion == 'mse':
        y_onehot = np.eye(n_classes)[y_true]
        error = ensemble_probs - y_onehot
        return np.mean(error ** 2)

    elif criterion == 'ce':
        return -np.mean(np.log(ensemble_probs[np.arange(len(y_true)), y_true]))

    elif criterion == 'focal':
        pt = ensemble_probs[np.arange(len(y_true)), y_true]
        alpha, gamma = 0.25, 2.0
        return np.mean(alpha * (1 - pt) ** gamma * (-np.log(pt)))

    else:
        y_onehot = np.eye(n_classes)[y_true]
        error = ensemble_probs - y_onehot
        return np.mean(error ** 2)


def compute_cv_weights_for_group(oof_predictions, y_train, model_indices, n_classes, criterion='mse'):
    """
    为指定组的模型计算CV权重

    Args:
        criterion: 'mse' — 均方误差QP优化; 'ce' — 交叉熵; 'focal' — Focal Loss
    """
    n_models_group = len(model_indices)
    n_samples = len(y_train)
    oof_group = oof_predictions[model_indices, :, :]

    if criterion == 'mse':
        # MSE: 凸二次规划（全局最优）
        y_onehot = np.eye(n_classes)[y_train]
        errors = oof_group - y_onehot
        errors_flat = errors.reshape(n_models_group, -1)
        Q = (errors_flat @ errors_flat.T) / (n_samples * n_classes)
        Q = Q + 1e-8 * np.eye(n_models_group)

        n_models_q = Q.shape[0]

        try:
            from cvxopt import matrix, solvers
            P = matrix(Q.astype(float))
            q = matrix(np.zeros(n_models_q))
            G = matrix(-np.eye(n_models_q))
            h = matrix(np.zeros(n_models_q))
            A = matrix(np.ones((1, n_models_q)).astype(float))
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
            bounds = [(0, 1) for _ in range(n_models_q)]
            initial = np.ones(n_models_q) / n_models_q
            result = minimize(objective, initial, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 1000, 'disp': False})
            weights = result.x

    else:
        # CE / Focal: 使用scipy直接优化
        from scipy.optimize import minimize

        def objective(w):
            return _cv_criterion_loss(w, oof_group, y_train, n_classes, criterion)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_models_group)]
        initial = np.ones(n_models_group) / n_models_group

        result = minimize(objective, initial, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'disp': False})
        weights = result.x

    weights = np.maximum(weights, 0)
    weights = weights / np.sum(weights)
    return weights


# ======================== 保存结果 ========================
def save_predictions(predictions_dir, seed, group_name, sample_ids_test,
                     y_true, y_pred, y_probs, class_labels, metadata):
    """保存详细预测结果"""
    os.makedirs(predictions_dir, exist_ok=True)

    safe_group_name = group_name.replace('+', '_')
    pred_file = os.path.join(predictions_dir, f"seed_{seed}_{safe_group_name}_predictions.npz")

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    metadata_serializable = convert_to_serializable(metadata)

    np.savez(
        pred_file,
        sample_ids=sample_ids_test,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        class_labels=class_labels,
        metadata=json.dumps(metadata_serializable)
    )

    df = pd.DataFrame({
        'sample_id': sample_ids_test,
        'true_label': [class_labels[true] for true in y_true],
        'true_label_code': y_true,
        'pred_label': [class_labels[pred] for pred in y_pred],
        'pred_label_code': y_pred,
    })
    for i, label in enumerate(class_labels):
        df[f'prob_{label}'] = y_probs[:, i]

    csv_file = os.path.join(predictions_dir, f"seed_{seed}_{safe_group_name}_predictions.csv")
    df.to_csv(csv_file, index=False)

    return pred_file, csv_file


def flatten_metrics(metrics_dict):
    """展平指标字典"""
    flat = {
        'accuracy': metrics_dict['accuracy'],
        'log_loss': metrics_dict['log_loss'],
        'mse': metrics_dict['mse'],
        'mae': metrics_dict['mae']
    }
    for avg in ['macro', 'weighted', 'micro']:
        for metric in ['precision', 'recall', 'f1', 'roc_auc', 'prauc']:
            flat[f'{metric}_{avg}'] = metrics_dict[avg][metric]
    return flat


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(description='TCGA HRD数据 CAGMF-Net 模型平均组合评估')
    parser.add_argument('--data_dir', type=str, default='./data/tcga',
                        help='TCGA数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./eval_results/tcga_hrd_cagmf_ma',
                        help='输出结果目录')
    parser.add_argument('--n_splits', type=int, default=100,
                        help='随机划分次数，默认100')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例，默认0.2')
    parser.add_argument('--random_seed_base', type=int, default=42,
                        help='随机种子基数')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备，默认自动选择')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='交叉验证折数')
    parser.add_argument('--smote', action='store_true',
                        help='使用SMOTE数据增强')
    parser.add_argument('--oversample', action='store_true',
                        help='使用随机过采样数据增强')
    parser.add_argument('--youden', action='store_true',
                        help='使用Youden阈值选取方法（默认使用argmax）')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal'],
                        help='模型损失函数: ce (交叉熵) 或 focal (Focal Loss)，默认ce')
    parser.add_argument('--cv_criterion', type=str, default='mse', choices=['mse', 'ce', 'focal'],
                        help='CV准则: mse (均方误差), ce (交叉熵), focal (Focal Loss)，默认mse')
    parser.add_argument('--skip_split_generation', action='store_true',
                        help='跳过划分生成，使用已有划分')

    args = parser.parse_args()

    # 数据增强方式（互斥）
    data_augmentation = None
    if args.smote:
        data_augmentation = 'smote'
    elif args.oversample:
        data_augmentation = 'oversample'

    print("=" * 80)
    print("TCGA HRD数据 CAGMF-Net 模型平均组合评估")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"划分次数: {args.n_splits}, 测试比例: {args.test_size}")
    print(f"CV折数: {args.cv_folds}")
    print(f"损失函数: {'Focal Loss' if args.loss == 'focal' else '交叉熵'}")
    print(f"CV准则: {'均方误差' if args.cv_criterion == 'mse' else '交叉熵' if args.cv_criterion == 'ce' else 'Focal Loss'}")
    if data_augmentation:
        print(f"数据增强: {data_augmentation}")
    if args.youden:
        print(f"阈值方法: Youden指数")
    else:
        print(f"阈值方法: argmax (默认)")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 1. 加载数据
    print("\n1. 加载TCGA HRD数据...")
    X_dict, y, le_y, feature_dims, sample_ids = load_hrd_data(
        args.data_dir, return_sample_ids=True
    )
    n_classes = len(np.unique(y))
    print(f"总样本数: {len(y)}, 类别数: {n_classes}")
    print(f"各模态维度: {feature_dims}")

    class_labels = [int(c) for c in le_y.classes_]

    # 2. 生成/加载划分索引
    print("\n2. 生成/加载数据划分...")
    split_dir_name = "split_indices_tcga_hrd"
    split_dir = os.path.join(args.output_dir, split_dir_name)

    if args.skip_split_generation and os.path.exists(split_dir):
        print(f"使用已有划分目录: {split_dir}")
        split_files = [f for f in os.listdir(split_dir) if f.endswith('.npz')]
        if len(split_files) < args.n_splits:
            print(f"已有划分 {len(split_files)} 个，目标 {args.n_splits} 个，重新生成...")
            generate_splits(split_dir, y, args.n_splits, args.test_size, args.random_seed_base)
    else:
        generate_splits(split_dir, y, args.n_splits, args.test_size, args.random_seed_base)

    # 3. 准备模型组合
    print("\n3. 模型组合配置:")
    all_modal_candidates = ALL_CANDIDATE_MODELS
    combination_groups = COMBINATION_GROUPS
    for group_name, indices in combination_groups.items():
        model_names = [get_combination_display_name(all_modal_candidates[i]) for i in indices]
        print(f"  {group_name}: {len(indices)} 个模型 -> {model_names}")

    # 4. 创建存储目录
    print("\n4. 初始化结果存储...")
    results_dir = os.path.join(args.output_dir, "results")
    models_dir = os.path.join(args.output_dir, "saved_models")
    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # 5. 遍历每个划分
    print(f"\n5. 开始实验，共 {args.n_splits} 个划分...")
    seeds = range(args.random_seed_base, args.random_seed_base + args.n_splits)

    all_seeds_results = {}
    saved_predictions = []

    for i, seed in enumerate(tqdm(seeds, desc="处理种子")):
        print(f"\n--- 划分 {i + 1}/{args.n_splits} (种子: {seed}) ---")

        # 加载划分索引
        split_path = os.path.join(split_dir, f"seed_{seed}_split.npz")
        if not os.path.exists(split_path):
            print(f"  种子 {seed} 划分文件不存在，跳过")
            continue
        split_data = np.load(split_path, allow_pickle=True)
        train_idx = split_data['train_idx']
        test_idx = split_data['test_idx']

        # 获取训练和测试数据
        X_train_raw = {}
        X_test_raw = {}
        for mod in X_dict.keys():
            X_train_raw[mod] = X_dict[mod][train_idx]
            X_test_raw[mod] = X_dict[mod][test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]
        sample_ids_test = [sample_ids[i] for i in test_idx] if sample_ids is not None else np.arange(len(y_test))

        print(f"  训练集: {len(y_train)} 样本, 测试集: {len(y_test)} 样本")
        if data_augmentation:
            print(f"  数据增强: {data_augmentation}")

        # 5折交叉验证计算所有候选模型的OOF预测
        print(f"  执行 {args.cv_folds} 折交叉验证计算OOF预测...")

        K = args.cv_folds
        n_samples_train = len(y_train)
        n_models = len(all_modal_candidates)
        oof_predictions_all = np.zeros((n_models, n_samples_train, n_classes))

        kf = KFold(n_splits=K, shuffle=True, random_state=seed)

        for fold_idx, (train_idx_fold, val_idx_fold) in enumerate(kf.split(y_train)):
            print(f"    Fold {fold_idx + 1}/{K}")

            # 准备fold数据（标准化）
            train_X_fold = {}
            val_X_fold = {}
            for mod in X_train_raw.keys():
                scaler = StandardScaler()
                X_train_fold_data = X_train_raw[mod][train_idx_fold]
                scaler.fit(X_train_fold_data)
                train_X_fold[mod] = scaler.transform(X_train_fold_data)
                val_X_fold[mod] = scaler.transform(X_train_raw[mod][val_idx_fold])

            y_train_fold = y_train[train_idx_fold]

            # 训练每个候选模型
            for m, modalities in enumerate(all_modal_candidates):
                dims = {mod: feature_dims[mod] for mod in modalities}
                X_train_sub = {mod: train_X_fold[mod] for mod in modalities}

                model = train_model(
                    X_train_sub, y_train_fold, modalities, dims, n_classes,
                    device=device, seed=seed + fold_idx * 10 + m,
                    data_augmentation=data_augmentation,
                    loss_type=args.loss
                )

                X_val_sub = {mod: val_X_fold[mod] for mod in modalities}
                probs = predict_model(model, X_val_sub, modalities, device)
                oof_predictions_all[m, val_idx_fold, :] = probs

        print(f"    CV完成")

        # 标准化完整训练集和测试集
        X_train_std = {}
        X_test_std = {}
        scalers = {}
        for mod in X_dict.keys():
            scaler = StandardScaler()
            X_train_std[mod] = scaler.fit_transform(X_train_raw[mod])
            X_test_std[mod] = scaler.transform(X_test_raw[mod])
            scalers[mod] = scaler

        # 训练最终模型
        print(f"  训练最终模型并预测测试集...")
        final_models = []
        model_paths = []

        seed_model_dir = os.path.join(models_dir, f"seed_{seed}")
        os.makedirs(seed_model_dir, exist_ok=True)

        for m, modalities in enumerate(all_modal_candidates):
            dims = {mod: feature_dims[mod] for mod in modalities}
            X_train_sub = {mod: X_train_std[mod] for mod in modalities}

            model = train_model(
                X_train_sub, y_train, modalities, dims, n_classes,
                device=device, seed=seed + m,
                data_augmentation=data_augmentation,
                loss_type=args.loss
            )

            model_name = get_combination_display_name(modalities)
            model_path = os.path.join(seed_model_dir, f"{model_name}_model.pth")
            torch.save(model.state_dict(), model_path)

            metadata = {
                'modalities': modalities,
                'dims': dims,
                'n_classes': n_classes,
                'model_type': 'SingleModalMLP' if len(modalities) == 1 else 'CAGMFNet',
                'model_name': model_name
            }
            metadata_path = model_path.replace('.pth', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            final_models.append(model)
            model_paths.append(model_path)

        # 获取测试集预测
        test_probs_all = []
        for m, modalities in enumerate(all_modal_candidates):
            X_test_sub = {mod: X_test_std[mod] for mod in modalities}
            probs = predict_model(final_models[m], X_test_sub, modalities, device)
            test_probs_all.append(probs)
        test_probs_all = np.array(test_probs_all)

        # 对每个组合计算CV权重和最终结果
        seed_results = {}

        for group_name, model_indices in combination_groups.items():
            if len(model_indices) == 0:
                continue

            # 计算CV权重
            weights = compute_cv_weights_for_group(
                oof_predictions_all, y_train, model_indices, n_classes,
                criterion=args.cv_criterion
            )

            # 加权平均预测
            group_test_probs = test_probs_all[model_indices, :, :]
            final_probs = np.zeros_like(group_test_probs[0])
            for w, probs in zip(weights, group_test_probs):
                final_probs += w * probs

            # 预测类别
            youden_threshold = None
            if args.youden and n_classes == 2:
                # 从OOF加权预测计算Youden阈值
                group_oop_probs = oof_predictions_all[model_indices, :, :]
                oof_ensemble = np.zeros_like(group_oop_probs[0])
                for w, probs in zip(weights, group_oop_probs):
                    oof_ensemble += w * probs
                youden_threshold, youden_j = find_youden_threshold(y_train, oof_ensemble[:, 1])
                final_pred = predict_with_youden_threshold(final_probs, youden_threshold)
                print(f"    {group_name}: Youden阈值={youden_threshold:.4f}, J={youden_j:.4f}")
            else:
                final_pred = np.argmax(final_probs, axis=1)

            # 评估
            eval_metrics_nested = evaluate_predictions(final_probs, final_pred, y_test, n_classes)
            eval_metrics_flat = flatten_metrics(eval_metrics_nested)

            # 保存预测结果
            metadata = {
                'seed': seed,
                'group': group_name,
                'n_models': len(model_indices),
                'models': [get_combination_display_name(all_modal_candidates[idx]) for idx in model_indices],
                'weights': weights.tolist(),
                'metrics': eval_metrics_nested,
                'n_classes': n_classes,
                'class_labels': class_labels,
                'data_augmentation': data_augmentation,
                'loss_type': args.loss,
                'cv_criterion': args.cv_criterion,
                'threshold_method': 'youden' if args.youden else 'argmax',
                'youden_threshold': float(youden_threshold) if youden_threshold is not None else None,
            }

            pred_file, csv_file = save_predictions(
                predictions_dir, seed, group_name, sample_ids_test,
                y_test, final_pred, final_probs, class_labels, metadata
            )
            saved_predictions.append({
                'seed': seed,
                'group': group_name,
                'npz_file': pred_file,
                'csv_file': csv_file,
                'data_augmentation': data_augmentation,
            })

            seed_results[group_name] = {
                'weights': weights.tolist(),
                'model_indices': model_indices,
                'metrics': eval_metrics_flat,
                'metrics_nested': eval_metrics_nested,
                'model_paths': [model_paths[idx] for idx in model_indices],
                'predictions_file': pred_file,
                'predictions_csv': csv_file,
                'youden_threshold': float(youden_threshold) if youden_threshold is not None else None,
            }

            print(f"    {group_name}: "
                  f"Acc={eval_metrics_nested['accuracy']:.4f}, "
                  f"F1={eval_metrics_nested['macro']['f1']:.4f}, "
                  f"AUC={eval_metrics_nested['macro']['roc_auc']:.4f}")

        # 保存scaler
        scaler_path = os.path.join(seed_model_dir, "scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)

        all_seeds_results[seed] = seed_results

    # 6. 汇总结果
    print(f"\n{'=' * 80}")
    print("6. 汇总实验结果...")
    print(f"{'=' * 80}")

    flat_metrics_names = [
        'accuracy', 'log_loss', 'mse', 'mae',
        'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_macro', 'prauc_macro',
        'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_weighted', 'prauc_weighted',
        'precision_micro', 'recall_micro', 'f1_micro', 'roc_auc_micro', 'prauc_micro'
    ]

    all_groups_results = {}

    for group_name in combination_groups.keys():
        group_metrics = {metric: [] for metric in flat_metrics_names}
        group_weights = []
        group_thresholds = []

        for seed, seed_result in all_seeds_results.items():
            if group_name in seed_result:
                metrics = seed_result[group_name]['metrics']
                for metric in flat_metrics_names:
                    if metric in metrics:
                        group_metrics[metric].append(metrics[metric])
                group_weights.append(seed_result[group_name]['weights'])
                th = seed_result[group_name].get('youden_threshold')
                if th is not None:
                    group_thresholds.append(th)

        summary = {}
        for metric in flat_metrics_names:
            values = group_metrics[metric]
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        if group_weights:
            weights_array = np.array(group_weights)
            summary['weights_mean'] = weights_array.mean(axis=0).tolist()
            summary['weights_std'] = weights_array.std(axis=0).tolist()

        if group_thresholds:
            summary['youden_threshold_mean'] = float(np.mean(group_thresholds))
            summary['youden_threshold_std'] = float(np.std(group_thresholds))

        all_groups_results[group_name] = summary

        print(f"\n{group_name}:")
        print(f"  候选模型数: {len(combination_groups[group_name])}")
        if 'weights_mean' in summary:
            print(f"  CV权重均值: {[f'{w:.4f}' for w in summary['weights_mean']]}")
        print(f"  F1 (macro): {summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
        print(f"  AUC (macro): {summary['roc_auc_macro']['mean']:.4f}±{summary['roc_auc_macro']['std']:.4f}")
        print(f"  Accuracy: {summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}")

    # 7. 保存结果
    print("\n7. 保存结果...")

    # 汇总表
    summary_rows = []
    for group_name, summary in all_groups_results.items():
        row = {'Group': group_name, 'n_models': len(combination_groups[group_name])}
        for metric in flat_metrics_names:
            if metric in summary:
                row[metric] = f"{summary[metric]['mean']:.4f}±{summary[metric]['std']:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, "all_groups_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  汇总结果保存至: {summary_path}")

    # 详细结果表
    detailed_rows = []
    for seed, seed_result in all_seeds_results.items():
        for group_name, result in seed_result.items():
            row = {'seed': seed, 'group': group_name}
            for metric, value in result['metrics'].items():
                row[metric] = value
            row['predictions_file'] = result.get('predictions_file', '')
            row['predictions_csv'] = result.get('predictions_csv', '')
            row['youden_threshold'] = result.get('youden_threshold', '')
            detailed_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = os.path.join(results_dir, "all_groups_detailed.csv")
    detailed_df.to_csv(detailed_path, index=False)
    print(f"  详细结果保存至: {detailed_path}")

    # 权重表
    weights_rows = []
    for seed, seed_result in all_seeds_results.items():
        for group_name, result in seed_result.items():
            for idx, w in enumerate(result['weights']):
                combo = all_modal_candidates[result['model_indices'][idx]]
                combo_name = get_combination_display_name(combo)
                weights_rows.append({
                    'seed': seed,
                    'group': group_name,
                    'model_index': idx,
                    'model_name': combo_name,
                    'weight': w,
                })

    weights_df = pd.DataFrame(weights_rows)
    weights_path = os.path.join(results_dir, "all_groups_weights.csv")
    weights_df.to_csv(weights_path, index=False)
    print(f"  CV权重保存至: {weights_path}")

    # 预测文件索引
    predictions_index_df = pd.DataFrame(saved_predictions)
    predictions_index_path = os.path.join(results_dir, "predictions_index.csv")
    predictions_index_df.to_csv(predictions_index_path, index=False)
    print(f"  预测文件索引保存至: {predictions_index_path}")

    # 标签映射
    label_mapping = {
        'class_labels': class_labels,
        'class_codes': list(range(len(class_labels))),
        'n_classes': n_classes,
    }
    label_mapping_path = os.path.join(results_dir, "label_mapping.json")
    with open(label_mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"  标签映射保存至: {label_mapping_path}")

    # 完整结果JSON
    results_dict = {
        'groups': {name: [get_combination_display_name(all_modal_candidates[i]) for i in indices]
                   for name, indices in combination_groups.items()},
        'summary': {group: {
            'n_models': len(combination_groups[group]),
            'metrics': {metric: {'mean': summary[metric]['mean'], 'std': summary[metric]['std']}
                        for metric in flat_metrics_names if metric in summary}
        } for group, summary in all_groups_results.items()},
        'models_save_dir': models_dir,
        'predictions_dir': predictions_dir,
        'label_mapping': label_mapping,
        'data_augmentation': data_augmentation,
        'loss_type': args.loss,
        'cv_criterion': args.cv_criterion,
        'threshold_method': 'youden' if args.youden else 'argmax',
    }

    results_path = os.path.join(results_dir, "all_groups_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  完整结果保存至: {results_path}")

    # 实验配置
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'n_splits': args.n_splits,
        'test_size': args.test_size,
        'random_seed_base': args.random_seed_base,
        'cv_folds': args.cv_folds,
        'data_augmentation': data_augmentation,
        'loss_type': args.loss,
        'cv_criterion': args.cv_criterion,
        'threshold_method': 'youden' if args.youden else 'argmax',
        'n_classes': n_classes,
        'feature_dims': feature_dims,
        'total_samples': len(y),
        'candidate_models': [get_combination_display_name(m) for m in all_modal_candidates],
        'combination_groups': {k: [get_combination_display_name(all_modal_candidates[i]) for i in v]
                               for k, v in combination_groups.items()},
    }

    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  实验配置保存至: {config_path}")

    print(f"\n{'=' * 80}")
    print("TCGA HRD数据实验完成！")
    print(f"结果目录: {args.output_dir}")
    print(f"{'=' * 80}")


def generate_splits(split_dir, y, n_splits=100, test_size=0.2, random_seed_base=42):
    """生成并保存随机划分索引（单独函数，不依赖数据文件）"""
    os.makedirs(split_dir, exist_ok=True)
    print(f"  生成 {n_splits} 个划分，保存至: {split_dir}")

    for i in range(n_splits):
        seed = random_seed_base + i
        set_seed(seed)

        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            random_state=seed,
            stratify=y
        )

        split_path = os.path.join(split_dir, f"seed_{seed}_split.npz")
        np.savez(split_path, train_idx=train_idx, test_idx=test_idx)

    print(f"  完成")


if __name__ == "__main__":
    main()
