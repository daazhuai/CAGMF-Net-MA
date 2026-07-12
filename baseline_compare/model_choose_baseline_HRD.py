"""
模型选择基线评估 (SAIC/SBIC) - TCGA HRD数据专用版
使用SAIC/SBIC准则从候选模型池中选取模型权重，替代CV优化权重。
候选模型池与主实验(cagmf_net_ma_cv_addeval_HRD.py)相同(8个CAGMF-Net/MLP)，
通过AIC/BIC信息准则计算softmax权重进行模型平均。
基模型为CAGMF-Net，单模态退化至MLP。

SAIC权重: w_i = exp(-AIC_i/2) / Σ_j exp(-AIC_j/2)
SBIC权重: w_i = exp(-BIC_i/2) / Σ_j exp(-BIC_j/2)
其中 AIC_i = 2k_i + 2*NLL_i, BIC_i = k_i*ln(n) + 2*NLL_i
n为训练集样本数，k_i为模型参数量，NLL_i为训练集上的负对数似然(求和)
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
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
# 8个唯一的候选模型（与主实验完全相同）
ALL_CANDIDATE_MODELS = [
    ['clin'],                           # 0: Clinical（模型退化为MLP）
    ['clin', 'snv'],                    # 1: Clinical+SNV
    ['clin', 'cnv'],                    # 2: Clinical+CNA
    ['clin', 'mrna'],                   # 3: Clinical+RNA
    ['clin', 'snv', 'cnv'],             # 4: Clinical+SNV+CNA
    ['clin', 'snv', 'mrna'],            # 5: Clinical+SNV+RNA
    ['clin', 'cnv', 'mrna'],            # 6: Clinical+CNA+RNA
    ['clin', 'snv', 'cnv', 'mrna'],     # 7: Full（所有组学）
]

# 8个模型平均组合（与主实验完全相同）
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

    Returns:
        X_data: dict of modality -> np.array
        y_enc: encoded labels
        le_y: LabelEncoder
        feature_dims: dict of modality -> dimension
        (optional) sample_ids: list of sample IDs
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

    # 处理临床数据
    clin = set_index_from_column(clin, id_col='patient')

    if 'HRD_label' in clin.columns:
        y = clin['HRD_label'].astype(int)
    else:
        raise ValueError("临床数据中未找到 HRD_label 列")

    clin_features = clin.drop(columns=['HRD_label']).copy()
    for col in clin_features.columns:
        if not pd.api.types.is_numeric_dtype(clin_features[col]):
            clin_features[col] = clin_features[col].fillna('NA')
            clin_features[col] = LabelEncoder().fit_transform(clin_features[col].astype(str))
    clin_feat = clean_numeric(clin_features)

    # 处理组学数据
    snv = set_index_from_column(snv)
    cna = set_index_from_column(cna)

    first_col_name = mrna.columns[0]
    if first_col_name in ['GeneSet', 'Unnamed: 0', ''] or 'HALLMARK' in str(mrna.iloc[0, 0]):
        print("检测到Hallmark格式数据，进行转置...")
        mrna = mrna.set_index(mrna.columns[0])
        mrna = mrna.T
        print(f"转置后形状: {mrna.shape}")
    mrna = set_index_from_column(mrna)

    snv = clean_numeric(snv)
    cna = clean_numeric(cna)
    mrna = clean_numeric(mrna)

    # 样本对齐
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

    X_snv = snv.values.astype(np.float32)
    X_cna = cna.values.astype(np.float32)
    X_mrna = mrna.values.astype(np.float32)
    X_clin = clin_feat.values.astype(np.float32)

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


# ======================== 划分生成 ========================
def generate_splits(split_dir, y, n_splits=100, test_size=0.2, random_seed_base=42):
    """生成并保存随机划分索引"""
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


# ======================== 数据增强 ========================
def apply_data_augmentation(X_dict, y, modalities, method, random_state=42):
    """对训练数据进行数据增强（SMOTE或随机过采样）"""
    if method is None:
        return X_dict, y

    X_concat = np.concatenate([X_dict[mod] for mod in modalities], axis=1)

    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            n_minority = np.sum(y == np.bincount(y).argmin())
            k_neighbors = min(5, max(1, n_minority - 1))
            smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
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
            classes, counts = np.unique(y, return_counts=True)
            max_count = counts.max()
            indices = []
            for cls in classes:
                cls_idx = np.where(y == cls)[0]
                if len(cls_idx) < max_count:
                    additional = np.random.RandomState(random_state).choice(
                        cls_idx, max_count - len(cls_idx), replace=True
                    )
                    indices.extend(cls_idx.tolist())
                    indices.extend(additional.tolist())
                else:
                    indices.extend(cls_idx.tolist())
            X_res = X_concat[np.array(indices)]
            y_res = y[np.array(indices)]
            print(f"      手动过采样增强: {len(y)} -> {len(y_res)} 样本")
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
def train_model(X_train_dict, y_train, modalities, dims, n_classes,
                hidden=64, epochs=50, lr=0.001, device=None, seed=42,
                val_ratio=0.1, patience=10, data_augmentation=None,
                loss_type='ce'):
    """训练单个CAGMF-Net模型"""
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
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
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
        scheduler.step()

        model.eval()
        val_loss = 0.0
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

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch + 1
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_name = get_combination_display_name(modalities)
    print(f"    {model_name} ({best_epoch}/{epochs} epochs)")

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


# ======================== AIC/BIC 计算 ========================
def count_parameters(model):
    """统计模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_nll_sum(model, X_dict, y, modalities, device):
    """
    计算模型在给定数据上的负对数似然之和 (sum of cross-entropy)
    使用 CrossEntropyLoss(reduction='sum')

    Args:
        model: 训练好的模型
        X_dict: 各模态数据
        y: 标签
        modalities: 模态列表
        device: 计算设备

    Returns:
        nll_sum: 负对数似然之和 (scalar)
    """
    model.eval()
    tensors = [torch.tensor(X_dict[mod], dtype=torch.float32) for mod in modalities]
    dataset = TensorDataset(*tensors, torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_nll = 0.0

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
            batch_y = batch_data[-1].to(device)
            total_nll += criterion(outputs, batch_y).item()

    return total_nll


def compute_aic_bic_values(k_list, nll_list, n_samples):
    """
    计算一组模型的AIC和BIC值（不除以样本数n）

    Args:
        k_list: 各模型参数量列表
        nll_list: 各模型NLL之和列表
        n_samples: 训练样本数（原始样本数，非增强后）

    Returns:
        aic_values: AIC 列表
        bic_values: BIC 列表
    """
    k_arr = np.array(k_list, dtype=np.float64)
    nll_arr = np.array(nll_list, dtype=np.float64)

    # AIC = 2k + 2*NLL
    aic_values = 2.0 * k_arr + 2.0 * nll_arr

    # BIC = k*ln(n) + 2*NLL
    bic_values = k_arr * np.log(n_samples) + 2.0 * nll_arr

    return aic_values, bic_values


def compute_saic_sbic_weights(aic_values, bic_values):
    """
    计算SAIC和SBIC权重（softmax归一化）

    SAIC: w_i = exp(-AIC_i/2) / Σ_j exp(-AIC_j/2)
          = softmax(-AIC_i/2)
    SBIC: w_i = exp(-BIC_i/2) / Σ_j exp(-BIC_j/2)
          = softmax(-BIC_i/2)

    Args:
        aic_values: AIC 数组
        bic_values: BIC 数组

    Returns:
        saic_weights: SAIC权重
        sbic_weights: SBIC权重
    """
    def stable_softmax_neg_half(values):
        """softmax(-values/2) with numerical stability"""
        v = np.array(values, dtype=np.float64)
        scaled = -v / 2.0
        scaled = scaled - np.max(scaled)  # subtract max for stability
        exp_vals = np.exp(scaled)
        return exp_vals / np.sum(exp_vals)

    saic_weights = stable_softmax_neg_half(aic_values)
    sbic_weights = stable_softmax_neg_half(bic_values)

    return saic_weights, sbic_weights


# ======================== Youden阈值 ========================
def find_youden_threshold(y_true, y_prob_class1):
    """寻找最大化Youden指数的分类阈值（仅用于二分类）"""
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


# ======================== 结果展平 ========================
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


# ======================== 预测结果保存 ========================
def save_predictions(predictions_dir, seed, group_name, method_name,
                     sample_ids_test, y_true, y_pred, y_probs,
                     class_labels, metadata):
    """保存详细预测结果"""
    os.makedirs(predictions_dir, exist_ok=True)

    safe_group_name = group_name.replace('+', '_')
    pred_file = os.path.join(
        predictions_dir,
        f"seed_{seed}_{safe_group_name}_{method_name}_predictions.npz"
    )

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

    csv_file = os.path.join(
        predictions_dir,
        f"seed_{seed}_{safe_group_name}_{method_name}_predictions.csv"
    )
    df.to_csv(csv_file, index=False)

    return pred_file, csv_file


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(
        description='TCGA HRD数据 SAIC/SBIC模型选择基线评估'
    )
    parser.add_argument('--data_dir', type=str, default='./data/tcga',
                        help='TCGA数据目录路径')
    parser.add_argument('--output_dir', type=str,
                        default='./eval_results/tcga_hrd_model_choose',
                        help='输出结果目录')
    parser.add_argument('--n_splits', type=int, default=100,
                        help='随机划分次数，默认100')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例，默认0.2')
    parser.add_argument('--random_seed_base', type=int, default=42,
                        help='随机种子基数')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备，默认自动选择')
    parser.add_argument('--smote', action='store_true',
                        help='使用SMOTE数据增强')
    parser.add_argument('--oversample', action='store_true',
                        help='使用随机过采样数据增强')
    parser.add_argument('--youden', action='store_true',
                        help='使用Youden阈值选取方法（默认使用argmax）')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal'],
                        help='模型损失函数: ce (交叉熵) 或 focal (Focal Loss)，默认ce')
    parser.add_argument('--method', type=str, default='both',
                        choices=['saic', 'sbic', 'both'],
                        help='模型选择方法: saic, sbic, both（默认both）')
    parser.add_argument('--skip_split_generation', action='store_true',
                        help='跳过划分生成，使用已有划分')

    args = parser.parse_args()

    # 数据增强方式（互斥）
    data_augmentation = None
    if args.smote:
        data_augmentation = 'smote'
    elif args.oversample:
        data_augmentation = 'oversample'

    # 确定运行哪些方法
    run_methods = []
    if args.method in ['saic', 'both']:
        run_methods.append('SAIC')
    if args.method in ['sbic', 'both']:
        run_methods.append('SBIC')

    print("=" * 80)
    print("TCGA HRD数据 SAIC/SBIC 模型选择基线评估")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"划分次数: {args.n_splits}, 测试比例: {args.test_size}")
    print(f"模型选择方法: {', '.join(run_methods)}")
    print(f"损失函数: {'Focal Loss' if args.loss == 'focal' else '交叉熵'}")
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

    # 候选模型信息
    print(f"\n候选模型池: {len(ALL_CANDIDATE_MODELS)} 个")
    for i, mods in enumerate(ALL_CANDIDATE_MODELS):
        print(f"  [{i}] {get_combination_display_name(mods)}")
    print(f"\n组合组: {len(COMBINATION_GROUPS)} 个")
    for name, indices in COMBINATION_GROUPS.items():
        model_names = [get_combination_display_name(ALL_CANDIDATE_MODELS[i]) for i in indices]
        print(f"  {name}: {len(indices)} 个模型 -> {model_names}")

    # 1. 加载数据
    print("\n1. 加载TCGA HRD数据...")
    X_dict, y, le_y, feature_dims, sample_ids = load_hrd_data(
        args.data_dir, return_sample_ids=True
    )
    n_classes = len(np.unique(y))
    class_labels = [int(c) for c in le_y.classes_]
    print(f"总样本数: {len(y)}, 类别数: {n_classes}")
    print(f"各模态维度: {feature_dims}")

    # 2. 生成/加载划分索引
    print("\n2. 生成/加载数据划分...")
    split_dir_name = "split_indices_tcga_hrd_model_choose"
    split_dir = os.path.join(args.output_dir, split_dir_name)

    if args.skip_split_generation and os.path.exists(split_dir):
        print(f"使用已有划分目录: {split_dir}")
        split_files = [f for f in os.listdir(split_dir) if f.endswith('.npz')]
        if len(split_files) < args.n_splits:
            print(f"已有划分 {len(split_files)} 个，目标 {args.n_splits} 个，重新生成...")
            generate_splits(split_dir, y, args.n_splits, args.test_size, args.random_seed_base)
    else:
        generate_splits(split_dir, y, args.n_splits, args.test_size, args.random_seed_base)

    # 3. 创建存储目录
    print("\n3. 初始化结果存储...")
    results_dir = os.path.join(args.output_dir, "results")
    models_dir = os.path.join(args.output_dir, "saved_models")
    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # 4. 遍历每个划分
    print(f"\n4. 开始实验，共 {args.n_splits} 个划分...")
    seeds = range(args.random_seed_base, args.random_seed_base + args.n_splits)

    # 结果存储: all_results[method][seed][group_name] = {metrics, weights, ...}
    all_results = {method: {} for method in run_methods}
    saved_predictions = []

    for i, seed in enumerate(tqdm(seeds, desc="处理种子")):
        set_seed(seed)

        # 加载划分索引
        split_path = os.path.join(split_dir, f"seed_{seed}_split.npz")
        if not os.path.exists(split_path):
            print(f"\n  种子 {seed} 划分文件不存在，跳过")
            continue
        split_data = np.load(split_path, allow_pickle=True)
        train_idx = split_data['train_idx']
        test_idx = split_data['test_idx']

        # 获取并标准化数据
        X_train_raw = {}
        X_test_raw = {}
        scalers = {}
        for mod in X_dict.keys():
            scaler = StandardScaler()
            X_train_raw[mod] = scaler.fit_transform(X_dict[mod][train_idx])
            X_test_raw[mod] = scaler.transform(X_dict[mod][test_idx])
            scalers[mod] = scaler

        y_train = y[train_idx]
        y_test = y[test_idx]
        n_train_original = len(y_train)  # 原始训练样本数（用于AIC/BIC）
        sample_ids_test = [sample_ids[idx] for idx in test_idx] if sample_ids is not None else np.arange(len(y_test))

        print(f"\n--- 划分 {i + 1}/{args.n_splits} (种子: {seed}) ---")
        print(f"  训练集: {n_train_original} 样本, 测试集: {len(y_test)} 样本")

        # ============================================================
        # 阶段1: 训练8个候选模型 + 计算AIC/BIC
        # ============================================================
        print(f"  训练 {len(ALL_CANDIDATE_MODELS)} 个候选模型并计算AIC/BIC...")

        final_models = []
        k_list = []       # 各模型参数量
        nll_list = []     # 各模型在原始训练集上的NLL之和

        seed_model_dir = os.path.join(models_dir, f"seed_{seed}")
        os.makedirs(seed_model_dir, exist_ok=True)

        for m, modalities in enumerate(ALL_CANDIDATE_MODELS):
            dims = {mod: feature_dims[mod] for mod in modalities}

            # 训练模型（使用原始训练集，内部做train/val划分 + 可选数据增强）
            model = train_model(
                X_train_raw, y_train, modalities, dims, n_classes,
                device=device, seed=seed + m,
                data_augmentation=data_augmentation,
                loss_type=args.loss
            )

            # 统计输入特征数（各模态维度之和）
            k = sum(feature_dims[mod] for mod in modalities)
            k_list.append(k)

            # 计算原始训练集上的NLL（不增强，不dropout）
            nll = compute_nll_sum(model, X_train_raw, y_train, modalities, device)
            nll_list.append(nll)

            model_name = get_combination_display_name(modalities)
            print(f"    [{m}] {model_name}: k={k}, NLL={nll:.2f}, NLL/n={nll/n_train_original:.4f}")

            # 保存模型
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

        # 计算AIC和BIC
        aic_values, bic_values = compute_aic_bic_values(k_list, nll_list, n_train_original)

        print(f"    AIC: {[f'{v:.4f}' for v in aic_values]}")
        print(f"    BIC: {[f'{v:.4f}' for v in bic_values]}")

        # ============================================================
        # 阶段2: 获取测试集预测
        # ============================================================
        test_probs_all = []
        for m, modalities in enumerate(ALL_CANDIDATE_MODELS):
            X_test_sub = {mod: X_test_raw[mod] for mod in modalities}
            probs = predict_model(final_models[m], X_test_sub, modalities, device)
            test_probs_all.append(probs)
        test_probs_all = np.array(test_probs_all)

        # ============================================================
        # 阶段3: 对每个方法(SAIC/SBIC)和每个组合组，硬选择最优模型并评估
        # ============================================================
        for method_name in run_methods:
            # 确定使用AIC还是BIC进行选择
            if method_name == 'SAIC':
                criterion_values = aic_values  # 越小越好
            else:  # SBIC
                criterion_values = bic_values

            seed_results = {}

            for group_name, model_indices in COMBINATION_GROUPS.items():
                if len(model_indices) == 0:
                    continue

                # 硬选择：组内AIC/BIC最小的唯一模型，权重=1.0
                group_criterion = np.array([criterion_values[idx] for idx in model_indices])
                best_local_idx = np.argmin(group_criterion)
                best_global_idx = model_indices[best_local_idx]
                selected_model_name = get_combination_display_name(ALL_CANDIDATE_MODELS[best_global_idx])

                group_weights = np.zeros(len(model_indices))
                group_weights[best_local_idx] = 1.0

                # 直接用所选模型的预测
                group_test_probs = test_probs_all[model_indices, :, :]
                final_probs = group_test_probs[best_local_idx].copy()

                # 预测类别
                youden_threshold = None
                if args.youden and n_classes == 2:
                    train_probs_selected = predict_model(final_models[best_global_idx], X_train_raw, modalities, device)
                    youden_threshold, youden_j = find_youden_threshold(y_train, train_probs_selected[:, 1])
                    final_pred = predict_with_youden_threshold(final_probs, youden_threshold)
                    print(f"    [{method_name}] {group_name}: Youden阈值={youden_threshold:.4f}, J={youden_j:.4f}")
                else:
                    final_pred = np.argmax(final_probs, axis=1)

                # 评估
                eval_metrics_nested = evaluate_predictions(final_probs, final_pred, y_test, n_classes)
                eval_metrics_flat = flatten_metrics(eval_metrics_nested)

                # 保存预测结果
                metadata = {
                    'seed': seed,
                    'group': group_name,
                    'method': method_name,
                    'selected_model': selected_model_name,
                    'selected_model_idx': best_global_idx,
                    'selected_model_idx_in_group': best_local_idx,
                    'n_models_in_group': len(model_indices),
                    'models_in_group': [get_combination_display_name(ALL_CANDIDATE_MODELS[idx]) for idx in model_indices],
                    'aic_values': aic_values.tolist(),
                    'bic_values': bic_values.tolist(),
                    'criterion_values': criterion_values.tolist(),
                    'k_list': k_list,
                    'nll_list': [float(v) for v in nll_list],
                    'n_train': n_train_original,
                    'metrics': eval_metrics_nested,
                    'n_classes': n_classes,
                    'class_labels': class_labels,
                    'data_augmentation': data_augmentation,
                    'loss_type': args.loss,
                    'threshold_method': 'youden' if args.youden else 'argmax',
                    'youden_threshold': float(youden_threshold) if youden_threshold is not None else None,
                }

                pred_file, csv_file = save_predictions(
                    predictions_dir, seed, group_name, method_name,
                    sample_ids_test, y_test, final_pred, final_probs,
                    class_labels, metadata
                )
                saved_predictions.append({
                    'seed': seed,
                    'group': group_name,
                    'method': method_name,
                    'npz_file': pred_file,
                    'csv_file': csv_file,
                })

                seed_results[group_name] = {
                    'selected_model': selected_model_name,
                    'selected_model_idx': best_global_idx,
                    'aic_values': aic_values.tolist(),
                    'bic_values': bic_values.tolist(),
                    'k_list': k_list,
                    'nll_list': [float(v) for v in nll_list],
                    'model_indices': model_indices,
                    'metrics': eval_metrics_flat,
                    'metrics_nested': eval_metrics_nested,
                    'predictions_file': pred_file,
                    'predictions_csv': csv_file,
                    'youden_threshold': float(youden_threshold) if youden_threshold is not None else None,
                }

                print(f"    [{method_name}] {group_name}: "
                      f"Acc={eval_metrics_nested['accuracy']:.4f}, "
                      f"F1={eval_metrics_nested['macro']['f1']:.4f}, "
                      f"AUC={eval_metrics_nested['macro']['roc_auc']:.4f}, "
                      f"选择={selected_model_name}")

            all_results[method_name][seed] = seed_results

        # 保存scaler
        scaler_path = os.path.join(seed_model_dir, "scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)

    # 5. 汇总结果
    print(f"\n{'=' * 80}")
    print("5. 汇总实验结果...")
    print(f"{'=' * 80}")

    flat_metrics_names = [
        'accuracy', 'log_loss', 'mse', 'mae',
        'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_macro', 'prauc_macro',
        'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_weighted', 'prauc_weighted',
        'precision_micro', 'recall_micro', 'f1_micro', 'roc_auc_micro', 'prauc_micro'
    ]

    for method_name in run_methods:
        print(f"\n{'=' * 60}")
        print(f"  {method_name} 方法结果汇总")
        print(f"{'=' * 60}")

        all_groups_results = {}

        for group_name in COMBINATION_GROUPS.keys():
            group_metrics = {metric: [] for metric in flat_metrics_names}
            model_selection_counts = {}  # 统计各模型被选中次数
            group_thresholds = []

            for seed in seeds:
                if seed in all_results[method_name]:
                    seed_result = all_results[method_name][seed]
                    if group_name in seed_result:
                        metrics = seed_result[group_name]['metrics']
                        for metric in flat_metrics_names:
                            if metric in metrics:
                                group_metrics[metric].append(metrics[metric])
                        sel_model = seed_result[group_name].get('selected_model', '')
                        model_selection_counts[sel_model] = model_selection_counts.get(sel_model, 0) + 1
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

            summary['model_selection_counts'] = model_selection_counts

            if group_thresholds:
                summary['youden_threshold_mean'] = float(np.mean(group_thresholds))
                summary['youden_threshold_std'] = float(np.std(group_thresholds))

            all_groups_results[group_name] = summary

            print(f"\n  {group_name}:")
            print(f"    候选模型数: {len(COMBINATION_GROUPS[group_name])}")
            if 'model_selection_counts' in summary:
                print(f"    模型被选次数: {summary['model_selection_counts']}")
            print(f"    F1 (macro): {summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
            print(f"    AUC (macro): {summary['roc_auc_macro']['mean']:.4f}±{summary['roc_auc_macro']['std']:.4f}")
            print(f"    Accuracy: {summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}")

        # 保存该方法的结果
        method_label = method_name.lower()

        # 汇总表
        summary_rows = []
        for group_name, summary in all_groups_results.items():
            row = {'Group': group_name, 'n_models': len(COMBINATION_GROUPS[group_name])}
            for metric in flat_metrics_names:
                if metric in summary:
                    row[metric] = f"{summary[metric]['mean']:.4f}±{summary[metric]['std']:.4f}"
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(results_dir, f"{method_label}_all_groups_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  汇总结果保存至: {summary_path}")

        # 详细结果表
        detailed_rows = []
        for seed in seeds:
            if seed not in all_results[method_name]:
                continue
            for group_name, result in all_results[method_name][seed].items():
                row = {'seed': seed, 'group': group_name}
                row['selected_model'] = result.get('selected_model', '')
                for metric, value in result['metrics'].items():
                    row[metric] = value
                row['predictions_file'] = result.get('predictions_file', '')
                row['predictions_csv'] = result.get('predictions_csv', '')
                row['youden_threshold'] = result.get('youden_threshold', '')
                detailed_rows.append(row)

        detailed_df = pd.DataFrame(detailed_rows)
        detailed_path = os.path.join(results_dir, f"{method_label}_all_groups_detailed.csv")
        detailed_df.to_csv(detailed_path, index=False)
        print(f"  详细结果保存至: {detailed_path}")

        # 模型选择统计表
        selection_rows = []
        for seed in seeds:
            if seed not in all_results[method_name]:
                continue
            for group_name, result in all_results[method_name][seed].items():
                selection_rows.append({
                    'seed': seed,
                    'group': group_name,
                    'selected_model': result.get('selected_model', ''),
                    'selected_model_idx': result.get('selected_model_idx', -1),
                })

        selection_df = pd.DataFrame(selection_rows)
        selection_path = os.path.join(results_dir, f"{method_label}_model_selection.csv")
        selection_df.to_csv(selection_path, index=False)
        print(f"  模型选择记录保存至: {selection_path}")

        # AIC/BIC信息表（每个seed保存一次）
        aic_bic_rows = []
        for seed in seeds:
            if seed not in all_results[method_name]:
                continue
            seed_result = all_results[method_name][seed]
            first_group = list(seed_result.keys())[0] if seed_result else None
            if first_group and 'aic_values' in seed_result[first_group]:
                r = seed_result[first_group]
                for m_idx, modalities in enumerate(ALL_CANDIDATE_MODELS):
                    aic_bic_rows.append({
                        'seed': seed,
                        'model_index': m_idx,
                        'model_name': get_combination_display_name(modalities),
                        'k': r['k_list'][m_idx],
                        'nll_sum': r['nll_list'][m_idx],
                        'aic_values': r['aic_values'][m_idx],
                        'bic_values': r['bic_values'][m_idx],
                    })

        if aic_bic_rows:
            aic_bic_df = pd.DataFrame(aic_bic_rows)
            aic_bic_path = os.path.join(results_dir, f"{method_label}_aic_bic_details.csv")
            aic_bic_df.to_csv(aic_bic_path, index=False)
            print(f"  AIC/BIC详情保存至: {aic_bic_path}")

        # JSON格式完整结果
        results_json = {
            'method': method_name,
            'candidate_models': [get_combination_display_name(m) for m in ALL_CANDIDATE_MODELS],
            'combination_groups': {k: [get_combination_display_name(ALL_CANDIDATE_MODELS[i]) for i in v]
                                   for k, v in COMBINATION_GROUPS.items()},
            'summary': {
                group: {
                    'n_models': len(COMBINATION_GROUPS[group]),
                    'metrics': {
                        metric: {'mean': summary[metric]['mean'], 'std': summary[metric]['std']}
                        for metric in flat_metrics_names if metric in summary
                    }
                }
                for group, summary in all_groups_results.items()
            },
            'data_augmentation': data_augmentation,
            'loss_type': args.loss,
            'threshold_method': 'youden' if args.youden else 'argmax',
            'n_splits': args.n_splits,
            'n_classes': n_classes,
            'feature_dims': feature_dims,
            'total_samples': len(y),
        }

        results_json_path = os.path.join(results_dir, f"{method_label}_results.json")
        with open(results_json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"  JSON结果保存至: {results_json_path}")

    # 6. 生成跨方法对比表
    if len(run_methods) == 2:
        print(f"\n{'=' * 80}")
        print("6. SAIC vs SBIC 对比")
        print(f"{'=' * 80}")

        compare_rows = []
        for group_name in COMBINATION_GROUPS.keys():
            row = {'Group': group_name}
            for method_name in run_methods:
                method_label = method_name.lower()
                method_summary_path = os.path.join(results_dir, f"{method_label}_all_groups_summary.csv")
                if os.path.exists(method_summary_path):
                    method_df = pd.read_csv(method_summary_path)
                    group_row = method_df[method_df['Group'] == group_name]
                    if len(group_row) > 0:
                        for col in ['accuracy', 'f1_macro', 'roc_auc_macro', 'prauc_macro']:
                            if col in group_row.columns:
                                row[f'{method_name}_{col}'] = group_row[col].values[0]
            compare_rows.append(row)

        if compare_rows:
            compare_df = pd.DataFrame(compare_rows)
            compare_path = os.path.join(results_dir, "saic_vs_sbic_comparison.csv")
            compare_df.to_csv(compare_path, index=False)
            print(f"  对比表保存至: {compare_path}")

    # 预测文件索引
    predictions_index_df = pd.DataFrame(saved_predictions)
    predictions_index_path = os.path.join(results_dir, "predictions_index.csv")
    predictions_index_df.to_csv(predictions_index_path, index=False)
    print(f"\n  预测文件索引保存至: {predictions_index_path}")

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

    # 实验配置
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'n_splits': args.n_splits,
        'test_size': args.test_size,
        'random_seed_base': args.random_seed_base,
        'data_augmentation': data_augmentation,
        'loss_type': args.loss,
        'threshold_method': 'youden' if args.youden else 'argmax',
        'methods': run_methods,
        'n_classes': n_classes,
        'feature_dims': feature_dims,
        'total_samples': len(y),
        'candidate_models': [get_combination_display_name(m) for m in ALL_CANDIDATE_MODELS],
        'combination_groups': {k: [get_combination_display_name(ALL_CANDIDATE_MODELS[i]) for i in v]
                               for k, v in COMBINATION_GROUPS.items()},
    }

    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  实验配置保存至: {config_path}")

    print(f"\n{'=' * 80}")
    print("TCGA HRD数据 SAIC/SBIC 模型选择基线评估完成！")
    print(f"结果目录: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
