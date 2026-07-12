"""
CAGMF-Net 单模型评估（无模型平均） - TCGA HRD数据专用版
使用TCGA_Clinical_HRD.csv，以patient为ID，HRD_label为标签（二分类）
评估CAGMF-Net在8种模态组合下的表现（每个组合单独训练单模型）
支持SMOTE/Oversample数据增强、Youden阈值选取和Focal Loss
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
MODALITY_COMBINATIONS = {
    'Clinical':              ['clin'],
    'Clinical+RNA':          ['clin', 'mrna'],
    'Clinical+CNA':          ['clin', 'cnv'],
    'Clinical+SNV':          ['clin', 'snv'],
    'Clinical+RNA+CNA':      ['clin', 'mrna', 'cnv'],
    'Clinical+RNA+SNV':      ['clin', 'mrna', 'snv'],
    'Clinical+CNA+SNV':      ['clin', 'cnv', 'snv'],
    'Clinical+RNA+CNA+SNV':  ['clin', 'mrna', 'cnv', 'snv'],
}


# ======================== 数据加载 ========================
def standardize_sample_id(sample_id):
    if isinstance(sample_id, str):
        if sample_id.endswith('-01') or sample_id.endswith('-02') or sample_id.endswith('-03'):
            sample_id = sample_id.rsplit('-', 1)[0]
    return sample_id


def load_hrd_data(data_dir, return_sample_ids=False):
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


# ======================== Youden阈值 ========================
def find_youden_threshold(y_true, y_prob_class1):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_class1)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    return best_threshold, youden_j[best_idx]


def predict_with_youden_threshold(probs, threshold):
    preds = np.zeros(len(probs), dtype=int)
    preds[probs[:, 1] >= threshold] = 1
    return preds


# ======================== 模型训练 ========================
def train_model(X_train_dict, y_train, modalities, dims, n_classes,
                hidden=64, epochs=50, lr=0.001, device=None, seed=42,
                val_ratio=0.1, patience=10, data_augmentation=None,
                loss_type='ce'):
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

    return model


def predict_model(model, X_dict, modalities, device=None):
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


# ======================== 结果展平 ========================
def flatten_metrics(metrics_dict):
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
    parser = argparse.ArgumentParser(description='TCGA HRD数据 CAGMF-Net 单模型评估（无模型平均）')
    parser.add_argument('--data_dir', type=str, default='./data/tcga',
                        help='TCGA数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./eval_results/tcga_hrd_cagmf_single',
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
                        help='损失函数: ce (交叉熵) 或 focal (Focal Loss)，默认ce')
    parser.add_argument('--skip_split_generation', action='store_true',
                        help='跳过划分生成，使用已有划分')

    args = parser.parse_args()

    data_augmentation = None
    if args.smote:
        data_augmentation = 'smote'
    elif args.oversample:
        data_augmentation = 'oversample'

    print("=" * 80)
    print("TCGA HRD数据 CAGMF-Net 单模型评估（无模型平均）")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"划分次数: {args.n_splits}, 测试比例: {args.test_size}")
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

    print(f"模态组合数: {len(MODALITY_COMBINATIONS)}")
    print(f"总实验数: {len(MODALITY_COMBINATIONS)} × {args.n_splits} = {len(MODALITY_COMBINATIONS) * args.n_splits}")

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
    split_dir_name = "split_indices_tcga_hrd_cagmf"
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
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 4. 遍历每个划分
    print(f"\n4. 开始实验，共 {args.n_splits} 个划分...")
    seeds = range(args.random_seed_base, args.random_seed_base + args.n_splits)

    all_results = {}

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

        seed_results = {}

        # 对每个模态组合
        for combo_name, modalities in MODALITY_COMBINATIONS.items():
            dims = {mod: feature_dims[mod] for mod in modalities}
            X_train_sub = {mod: X_train_raw[mod].copy() for mod in modalities}
            X_test_sub = {mod: X_test_raw[mod] for mod in modalities}

            # 训练模型
            model = train_model(
                X_train_sub, y_train, modalities, dims, n_classes,
                device=device, seed=seed,
                data_augmentation=data_augmentation,
                loss_type=args.loss
            )

            # 预测测试集
            test_probs = predict_model(model, X_test_sub, modalities, device)

            # 预测训练集（用于Youden阈值计算）
            train_probs = predict_model(model, X_train_sub, modalities, device)

            # Youden阈值
            youden_threshold = None
            if args.youden and n_classes == 2:
                youden_threshold, youden_j = find_youden_threshold(y_train, train_probs[:, 1])
                y_pred = predict_with_youden_threshold(test_probs, youden_threshold)
            else:
                y_pred = np.argmax(test_probs, axis=1)

            # 评估
            eval_metrics_nested = evaluate_predictions(test_probs, y_pred, y_test, n_classes)
            eval_metrics_flat = flatten_metrics(eval_metrics_nested)

            seed_results[combo_name] = {
                'metrics': eval_metrics_flat,
                'metrics_nested': eval_metrics_nested,
                'youden_threshold': float(youden_threshold) if youden_threshold is not None else None,
            }

        # 保存模型
        seed_model_dir = os.path.join(models_dir, f"seed_{seed}")
        os.makedirs(seed_model_dir, exist_ok=True)

        all_results[seed] = seed_results

        # 进度打印
        if (i + 1) % 10 == 0 or i == 0:
            sample_combo = list(MODALITY_COMBINATIONS.keys())[0]
            if sample_combo in seed_results:
                m = seed_results[sample_combo]['metrics']
                print(f"\n  Seed {seed}: {sample_combo} "
                      f"Acc={m['accuracy']:.4f} F1_macro={m['f1_macro']:.4f} AUC={m['roc_auc_macro']:.4f}")

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

    all_summaries = {}

    for combo_name in MODALITY_COMBINATIONS.keys():
        metrics_accum = {metric: [] for metric in flat_metrics_names}
        thresholds_accum = []

        for seed in seeds:
            if seed in all_results and combo_name in all_results[seed]:
                m = all_results[seed][combo_name]['metrics']
                for metric in flat_metrics_names:
                    if metric in m:
                        metrics_accum[metric].append(m[metric])
                th = all_results[seed][combo_name].get('youden_threshold')
                if th is not None:
                    thresholds_accum.append(th)

        summary = {}
        for metric in flat_metrics_names:
            values = metrics_accum[metric]
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        if thresholds_accum:
            summary['youden_threshold_mean'] = float(np.mean(thresholds_accum))
            summary['youden_threshold_std'] = float(np.std(thresholds_accum))

        all_summaries[combo_name] = summary

    # 打印汇总
    print(f"\n{'=' * 80}")
    print("CAGMF-Net 单模型结果汇总")
    print(f"{'=' * 80}")
    for combo_name in MODALITY_COMBINATIONS.keys():
        s = all_summaries[combo_name]
        if 'accuracy' in s:
            print(f"  {combo_name}: "
                  f"Acc={s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f}, "
                  f"F1={s['f1_macro']['mean']:.4f}±{s['f1_macro']['std']:.4f}, "
                  f"AUC={s['roc_auc_macro']['mean']:.4f}±{s['roc_auc_macro']['std']:.4f}")

    # 6. 保存结果
    print(f"\n6. 保存结果...")

    # 汇总表
    summary_rows = []
    for combo_name in MODALITY_COMBINATIONS.keys():
        s = all_summaries[combo_name]
        row = {'Modality': combo_name}
        for metric in flat_metrics_names:
            if metric in s:
                row[metric] = f"{s[metric]['mean']:.4f}±{s[metric]['std']:.4f}"
        if 'youden_threshold_mean' in s:
            row['youden_threshold'] = f"{s['youden_threshold_mean']:.4f}±{s['youden_threshold_std']:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, "cagmf_single_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  汇总结果保存至: {summary_path}")

    # 详细结果表
    detailed_rows = []
    for seed in seeds:
        if seed not in all_results:
            continue
        for combo_name in MODALITY_COMBINATIONS.keys():
            if combo_name not in all_results[seed]:
                continue
            m = all_results[seed][combo_name]
            row = {'seed': seed, 'modality': combo_name}
            for metric, value in m['metrics'].items():
                row[metric] = value
            row['youden_threshold'] = m.get('youden_threshold', '')
            detailed_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = os.path.join(results_dir, "cagmf_single_detailed.csv")
    detailed_df.to_csv(detailed_path, index=False)
    print(f"  详细结果保存至: {detailed_path}")

    # JSON格式
    results_json = {
        'modality_combinations': {k: v for k, v in MODALITY_COMBINATIONS.items()},
        'model': 'CAGMF-Net',
        'summary': {
            combo: {
                metric: {'mean': v['mean'], 'std': v['std']}
                for metric, v in summary.items()
                if isinstance(v, dict) and 'mean' in v
            }
            for combo, summary in all_summaries.items()
        },
        'data_augmentation': data_augmentation,
        'loss_type': args.loss,
        'threshold_method': 'youden' if args.youden else 'argmax',
        'n_splits': args.n_splits,
        'n_classes': n_classes,
        'feature_dims': feature_dims,
        'total_samples': len(y),
    }

    results_json_path = os.path.join(results_dir, "cagmf_single_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  JSON结果保存至: {results_json_path}")

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
        'n_classes': n_classes,
        'feature_dims': feature_dims,
        'total_samples': len(y),
        'model': 'CAGMF-Net',
        'modality_combinations': {k: v for k, v in MODALITY_COMBINATIONS.items()},
    }

    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  实验配置保存至: {config_path}")

    label_mapping = {
        'class_labels': class_labels,
        'class_codes': list(range(len(class_labels))),
        'n_classes': n_classes,
    }
    label_mapping_path = os.path.join(results_dir, "label_mapping.json")
    with open(label_mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"  标签映射保存至: {label_mapping_path}")

    print(f"\n{'=' * 80}")
    print("TCGA HRD数据 CAGMF-Net 单模型评估完成！")
    print(f"结果目录: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
