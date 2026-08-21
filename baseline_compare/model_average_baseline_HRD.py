"""
模型平均基线评估 - TCGA HRD数据专用版
复用主实验(cagmf_net_ma_cv_addeval_HRD.py)已训练的候选模型和划分索引，
无需重新训练，直接加载模型进行三种模型平均评估:
  - EW   (Equal Weight):  等权重，每个候选模型 w = 1/n
  - SAIC (Smoothed AIC):  w_k = exp(-AIC_k/2) / Σ_j exp(-AIC_j/2)
  - SBIC (Smoothed BIC):  w_k = exp(-BIC_k/2) / Σ_j exp(-BIC_j/2)

AIC/BIC 计算方法与 model_choose_baseline_HRD.py 一致:
  AIC = 2k + 2*NLL,  BIC = k*ln(n) + 2*NLL
  其中 k = 输入特征维度之和, NLL = 训练集负对数似然之和

用法:
  python baseline_compare/model_average_baseline_HRD.py \
      --main_exp_dir ./baseline_compare/tcga_hrd_layersplit_smote_youden
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
import warnings
import json
import pickle
from tqdm import tqdm
import argparse

warnings.filterwarnings('ignore')

sys.path.append('.')
from utils import evaluate_predictions


# ======================== 模型定义（必须与主实验一致，用于加载state_dict） ========================
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
    def __init__(self, in_dim, hidden, n_class):
        super().__init__()
        self.mlp = MLP(in_dim, hidden)
        self.classifier = nn.Linear(hidden, n_class)

    def forward(self, x):
        z = self.mlp(x)
        return self.classifier(z)


# ======================== AIC/BIC 计算（与 model_choose_baseline_HRD.py 一致） ========================
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
    计算一组模型的AIC和BIC值

    AIC = 2k + 2*NLL
    BIC = k*ln(n) + 2*NLL

    Args:
        k_list: 各模型参数量（输入特征维度之和）列表
        nll_list: 各模型NLL之和列表
        n_samples: 训练样本数

    Returns:
        aic_values: AIC 数组
        bic_values: BIC 数组
    """
    k_arr = np.array(k_list, dtype=np.float64)
    nll_arr = np.array(nll_list, dtype=np.float64)

    aic_values = 2.0 * k_arr + 2.0 * nll_arr
    bic_values = k_arr * np.log(n_samples) + 2.0 * nll_arr

    return aic_values, bic_values


def _stable_softmax_neg_half(values):
    """softmax(-values/2) with numerical stability"""
    v = np.array(values, dtype=np.float64)
    scaled = -v / 2.0
    scaled = scaled - np.max(scaled)  # subtract max for stability
    exp_vals = np.exp(scaled)
    return exp_vals / np.sum(exp_vals)


def compute_saic_sbic_weights(aic_values, bic_values):
    """
    计算SAIC和SBIC权重（softmax归一化）

    SAIC: w_i = exp(-AIC_i/2) / Σ_j exp(-AIC_j/2)
    SBIC: w_i = exp(-BIC_i/2) / Σ_j exp(-BIC_j/2)

    Args:
        aic_values: AIC 数组
        bic_values: BIC 数组

    Returns:
        saic_weights: SAIC权重
        sbic_weights: SBIC权重
    """
    saic_weights = _stable_softmax_neg_half(aic_values)
    sbic_weights = _stable_softmax_neg_half(bic_values)
    return saic_weights, sbic_weights


# ======================== 常量（与主实验一致） ========================
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

# 模型平均方法: 等权重(EW)、SAIC软权重、SBIC软权重
METHODS = ['EW', 'SAIC', 'SBIC']

MODALITY_NAME_MAP = {
    'clin': 'Clinical',
    'snv': 'SNV',
    'cnv': 'CNA',
    'mrna': 'RNA'
}


def get_model_name(modalities):
    return '+'.join([MODALITY_NAME_MAP[mod] for mod in modalities])


# ======================== 模型加载 ========================
def build_model(model_type, modalities, dims, n_classes, hidden=32):
    """根据metadata构建模型结构（hidden由state_dict推导，默认32与主实验一致）"""
    if model_type == 'SingleModalMLP':
        mod_dim = dims[modalities[0]]
        model = SingleModalMLP(mod_dim, hidden, n_classes)
    else:
        model = CAGMFNet(dims, hidden, n_classes)
    return model


def load_models_and_scaler(model_dir, device):
    """
    从一个seed的模型目录加载所有候选模型

    Returns:
        models: list of 8 loaded models
    """
    models = []

    for modalities in ALL_CANDIDATE_MODELS:
        model_name = get_model_name(modalities)
        model_path = os.path.join(model_dir, f"{model_name}_model.pth")
        metadata_path = os.path.join(model_dir, f"{model_name}_model_metadata.json")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        model_type = metadata['model_type']
        dims = metadata['dims']
        n_classes = metadata['n_classes']

        # 先加载state_dict，从classifier.weight的shape推导hidden维度（主实验hidden=32）
        state_dict = torch.load(model_path, map_location=device)
        hidden = state_dict['classifier.weight'].shape[1]

        model = build_model(model_type, modalities, dims, n_classes, hidden=hidden)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)

    # scaler pickle 可能存在 numpy 版本不兼容问题（numpy 1.x vs 2.x），
    # 改为在外部从训练数据重新计算（结果完全一致，StandardScaler 是确定性计算）
    return models


# ======================== 数据加载（与主实验一致） ========================
def standardize_sample_id(sample_id):
    if isinstance(sample_id, str):
        if sample_id.endswith('-01') or sample_id.endswith('-02') or sample_id.endswith('-03'):
            sample_id = sample_id.rsplit('-', 1)[0]
    return sample_id


def load_hrd_data(data_dir, return_sample_ids=False):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

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


# ======================== 预测 ========================
def predict_model(model, X_dict, modalities, device):
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
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_class1)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    return best_threshold, youden_j[best_idx]


def predict_with_youden_threshold(probs, threshold):
    preds = np.zeros(len(probs), dtype=int)
    preds[probs[:, 1] >= threshold] = 1
    return preds


# ======================== 结果展平/保存 ========================
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


def save_predictions(predictions_dir, seed, group_name, method_name,
                     sample_ids_test, y_true, y_pred, y_probs,
                     class_labels, metadata):
    os.makedirs(predictions_dir, exist_ok=True)

    safe_group_name = group_name.replace('+', '_')
    pred_file = os.path.join(predictions_dir, f"seed_{seed}_{safe_group_name}_{method_name}_predictions.npz")

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

    csv_file = os.path.join(predictions_dir, f"seed_{seed}_{safe_group_name}_{method_name}_predictions.csv")
    df.to_csv(csv_file, index=False)

    return pred_file, csv_file


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(
        description='等权重模型平均基线评估 - 复用主实验已训练模型，无需重新训练'
    )
    parser.add_argument('--main_exp_dir', type=str,
                        default='./baseline_compare/tcga_hrd_layersplit_smote_youden',
                        help='主实验(cagmf_net_ma_cv_addeval_HRD)的输出目录（包含saved_models和split_indices）')
    parser.add_argument('--output_dir', type=str,
                        default='./eval_results/tcga_hrd_model_average',
                        help='本实验输出目录')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备，默认自动选择')
    parser.add_argument('--youden', action='store_true',
                        help='使用Youden阈值选取方法，已废弃（默认读取主实验配置，一般为youden）')
    parser.add_argument('--argmax', action='store_true',
                        help='使用0.5 argmax阈值，覆盖主实验配置的youden阈值')

    args = parser.parse_args()

    # 读取主实验配置，确定数据目录和阈值方法（默认youden）
    exp_config_path = os.path.join(args.main_exp_dir, "results", "experiment_config.json")
    if os.path.exists(exp_config_path):
        with open(exp_config_path, 'r') as f:
            exp_config = json.load(f)
        data_dir = exp_config.get('data_dir', './data/tcga')
        n_splits = exp_config.get('n_splits', 100)
        random_seed_base = exp_config.get('random_seed_base', 42)
        n_classes = exp_config.get('n_classes', 2)
        threshold_method = exp_config.get('threshold_method', 'youden')
        print(f"从主实验配置读取: data_dir={data_dir}, n_splits={n_splits}, seed_base={random_seed_base}")
    else:
        data_dir = './data/tcga'
        n_splits = 100
        random_seed_base = 42
        n_classes = 2
        threshold_method = 'youden'
        print(f"未找到主实验配置，使用默认值: data_dir={data_dir}, n_splits={n_splits}")

    # 阈值方法：--argmax 强制argmax，--youden 向后兼容
    if args.argmax:
        threshold_method = 'argmax'
    elif args.youden:
        threshold_method = 'youden'

    print("=" * 80)
    print("TCGA HRD数据 等权重(Equal Weight)模型平均基线评估")
    print(f"主实验目录: {args.main_exp_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"阈值方法: {threshold_method}")
    print("注意: 等权重=对组内每个候选模型赋予相同权重 w=1/n")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 路径
    saved_models_dir = os.path.join(args.main_exp_dir, "saved_models")
    split_dir = os.path.join(args.main_exp_dir, "split_indices_tcga_hrd")

    if not os.path.exists(saved_models_dir):
        raise FileNotFoundError(f"模型目录不存在: {saved_models_dir}")
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"划分目录不存在: {split_dir}")

    # 获取可用seed列表
    available_seeds = sorted([
        int(d.split('_')[1]) for d in os.listdir(saved_models_dir)
        if d.startswith('seed_') and os.path.isdir(os.path.join(saved_models_dir, d))
    ])
    split_files = [f for f in os.listdir(split_dir) if f.endswith('.npz')]
    split_seeds = sorted([int(f.split('_')[1]) for f in split_files])

    # 取交集
    available_seeds = sorted(set(available_seeds) & set(split_seeds))
    n_available = len(available_seeds)

    # 限制为n_splits个
    expected_seeds = list(range(random_seed_base, random_seed_base + n_splits))
    available_seeds = [s for s in expected_seeds if s in available_seeds]

    print(f"可用的模型/划分: {len(available_seeds)} 个seed (期望 {n_splits})")
    if len(available_seeds) < n_splits:
        print(f"警告: 只有 {len(available_seeds)}/{n_splits} 个seed可用，将使用现有的进行实验")

    # 1. 加载数据
    print("\n1. 加载TCGA HRD数据...")
    X_dict, y, le_y, feature_dims, sample_ids = load_hrd_data(
        data_dir, return_sample_ids=True
    )
    n_classes = len(np.unique(y))
    class_labels = [int(c) for c in le_y.classes_]
    print(f"总样本数: {len(y)}, 类别数: {n_classes}")
    print(f"各模态维度: {feature_dims}")

    # 2. 创建存储目录
    print("\n2. 初始化结果存储...")
    results_dir = os.path.join(args.output_dir, "results")
    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # 3. 遍历可用seed
    print(f"\n3. 开始评估（EW / SAIC / SBIC 三种模型平均方法），共 {len(available_seeds)} 个划分...")
    all_results = {method: {} for method in METHODS}
    saved_predictions = []

    for i, seed in enumerate(tqdm(available_seeds, desc="处理种子")):
        # 加载划分索引
        split_path = os.path.join(split_dir, f"seed_{seed}_split.npz")
        split_data = np.load(split_path, allow_pickle=True)
        train_idx = split_data['train_idx']
        test_idx = split_data['test_idx']

        # 加载预训练模型
        model_dir = os.path.join(saved_models_dir, f"seed_{seed}")
        models = load_models_and_scaler(model_dir, device)

        y_train = y[train_idx]
        y_test = y[test_idx]
        n_train_original = len(y_train)
        sample_ids_test = [sample_ids[idx] for idx in test_idx]

        # 从训练数据计算scaler（与训练时一致，避免numpy版本不兼容）
        X_train_raw = {}
        X_test_raw = {}
        for mod in X_dict.keys():
            scaler = StandardScaler()
            scaler.fit(X_dict[mod][train_idx])
            X_train_raw[mod] = scaler.transform(X_dict[mod][train_idx])
            X_test_raw[mod] = scaler.transform(X_dict[mod][test_idx])

        # 获取所有候选模型的测试集+训练集预测
        test_probs_all = []
        train_probs_all = []
        for m, modalities in enumerate(ALL_CANDIDATE_MODELS):
            X_test_sub = {mod: X_test_raw[mod] for mod in modalities}
            probs_te = predict_model(models[m], X_test_sub, modalities, device)
            test_probs_all.append(probs_te)

            X_train_sub = {mod: X_train_raw[mod] for mod in modalities}
            probs_tr = predict_model(models[m], X_train_sub, modalities, device)
            train_probs_all.append(probs_tr)

        test_probs_all = np.array(test_probs_all)
        train_probs_all = np.array(train_probs_all)

        # =============================================
        # 计算 AIC/BIC（与 model_choose_baseline_HRD.py 一致）
        # k = 输入特征维度之和, NLL = 训练集负对数似然之和
        # =============================================
        k_list = []
        nll_list = []
        for m, modalities in enumerate(ALL_CANDIDATE_MODELS):
            k = sum(feature_dims[mod] for mod in modalities)
            k_list.append(k)
            nll = compute_nll_sum(models[m], X_train_raw, y_train, modalities, device)
            nll_list.append(nll)

        aic_values, bic_values = compute_aic_bic_values(k_list, nll_list, n_train_original)

        seed_results = {}

        for group_name, model_indices in COMBINATION_GROUPS.items():
            n_models = len(model_indices)

            # 组内 AIC/BIC 子集
            group_aic = aic_values[model_indices]
            group_bic = bic_values[model_indices]

            # 预计算各方法的权重
            method_weights = {}
            method_weights['EW'] = np.ones(n_models) / n_models
            method_weights['SAIC'] = _stable_softmax_neg_half(group_aic)
            method_weights['SBIC'] = _stable_softmax_neg_half(group_bic)

            for method in METHODS:
                weights = method_weights[method]

                # 加权平均预测（测试集）
                group_test_probs = test_probs_all[model_indices, :, :]
                final_probs = np.zeros_like(group_test_probs[0])
                for w, probs in zip(weights, group_test_probs):
                    final_probs += w * probs

                # 预测类别
                youden_threshold = None
                if threshold_method == 'youden' and n_classes == 2:
                    # 训练集上的加权平均（用于Youden阈值）
                    group_train_probs = train_probs_all[model_indices, :, :]
                    train_ensemble = np.zeros_like(group_train_probs[0])
                    for w, probs in zip(weights, group_train_probs):
                        train_ensemble += w * probs
                    youden_threshold, youden_j = find_youden_threshold(y_train, train_ensemble[:, 1])
                    final_pred = predict_with_youden_threshold(final_probs, youden_threshold)
                else:
                    final_pred = np.argmax(final_probs, axis=1)

                # 评估
                eval_metrics_nested = evaluate_predictions(final_probs, final_pred, y_test, n_classes)
                eval_metrics_flat = flatten_metrics(eval_metrics_nested)

                # 保存预测结果
                metadata = {
                    'seed': seed,
                    'group': group_name,
                    'method': method,
                    'n_models': n_models,
                    'models': [get_model_name(ALL_CANDIDATE_MODELS[idx]) for idx in model_indices],
                    'weights': weights.tolist(),
                    'aic_values': aic_values.tolist(),
                    'bic_values': bic_values.tolist(),
                    'k_list': k_list,
                    'nll_list': [float(v) for v in nll_list],
                    'n_train': n_train_original,
                    'metrics': eval_metrics_nested,
                    'n_classes': n_classes,
                    'class_labels': class_labels,
                    'threshold_method': threshold_method,
                    'youden_threshold': float(youden_threshold) if youden_threshold is not None else None,
                }

                pred_file, csv_file = save_predictions(
                    predictions_dir, seed, group_name, method,
                    sample_ids_test, y_test, final_pred, final_probs,
                    class_labels, metadata
                )
                saved_predictions.append({
                    'seed': seed,
                    'group': group_name,
                    'method': method,
                    'npz_file': pred_file,
                    'csv_file': csv_file,
                })

                if group_name not in seed_results:
                    seed_results[group_name] = {}
                seed_results[group_name][method] = {
                    'weights': weights.tolist(),
                    'model_indices': model_indices,
                    'aic_values': aic_values.tolist(),
                    'bic_values': bic_values.tolist(),
                    'k_list': k_list,
                    'nll_list': [float(v) for v in nll_list],
                    'metrics': eval_metrics_flat,
                    'metrics_nested': eval_metrics_nested,
                    'predictions_file': pred_file,
                    'predictions_csv': csv_file,
                    'youden_threshold': float(youden_threshold) if youden_threshold is not None else None,
                }

        # 将seed_results按method重组到all_results
        for group_name, group_methods in seed_results.items():
            for method, result in group_methods.items():
                if seed not in all_results[method]:
                    all_results[method][seed] = {}
                all_results[method][seed][group_name] = result

        # 进度打印
        if (i + 1) % 10 == 0 or i == 0:
            first_group = list(COMBINATION_GROUPS.keys())[-1]  # Full组合
            if first_group in seed_results:
                for method in METHODS:
                    if method in seed_results[first_group]:
                        m = seed_results[first_group][method]['metrics']
                        weights = seed_results[first_group][method]['weights']
                        w_str = ', '.join([f'{w:.3f}' for w in weights[:3]])
                        if len(weights) > 3:
                            w_str += ', ...'
                        print(f"  Seed {seed} {first_group}({method:4s}) "
                              f"Acc={m['accuracy']:.4f} F1={m['f1_macro']:.4f} AUC={m['roc_auc_macro']:.4f} "
                              f"W=[{w_str}]")

    # 4. 汇总结果（按方法分别汇总）
    print(f"\n{'=' * 80}")
    print("4. 汇总实验结果（EW / SAIC / SBIC）...")
    print(f"{'=' * 80}")

    flat_metrics_names = [
        'accuracy', 'log_loss', 'mse', 'mae',
        'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_macro', 'prauc_macro',
        'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_weighted', 'prauc_weighted',
        'precision_micro', 'recall_micro', 'f1_micro', 'roc_auc_micro', 'prauc_micro'
    ]

    all_methods_summary = {}  # {method: {group_name: summary}}

    for method in METHODS:
        print(f"\n{'=' * 60}")
        print(f"  {method} 方法结果汇总")
        print(f"{'=' * 60}")

        all_groups_results = {}

        for group_name in COMBINATION_GROUPS.keys():
            group_metrics = {metric: [] for metric in flat_metrics_names}
            group_thresholds = []

            for seed in available_seeds:
                if (seed in all_results[method] and
                        group_name in all_results[method][seed]):
                    metrics = all_results[method][seed][group_name]['metrics']
                    for metric in flat_metrics_names:
                        if metric in metrics:
                            group_metrics[metric].append(metrics[metric])
                    th = all_results[method][seed][group_name].get('youden_threshold')
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

            if group_thresholds:
                summary['youden_threshold_mean'] = float(np.mean(group_thresholds))
                summary['youden_threshold_std'] = float(np.std(group_thresholds))

            all_groups_results[group_name] = summary

            n_models = len(COMBINATION_GROUPS[group_name])
            print(f"\n  {group_name} ({method}, n={n_models}):")
            print(f"    F1 (macro): {summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
            print(f"    AUC (macro): {summary['roc_auc_macro']['mean']:.4f}±{summary['roc_auc_macro']['std']:.4f}")
            print(f"    Accuracy: {summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}")

        all_methods_summary[method] = all_groups_results

    # 5. 保存结果
    print(f"\n{'=' * 80}")
    print("5. 保存结果...")
    print(f"{'=' * 80}")

    for method in METHODS:
        method_label = method.lower()
        all_groups_results = all_methods_summary[method]

        # 汇总表
        summary_rows = []
        for group_name, summary in all_groups_results.items():
            row = {'Group': group_name, 'n_models': len(COMBINATION_GROUPS[group_name])}
            for metric in flat_metrics_names:
                if metric in summary:
                    row[metric] = f"{summary[metric]['mean']:.4f}±{summary[metric]['std']:.4f}"
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(results_dir, f"{method_label}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"  [{method}] 汇总结果保存至: {summary_path}")

        # 详细结果表
        detailed_rows = []
        for seed in available_seeds:
            if seed not in all_results[method]:
                continue
            for group_name, result in all_results[method][seed].items():
                row = {'seed': seed, 'group': group_name, 'method': method}
                for metric, value in result['metrics'].items():
                    row[metric] = value
                row['predictions_file'] = result.get('predictions_file', '')
                row['predictions_csv'] = result.get('predictions_csv', '')
                row['weights'] = str(result.get('weights', ''))
                detailed_rows.append(row)

        detailed_df = pd.DataFrame(detailed_rows)
        detailed_path = os.path.join(results_dir, f"{method_label}_detailed.csv")
        detailed_df.to_csv(detailed_path, index=False)
        print(f"  [{method}] 详细结果保存至: {detailed_path}")

        # JSON格式完整结果
        results_json = {
            'method': method,
            'candidate_models': [get_model_name(m) for m in ALL_CANDIDATE_MODELS],
            'combination_groups': {k: [get_model_name(ALL_CANDIDATE_MODELS[i]) for i in v]
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
            'threshold_method': threshold_method,
            'n_splits': len(available_seeds),
            'n_classes': n_classes,
            'feature_dims': feature_dims,
            'total_samples': len(y),
            'source_main_exp': args.main_exp_dir,
        }

        results_json_path = os.path.join(results_dir, f"{method_label}_results.json")
        with open(results_json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"  [{method}] JSON结果保存至: {results_json_path}")

    # 6. 生成跨方法对比表
    print(f"\n{'=' * 80}")
    print("6. EW vs SAIC vs SBIC 对比")
    print(f"{'=' * 80}")

    compare_rows = []
    for group_name in COMBINATION_GROUPS.keys():
        row = {'Group': group_name, 'n_models': len(COMBINATION_GROUPS[group_name])}
        for method in METHODS:
            summary = all_methods_summary[method][group_name]
            for metric in ['accuracy', 'f1_macro', 'roc_auc_macro', 'prauc_macro']:
                if metric in summary:
                    row[f'{method}_{metric}'] = f"{summary[metric]['mean']:.4f}±{summary[metric]['std']:.4f}"
        compare_rows.append(row)

    compare_df = pd.DataFrame(compare_rows)
    compare_path = os.path.join(results_dir, "methods_comparison.csv")
    compare_df.to_csv(compare_path, index=False)
    print(f"  方法对比表保存至: {compare_path}")

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

    # 实验配置
    config = {
        'methods': METHODS,
        'source_main_exp': args.main_exp_dir,
        'output_dir': args.output_dir,
        'threshold_method': threshold_method,
        'n_splits': len(available_seeds),
        'n_classes': n_classes,
        'n_available_seeds': len(available_seeds),
        'feature_dims': feature_dims,
        'total_samples': len(y),
        'candidate_models': [get_model_name(m) for m in ALL_CANDIDATE_MODELS],
        'combination_groups': {k: [get_model_name(ALL_CANDIDATE_MODELS[i]) for i in v]
                               for k, v in COMBINATION_GROUPS.items()},
    }

    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  实验配置保存至: {config_path}")

    print(f"\n{'=' * 80}")
    print("TCGA HRD数据 模型平均基线评估完成！（EW / SAIC / SBIC）")
    print(f"结果目录: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
