"""
CAGMF-Net 单模型评估（无模型平均） - TCGA HRD数据专用版
复用主实验(cagmf_net_ma_cv_addeval_HRD.py)已训练的候选模型和划分索引，
无需重新训练，直接加载模型评估CAGMF-Net在8种模态组合下的表现。

用法:
  python baseline_compare/cagmf_net_baseline_HRD.py \
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
import warnings
import json
from tqdm import tqdm
import argparse

warnings.filterwarnings('ignore')

sys.path.append('.')
from utils import evaluate_predictions


# ======================== 模型定义（CAGMF-Net 架构） ========================
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

MODALITY_NAME_MAP = {
    'clin': 'Clinical',
    'snv': 'SNV',
    'cnv': 'CNA',
    'mrna': 'RNA'
}


def get_model_name(modalities):
    return '+'.join([MODALITY_NAME_MAP[mod] for mod in modalities])


# combo（cagmf_net组合名）→ 主实验 ALL_CANDIDATE_MODELS 下标映射
# 因 cagmf_net 的组合名/顺序与主实验不同（如 Clinical+RNA+CNA ↔ 主实验 Clinical+CNA+RNA），
# 按模态集合相等匹配到正确的模型文件
MODALITY_COMBINATION_TO_INDEX = {
    frozenset(mods): idx for idx, mods in enumerate(ALL_CANDIDATE_MODELS)
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


# ======================== 预测 ========================
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
    parser = argparse.ArgumentParser(description='TCGA HRD数据 CAGMF-Net 单模型评估（加载主实验候选模型，无训练）')
    parser.add_argument('--main_exp_dir', type=str,
                        default='./baseline_compare/tcga_hrd_layersplit_smote_youden',
                        help='主实验(cagmf_net_ma_cv_addeval_HRD)的输出目录（包含saved_models和split_indices）')
    parser.add_argument('--output_dir', type=str, default='./eval_results/tcga_hrd_cagmf_single',
                        help='输出结果目录')
    parser.add_argument('--n_splits', type=int, default=None,
                        help='使用的划分次数，默认None表示读取主实验配置（用于测试时可限制seed数）')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备，默认自动选择')
    parser.add_argument('--youden', action='store_true',
                        help='使用Youden阈值选取方法，已弃用（默认读取主实验配置，一般为youden）')
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

    # --n_splits 覆盖限制 seed 数（用于测试）
    if args.n_splits is not None:
        n_splits = args.n_splits

    # 阈值方法：--argmax 强制argmax，--youden 向后兼容
    if args.argmax:
        threshold_method = 'argmax'
    elif args.youden:
        threshold_method = 'youden'

    print("=" * 80)
    print("TCGA HRD数据 CAGMF-Net 单模型评估（加载主实验候选模型）")
    print(f"主实验目录: {args.main_exp_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"划分次数: {n_splits}")
    print(f"阈值方法: {threshold_method}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    print(f"模态组合数: {len(MODALITY_COMBINATIONS)}")

    # 路径
    saved_models_dir = os.path.join(args.main_exp_dir, "saved_models")
    split_dir = os.path.join(args.main_exp_dir, "split_indices_tcga_hrd")

    if not os.path.exists(saved_models_dir):
        raise FileNotFoundError(f"模型目录不存在: {saved_models_dir}")
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"划分目录不存在: {split_dir}")

    # 获取可用seed列表（模型目录 ∩ 划分文件）
    available_seeds = sorted([
        int(d.split('_')[1]) for d in os.listdir(saved_models_dir)
        if d.startswith('seed_') and os.path.isdir(os.path.join(saved_models_dir, d))
    ])
    split_files = [f for f in os.listdir(split_dir) if f.endswith('.npz')]
    split_seeds = sorted([int(f.split('_')[1]) for f in split_files])
    available_seeds = sorted(set(available_seeds) & set(split_seeds))

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
    os.makedirs(results_dir, exist_ok=True)

    # 3. 遍历可用seed
    print(f"\n3. 开始评估，共 {len(available_seeds)} 个划分...")
    all_results = {}

    for i, seed in enumerate(tqdm(available_seeds, desc="处理种子")):
        # 加载划分索引
        split_path = os.path.join(split_dir, f"seed_{seed}_split.npz")
        split_data = np.load(split_path, allow_pickle=True)
        train_idx = split_data['train_idx']
        test_idx = split_data['test_idx']

        # 加载预训练候选模型
        model_dir = os.path.join(saved_models_dir, f"seed_{seed}")
        models = load_models_and_scaler(model_dir, device)

        y_train = y[train_idx]
        y_test = y[test_idx]

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

        seed_results = {}

        # 对每个模态组合，映射到主实验对应模型文件下标
        for combo_name, modalities in MODALITY_COMBINATIONS.items():
            idx = MODALITY_COMBINATION_TO_INDEX[frozenset(modalities)]
            test_probs = test_probs_all[idx]
            train_probs = train_probs_all[idx]

            # Youden阈值
            youden_threshold = None
            if threshold_method == 'youden' and n_classes == 2:
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

        all_results[seed] = seed_results

        # 进度打印
        if (i + 1) % 10 == 0 or i == 0:
            sample_combo = list(MODALITY_COMBINATIONS.keys())[0]
            if sample_combo in seed_results:
                m = seed_results[sample_combo]['metrics']
                print(f"\n  Seed {seed}: {sample_combo} "
                      f"Acc={m['accuracy']:.4f} F1_macro={m['f1_macro']:.4f} AUC={m['roc_auc_macro']:.4f}")

    # 4. 汇总结果
    print(f"\n{'=' * 80}")
    print("4. 汇总实验结果...")
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

        for seed in available_seeds:
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

    # 5. 保存结果
    print(f"\n5. 保存结果...")

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
    for seed in available_seeds:
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
    summary_dict = {}
    for combo, summary in all_summaries.items():
        s = {
            metric: {'mean': v['mean'], 'std': v['std']}
            for metric, v in summary.items()
            if isinstance(v, dict) and 'mean' in v
        }
        if 'youden_threshold_mean' in summary:
            s['youden_threshold_mean'] = summary['youden_threshold_mean']
            s['youden_threshold_std'] = summary['youden_threshold_std']
        summary_dict[combo] = s

    results_json = {
        'modality_combinations': {k: v for k, v in MODALITY_COMBINATIONS.items()},
        'model': 'CAGMF-Net',
        'summary': summary_dict,
        'threshold_method': threshold_method,
        'n_splits': len(available_seeds),
        'n_classes': n_classes,
        'feature_dims': feature_dims,
        'total_samples': len(y),
        'source_main_exp': args.main_exp_dir,
    }

    results_json_path = os.path.join(results_dir, "cagmf_single_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  JSON结果保存至: {results_json_path}")

    # 实验配置
    config = {
        'source_main_exp': args.main_exp_dir,
        'output_dir': args.output_dir,
        'threshold_method': threshold_method,
        'n_splits': len(available_seeds),
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
    print("TCGA HRD数据 CAGMF-Net 单模型评估完成！（加载主实验候选模型）")
    print(f"结果目录: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
