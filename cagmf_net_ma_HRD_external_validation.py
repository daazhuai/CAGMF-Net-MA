"""
METABRIC 外部验证脚本
使用 TCGA 上训练的 CAGMF-Net 模型（smote + argmax + CE损失 + CV准则CE）
在 METABRIC 数据集上进行外部验证

数据路径: ./data/metabric/
模型路径: ./eval_results/smote_argmax_ce_ce/
输出路径: ./eval_results/smote_argmax_ce_ce/metabric_external/
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import warnings
import argparse
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.append('.')
from utils import evaluate_predictions, set_seed


# ======================== 模型定义（与训练脚本一致） ========================
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


# ======================== 常量定义 ========================
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


# ======================== METABRIC 数据加载 ========================
def encode_clinical_to_tcga_mapping(clin_df):
    """
    将 METABRIC 临床特征编码为与 TCGA 训练数据一致的数值编码。

    TCGA LabelEncoder 编码（已验证）:
      ER:     {'ER+': 0, 'ER-': 1, 'NA': 2}
      Her2:   {'Her2+': 0, 'Her2-': 1, 'NA': 2}  注意: TCGA 使用 'Her2' (小写 r)
      LN:     {'LN+': 0, 'LN-': 1, 'NA': 2}
      STAGE:  {'NA': 0, 'Stage_I': 1, 'Stage_II': 2, 'Stage_III': 3, 'Stage_IV': 4}

    METABRIC 列: AGE, ER, HER2, LN, GRADE → 需重编码为 TCGA 格式
    """
    encoded = pd.DataFrame(index=clin_df.index)

    # AGE: 直接复制
    encoded['AGE'] = pd.to_numeric(clin_df['AGE'], errors='coerce')

    # ER: ER+ → 0, ER- → 1, NA/其他 → 2
    encoded['ER'] = clin_df['ER'].map({'ER+': 0, 'ER-': 1}).fillna(2).astype(int)

    # HER2: 先统一命名后编码
    her2_mapped = clin_df['HER2'].str.replace('HER2', 'Her2', regex=False)
    encoded['Her2'] = her2_mapped.map({'Her2+': 0, 'Her2-': 1}).fillna(2).astype(int)

    # LN: LN+ → 0, LN- → 1, NA/其他 → 2
    encoded['LN'] = clin_df['LN'].map({'LN+': 0, 'LN-': 1}).fillna(2).astype(int)

    # GRADE → STAGE: NA→0, Stage_I→1, Stage_II→2, Stage_III→3, (Stage_IV→4 无对应)
    encoded['STAGE'] = clin_df['GRADE'].map({
        'Stage_I': 1, 'Stage_II': 2, 'Stage_III': 3
    }).fillna(0).astype(int)

    return encoded


def load_metabric_data(data_dir, return_sample_ids=True):
    """
    加载 METABRIC 数据，编码临床特征以匹配 TCGA 训练格式。

    返回数据字典的键与 TCGA 训练一致: clin, snv, cnv, mrna
    其中 clin 维度为 5 (AGE, ER, Her2, LN, STAGE 编码后)
    """
    clinical_path = os.path.join(data_dir, "METABRIC_Clinical_HRD.csv")
    snv_path = os.path.join(data_dir, "metabric_SNV.csv")
    cna_path = os.path.join(data_dir, "metabric_CNA.csv")
    mrna_path = os.path.join(data_dir, "metabric_mRNA.csv")

    print(f"加载 METABRIC 数据:")
    print(f"  临床数据: {clinical_path}")
    print(f"  SNV数据:  {snv_path}")
    print(f"  CNA数据:  {cna_path}")
    print(f"  mRNA数据: {mrna_path}")

    clin = pd.read_csv(clinical_path)
    snv = pd.read_csv(snv_path)
    cna = pd.read_csv(cna_path)
    mrna = pd.read_csv(mrna_path)

    # --- 设置索引为 Sample_ID ---
    def set_index(df):
        df = df.copy()
        if 'Sample_ID' in df.columns:
            df = df.set_index('Sample_ID')
        else:
            sample_id_cols = ['SAMPLE_ID', 'sample_id', 'sample']
            found = False
            for col in sample_id_cols:
                if col in df.columns:
                    df = df.set_index(col)
                    found = True
                    break
            if not found:
                df = df.set_index(df.columns[0])
        df = df[~df.index.duplicated(keep='first')]
        return df

    clin = set_index(clin)
    snv = set_index(snv)
    cna = set_index(cna)
    mrna = set_index(mrna)

    # --- 提取标签 ---
    if 'HRD_label' in clin.columns:
        y = clin['HRD_label'].astype(int)
    else:
        raise ValueError("临床数据中未找到 HRD_label 列")

    # --- 编码临床特征为 TCGA 兼容格式 ---
    clin_feat_encoded = encode_clinical_to_tcga_mapping(clin)
    # 确保所有列为数值且无缺失
    clin_feat_numeric = clin_feat_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

    # --- 清洗组学数据 ---
    def clean_numeric(df):
        df = df.copy()
        bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        df = df.drop(columns=bad)
        return df.apply(pd.to_numeric, errors='coerce').fillna(0)

    snv = clean_numeric(snv)
    cna = clean_numeric(cna)
    mrna = clean_numeric(mrna)

    # --- 样本对齐 ---
    common_samples = sorted(list(
        set(snv.index) &
        set(cna.index) &
        set(mrna.index) &
        set(clin_feat_numeric.index) &
        set(y.index)
    ))

    print(f"匹配样本数: {len(common_samples)}")

    if len(common_samples) == 0:
        raise ValueError("无法找到共同样本，请检查数据索引是否匹配")

    snv = snv.loc[common_samples]
    cna = cna.loc[common_samples]
    mrna = mrna.loc[common_samples]
    clin_feat_numeric = clin_feat_numeric.loc[common_samples]
    y = y.loc[common_samples]

    # --- 转为 numpy ---
    X_snv = snv.values.astype(np.float32)
    X_cna = cna.values.astype(np.float32)
    X_mrna = mrna.values.astype(np.float32)
    X_clin = clin_feat_numeric.values.astype(np.float32)

    n_classes = len(np.unique(y))

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
        return X_data, y.values, feature_dims, common_samples
    else:
        return X_data, y.values, feature_dims


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


def load_model_from_checkpoint(model_path, metadata, hidden_map=None):
    """
    从 checkpoint 加载模型

    Args:
        model_path: .pth 文件路径
        metadata: 模型元数据字典
        hidden_map: 模态数量 → hidden_size 映射（与训练一致）
    """
    dims = metadata['dims']
    n_classes = metadata['n_classes']
    modalities = metadata['modalities']
    n_mods = len(modalities)

    # 确定 hidden size（与训练脚本逻辑一致）
    if n_mods <= 2:
        hidden = 64 if hidden_map is None else hidden_map.get('default', 64)
    elif n_mods == 3:
        hidden = 96
    else:
        hidden = 128

    if n_mods == 1:
        mod = modalities[0]
        model = SingleModalMLP(dims[mod], hidden, n_classes)
    else:
        model = CAGMFNet(dims, hidden, n_classes)

    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model, hidden


# ======================== 结果保存 ========================
def save_predictions(predictions_dir, group_name, sample_ids,
                     y_true, y_pred, y_probs, class_labels, seed):
    """保存预测结果"""
    os.makedirs(predictions_dir, exist_ok=True)
    safe_name = group_name.replace('+', '_')
    pred_file = os.path.join(predictions_dir, f"seed_{seed}_{safe_name}_predictions.npz")
    csv_file = os.path.join(predictions_dir, f"seed_{seed}_{safe_name}_predictions.csv")

    np.savez(
        pred_file,
        sample_ids=sample_ids,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        class_labels=class_labels,
        metadata=json.dumps({'seed': seed, 'group': group_name})
    )

    df = pd.DataFrame({
        'sample_id': sample_ids,
        'true_label': [class_labels[t] for t in y_true],
        'true_label_code': y_true,
        'pred_label': [class_labels[p] for p in y_pred],
        'pred_label_code': y_pred,
    })
    for i, label in enumerate(class_labels):
        df[f'prob_{label}'] = y_probs[:, i]
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
    parser = argparse.ArgumentParser(description='METABRIC 外部验证 - CAGMF-Net 模型平均组合')
    parser.add_argument('--metabric_dir', type=str, default='./data/metabric',
                        help='METABRIC 数据目录')
    parser.add_argument('--model_dir', type=str,
                        default='./eval_results/smote_argmax_ce_ce',
                        help='TCGA 训练结果目录（含 saved_models, results 等）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认在 model_dir 下创建 metabric_external）')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备，默认自动选择')
    parser.add_argument('--n_seeds', type=int, default=100,
                        help='要加载的种子数（从 42 开始），默认 100')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, 'metabric_external')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    print("=" * 80)
    print("METABRIC 外部验证 - CAGMF-Net 模型平均组合")
    print(f"METABRIC 数据: {args.metabric_dir}")
    print(f"TCGA 模型目录: {args.model_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"加载种子数: {args.n_seeds}")
    print("=" * 80)

    # ========== 1. 加载 METABRIC 数据 ==========
    print("\n1. 加载 METABRIC 数据...")
    X_metabric, y_metabric, feature_dims, sample_ids = load_metabric_data(
        args.metabric_dir, return_sample_ids=True
    )
    n_classes = len(np.unique(y_metabric))
    print(f"总样本数: {len(y_metabric)}, 类别数: {n_classes}")
    print(f"各模态维度: {feature_dims}")

    class_labels = ['0', '1'] if n_classes == 2 else [str(c) for c in range(n_classes)]

    # ========== 2. 准备组合配置 ==========
    print("\n2. 模型组合配置:")
    all_modal_candidates = ALL_CANDIDATE_MODELS
    combination_groups = COMBINATION_GROUPS
    for group_name, indices in combination_groups.items():
        model_names = [get_combination_display_name(all_modal_candidates[i]) for i in indices]
        print(f"  {group_name}: {len(indices)} 个模型 -> {model_names}")

    # ========== 3. 加载权重 ==========
    print("\n3. 加载 TCGA CV 权重...")
    weights_path = os.path.join(args.model_dir, "results", "all_groups_weights.csv")
    weights_df = pd.read_csv(weights_path)
    weights_pivot = {}
    for seed in weights_df['seed'].unique():
        seed_weights = weights_df[weights_df['seed'] == seed]
        weights_pivot[seed] = {}
        for group in seed_weights['group'].unique():
            gw = seed_weights[seed_weights['group'] == group]
            weights_pivot[seed][group] = gw['weight'].values
    print(f"  已加载 {len(weights_pivot)} 个种子的 CV 权重")

    # ========== 4. 创建输出目录 ==========
    print("\n4. 初始化结果存储...")
    results_dir = os.path.join(args.output_dir, "results")
    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # ========== 5. 逐种子进行外部验证 ==========
    print(f"\n5. 开始外部验证，共 {args.n_seeds} 个种子...")
    seeds = range(42, 42 + args.n_seeds)

    all_seeds_results = {}
    saved_predictions_list = []

    for seed in tqdm(seeds, desc="外部验证"):
        seed_model_dir = os.path.join(args.model_dir, "saved_models", f"seed_{seed}")
        scaler_path = os.path.join(seed_model_dir, "scalers.pkl")

        if not os.path.exists(seed_model_dir):
            print(f"  种子 {seed} 模型目录不存在，跳过")
            continue

        if not os.path.exists(scaler_path):
            print(f"  种子 {seed} scaler 文件不存在，跳过")
            continue

        # 加载 scalers（兼容不同numpy版本的pickle）
        class _NumpyCompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith('numpy._core'):
                    new_module = 'numpy.core' + module[len('numpy._core'):]
                    return super().find_class(new_module, name)
                return super().find_class(module, name)

        with open(scaler_path, 'rb') as f:
            scalers = _NumpyCompatUnpickler(f).load()

        # 标准化 METABRIC 数据
        X_std = {}
        for mod in X_metabric.keys():
            X_std[mod] = scalers[mod].transform(X_metabric[mod])

        # 加载 8 个候选模型
        models = []
        for m, modalities in enumerate(all_modal_candidates):
            model_name = get_combination_display_name(modalities)
            model_path = os.path.join(seed_model_dir, f"{model_name}_model.pth")
            metadata_path = os.path.join(seed_model_dir, f"{model_name}_model_metadata.json")

            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                print(f"  种子 {seed} 模型 {model_name} 文件不完整，跳过")
                break

            with open(metadata_path) as f:
                metadata = json.load(f)

            model, _ = load_model_from_checkpoint(model_path, metadata, {})
            model = model.to(device)
            models.append(model)
        else:
            # 所有 8 个模型加载成功
            pass

        if len(models) < 8:
            print(f"  种子 {seed} 仅加载了 {len(models)}/8 个模型，跳过")
            continue

        # 获取所有候选模型的 METABRIC 预测
        candidate_probs = []
        for m, modalities in enumerate(all_modal_candidates):
            X_sub = {mod: X_std[mod] for mod in modalities}
            probs = predict_model(models[m], X_sub, modalities, device)
            candidate_probs.append(probs)
        candidate_probs = np.array(candidate_probs)  # (8, n_samples, n_classes)

        # 对每个组合计算加权集成预测
        seed_results = {}
        for group_name, model_indices in combination_groups.items():
            if len(model_indices) == 0:
                continue

            weights = weights_pivot[seed][group_name]
            group_probs = candidate_probs[model_indices, :, :]

            # 加权平均
            final_probs = np.zeros_like(group_probs[0])
            for w, probs in zip(weights, group_probs):
                final_probs += w * probs

            final_pred = np.argmax(final_probs, axis=1)

            # 评估
            eval_metrics_nested = evaluate_predictions(final_probs, final_pred, y_metabric, n_classes)
            eval_metrics_flat = flatten_metrics(eval_metrics_nested)

            # 保存预测
            pred_file, csv_file = save_predictions(
                predictions_dir, group_name, sample_ids,
                y_metabric, final_pred, final_probs, class_labels, seed
            )
            saved_predictions_list.append({
                'seed': seed,
                'group': group_name,
                'npz_file': pred_file,
                'csv_file': csv_file,
            })

            seed_results[group_name] = {
                'weights': weights.tolist(),
                'model_indices': model_indices,
                'metrics': eval_metrics_flat,
                'metrics_nested': eval_metrics_nested,
                'predictions_file': pred_file,
                'predictions_csv': csv_file,
            }

            print(f"  种子{seed}/{group_name}: "
                  f"Acc={eval_metrics_nested['accuracy']:.4f}, "
                  f"F1={eval_metrics_nested['macro']['f1']:.4f}, "
                  f"AUC={eval_metrics_nested['macro']['roc_auc']:.4f}")

        all_seeds_results[seed] = seed_results

    # ========== 6. 汇总结果 ==========
    print(f"\n{'=' * 80}")
    print("6. 汇总外部验证结果...")
    print(f"{'=' * 80}")

    flat_metrics_names = [
        'accuracy', 'log_loss', 'mse', 'mae',
        'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_macro', 'prauc_macro',
        'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_weighted', 'prauc_weighted',
        'precision_micro', 'recall_micro', 'f1_micro', 'roc_auc_micro', 'prauc_micro'
    ]

    all_groups_summary = {}

    for group_name in combination_groups.keys():
        group_metrics = {metric: [] for metric in flat_metrics_names}

        for seed, seed_result in all_seeds_results.items():
            if group_name in seed_result:
                metrics = seed_result[group_name]['metrics']
                for metric in flat_metrics_names:
                    if metric in metrics:
                        group_metrics[metric].append(metrics[metric])

        summary = {}
        for metric in flat_metrics_names:
            values = group_metrics[metric]
            if values:
                summary[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        all_groups_summary[group_name] = summary

        print(f"\n{group_name}:")
        print(f"  F1 (macro): {summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
        print(f"  AUC (macro): {summary['roc_auc_macro']['mean']:.4f}±{summary['roc_auc_macro']['std']:.4f}")
        print(f"  Accuracy: {summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}")

    # ========== 7. 保存结果 ==========
    print("\n7. 保存结果...")

    # 汇总表
    summary_rows = []
    for group_name, summary in all_groups_summary.items():
        row = {'Group': group_name}
        for metric in flat_metrics_names:
            if metric in summary:
                row[metric] = f"{summary[metric]['mean']:.4f}±{summary[metric]['std']:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, "metabric_external_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  外部验证汇总结果保存至: {summary_path}")

    # 详细结果
    detailed_rows = []
    for seed, seed_result in all_seeds_results.items():
        for group_name, result in seed_result.items():
            row = {'seed': seed, 'group': group_name}
            for metric, value in result['metrics'].items():
                row[metric] = value
            row['predictions_file'] = result.get('predictions_file', '')
            row['predictions_csv'] = result.get('predictions_csv', '')
            detailed_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = os.path.join(results_dir, "metabric_external_detailed.csv")
    detailed_df.to_csv(detailed_path, index=False)
    print(f"  详细结果保存至: {detailed_path}")

    # 预测文件索引
    predictions_index_df = pd.DataFrame(saved_predictions_list)
    predictions_index_path = os.path.join(results_dir, "metabric_predictions_index.csv")
    predictions_index_df.to_csv(predictions_index_path, index=False)
    print(f"  预测文件索引保存至: {predictions_index_path}")

    # 完整结果 JSON
    results_dict = {
        'groups': {name: [get_combination_display_name(all_modal_candidates[i]) for i in indices]
                   for name, indices in combination_groups.items()},
        'summary': {group: {
            'metrics': {metric: {'mean': summary[metric]['mean'], 'std': summary[metric]['std']}
                        for metric in flat_metrics_names if metric in summary}
        } for group, summary in all_groups_summary.items()},
        'n_seeds': len(all_seeds_results),
        'n_metabric_samples': len(y_metabric),
        'feature_dims': feature_dims,
        'source_experiment': args.model_dir,
    }

    results_path = os.path.join(results_dir, "metabric_external_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  完整结果保存至: {results_path}")

    # 简洁摘要表格文本
    print(f"\n{'=' * 80}")
    print("METABRIC 外部验证汇总")
    print(f"{'=' * 80}")
    header = f"{'Group':<28} {'Accuracy':<16} {'F1_macro':<16} {'AUC_macro':<16}"
    print(header)
    print("-" * 80)
    for group_name in combination_groups.keys():
        if group_name in all_groups_summary:
            s = all_groups_summary[group_name]
            acc = f"{s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f}"
            f1 = f"{s['f1_macro']['mean']:.4f}±{s['f1_macro']['std']:.4f}"
            auc = f"{s['roc_auc_macro']['mean']:.4f}±{s['roc_auc_macro']['std']:.4f}"
            print(f"{group_name:<28} {acc:<16} {f1:<16} {auc:<16}")

    print(f"\n结果保存至: {args.output_dir}")
    print(f"完成外部验证！")


if __name__ == "__main__":
    main()
