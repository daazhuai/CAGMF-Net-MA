"""
METABRIC 外部验证脚本
加载 TCGA 上已训练好的 CAGMF-Net 模型，在 METABRIC 数据集上进行外部验证。

使用方式：
  1. 先运行 cagmf_net_ma_external_train_TCGA.py 完成 TCGA 训练
  2. 再运行本脚本：
     python cagmf_net_ma_external_validation_METABRIC.py \
         --tcga_model_dir ./eval_results/tcga_hrd_trained \
         --metabric_dir ./data/metabric \
         --output_dir ./eval_results/metabric_external
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
from sklearn.preprocessing import StandardScaler
import warnings
import argparse

warnings.filterwarnings('ignore')

sys.path.append('.')
from utils import evaluate_predictions


# ======================== 模型定义（需与训练脚本一致以正确加载权重） ========================
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


# ======================== 常量（与训练脚本一致） ========================
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

    TCGA LabelEncoder 编码（仅使用 AGE, ER, Her2, LN 四个特征）:
      ER:     {'ER+': 0, 'ER-': 1, 'NA': 2}
      Her2:   {'Her2+': 0, 'Her2-': 1, 'NA': 2}  注意: TCGA 使用 'Her2' (小写 r)
      LN:     {'LN+': 0, 'LN-': 1, 'NA': 2}

    METABRIC 列: AGE, ER, HER2, LN → 需重编码为 TCGA 格式
    """
    encoded = pd.DataFrame(index=clin_df.index)
    encoded['AGE'] = pd.to_numeric(clin_df['AGE'], errors='coerce')
    encoded['ER'] = clin_df['ER'].map({'ER+': 0, 'ER-': 1}).fillna(2).astype(int)
    her2_mapped = clin_df['HER2'].str.replace('HER2', 'Her2', regex=False)
    encoded['Her2'] = her2_mapped.map({'Her2+': 0, 'Her2-': 1}).fillna(2).astype(int)
    encoded['LN'] = clin_df['LN'].map({'LN+': 0, 'LN-': 1}).fillna(2).astype(int)
    return encoded


def load_metabric_data(data_dir, return_sample_ids=True):
    """
    加载 METABRIC 数据，编码临床特征以匹配 TCGA 训练格式。
    返回数据字典的键与 TCGA 训练一致: clin, snv, cnv, mrna
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

    if 'HRD_label' in clin.columns:
        y = clin['HRD_label'].astype(int)
    else:
        raise ValueError("临床数据中未找到 HRD_label 列")

    clin_feat_encoded = encode_clinical_to_tcga_mapping(clin)
    clin_feat_numeric = clin_feat_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

    def clean_numeric(df):
        df = df.copy()
        bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        df = df.drop(columns=bad)
        return df.apply(pd.to_numeric, errors='coerce').fillna(0)

    snv = clean_numeric(snv)
    cna = clean_numeric(cna)
    mrna = clean_numeric(mrna)

    common_samples = sorted(list(
        set(snv.index) & set(cna.index) & set(mrna.index) &
        set(clin_feat_numeric.index) & set(y.index)
    ))
    print(f"  匹配样本数: {len(common_samples)}")
    if len(common_samples) == 0:
        raise ValueError("无法找到共同样本，请检查数据索引是否匹配")

    snv = snv.loc[common_samples]
    cna = cna.loc[common_samples]
    mrna = mrna.loc[common_samples]
    clin_feat_numeric = clin_feat_numeric.loc[common_samples]
    y = y.loc[common_samples]

    X_data = {
        "clin": clin_feat_numeric.values.astype(np.float32),
        "snv": snv.values.astype(np.float32),
        "cnv": cna.values.astype(np.float32),
        "mrna": mrna.values.astype(np.float32),
    }
    feature_dims = {mod: X_data[mod].shape[1] for mod in X_data}

    print(f"  HRD标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  维度: SNV={X_data['snv'].shape}, CNA={X_data['cnv'].shape}, "
          f"mRNA={X_data['mrna'].shape}, Clinical={X_data['clin'].shape}")

    if return_sample_ids:
        return X_data, y.values, feature_dims, common_samples
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


def predict_with_threshold(probs, threshold):
    """使用阈值将概率转换为二分类预测"""
    preds = np.zeros(len(probs), dtype=int)
    preds[probs[:, 1] >= threshold] = 1
    return preds


# ======================== 结果保存 ========================
def save_predictions(predictions_dir, group_name, sample_ids,
                     y_true, y_pred, y_probs, class_labels):
    """保存预测结果"""
    os.makedirs(predictions_dir, exist_ok=True)
    safe_name = group_name.replace('+', '_')
    pred_file = os.path.join(predictions_dir, f"{safe_name}_predictions.npz")
    csv_file = os.path.join(predictions_dir, f"{safe_name}_predictions.csv")

    np.savez(
        pred_file,
        sample_ids=sample_ids,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        class_labels=class_labels,
        metadata=json.dumps({'group': group_name})
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


# ======================== TCGA 模型加载 ========================
def build_model_from_metadata(metadata, device=None):
    """根据元数据重建模型架构并加载权重"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modalities = metadata['modalities']
    dims = metadata['dims']
    n_classes = metadata['n_classes']
    hidden = metadata.get('hidden', 32)
    model_type = metadata.get('model_type', 'CAGMFNet')

    if model_type == 'SingleModalMLP':
        mod = modalities[0]
        model = SingleModalMLP(dims[mod], hidden, n_classes)
    else:
        model = CAGMFNet(dims, hidden, n_classes)

    return model.to(device)


def load_tcga_models(models_dir, candidate_models, device=None):
    """加载所有 TCGA 训练好的模型"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for modalities in candidate_models:
        model_name = get_combination_display_name(modalities)
        safe_name = model_name.replace('+', '_')
        metadata_path = os.path.join(models_dir, f"{safe_name}_model_metadata.json")
        model_path = os.path.join(models_dir, f"{safe_name}_model.pth")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        model = build_model_from_metadata(metadata, device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    return models


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(description='METABRIC 外部验证 - 加载已训练的TCGA模型')
    parser.add_argument('--tcga_model_dir', type=str, default='./eval_results/tcga_hrd_trained',
                        help='TCGA 训练产物目录（cagmf_net_ma_external_train_TCGA.py 的输出）')
    parser.add_argument('--metabric_dir', type=str, default='./data/metabric',
                        help='METABRIC 数据目录')
    parser.add_argument('--output_dir', type=str, default='./eval_results/metabric_external',
                        help='输出目录')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备，默认自动选择')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    models_dir = os.path.join(args.tcga_model_dir, "saved_models")

    print("=" * 80)
    print("METABRIC 外部验证 - 加载 TCGA 预训练模型")
    print(f"TCGA 模型目录: {args.tcga_model_dir}")
    print(f"METABRIC 数据: {args.metabric_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)

    # ========== 1. 加载 TCGA 训练产物 ==========
    print("\n1. 加载 TCGA 训练产物...")

    metadata_path = os.path.join(models_dir, "tcga_training_metadata.pkl")
    with open(metadata_path, 'rb') as f:
        tcga_meta = pickle.load(f)
    n_classes = tcga_meta['n_classes']
    print(f"  TCGA 样本数: {len(tcga_meta['y_tcga'])}, 类别数: {n_classes}")
    print(f"  训练随机种子: {tcga_meta['random_seed']}")
    if tcga_meta.get('data_augmentation'):
        print(f"  数据增强: {tcga_meta['data_augmentation']}")

    scaler_path = os.path.join(models_dir, "scalers.pkl")
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    print(f"  已加载 scalers: {list(scalers.keys())}")

    cv_weights_path = os.path.join(models_dir, "cv_weights.pkl")
    with open(cv_weights_path, 'rb') as f:
        cv_weights = pickle.load(f)
    print(f"  已加载 CV 权重: {list(cv_weights.keys())}")

    thresholds_path = os.path.join(models_dir, "thresholds.pkl")
    with open(thresholds_path, 'rb') as f:
        thresholds_data = pickle.load(f)
    thresholds = thresholds_data['thresholds']
    threshold_method = thresholds_data.get('threshold_method', 'argmax')
    print(f"  阈值方法: {threshold_method}")

    # ========== 2. 加载 TCGA 模型 ==========
    print("\n2. 加载 TCGA 模型...")
    final_models = load_tcga_models(models_dir, ALL_CANDIDATE_MODELS, device)
    for modalities in ALL_CANDIDATE_MODELS:
        print(f"  {get_combination_display_name(modalities)} 已加载")

    # ========== 3. 加载 METABRIC 数据 ==========
    print("\n3. 加载 METABRIC 数据...")
    X_metabric, y_metabric, metabric_feature_dims, metabric_sample_ids = load_metabric_data(
        args.metabric_dir, return_sample_ids=True
    )
    print(f"  METABRIC 总样本数: {len(y_metabric)}")
    class_labels = ['0', '1'] if n_classes == 2 else [str(c) for c in range(n_classes)]

    # 标准化 METABRIC（使用 TCGA scalers）
    X_metabric_std = {}
    for mod in X_metabric.keys():
        if mod in scalers:
            X_metabric_std[mod] = scalers[mod].transform(X_metabric[mod])
        else:
            X_metabric_std[mod] = X_metabric[mod]
            print(f"  警告: 模态 {mod} 无对应 scaler，使用原始值")

    # ========== 4. METABRIC 预测 ==========
    print("\n4. METABRIC 预测...")
    metabric_candidate_probs = []
    for m, modalities in enumerate(ALL_CANDIDATE_MODELS):
        X_sub = {mod: X_metabric_std[mod] for mod in modalities}
        probs = predict_model(final_models[m], X_sub, modalities, device)
        metabric_candidate_probs.append(probs)
    metabric_candidate_probs = np.array(metabric_candidate_probs)

    # ========== 5. 评估 ==========
    print("\n5. 评估...")
    results_dir = os.path.join(args.output_dir, "results")
    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    all_groups_results = {}

    for group_name, model_indices in COMBINATION_GROUPS.items():
        if group_name not in cv_weights:
            print(f"  跳过 {group_name}: CV权重中不存在")
            continue

        weights = cv_weights[group_name]
        group_probs = metabric_candidate_probs[model_indices, :, :]

        # CV 加权集成
        final_probs = np.zeros_like(group_probs[0])
        for w, probs in zip(weights, group_probs):
            final_probs += w * probs

        # 应用阈值
        th = thresholds.get(group_name, 0.5)
        if n_classes == 2 and threshold_method != 'argmax':
            final_pred = predict_with_threshold(final_probs, th)
        else:
            final_pred = np.argmax(final_probs, axis=1)

        # 评估
        eval_metrics_nested = evaluate_predictions(final_probs, final_pred, y_metabric, n_classes)
        eval_metrics_flat = flatten_metrics(eval_metrics_nested)
        if 'per_class' in eval_metrics_nested:
            for i in range(len(eval_metrics_nested['per_class']['recall'])):
                eval_metrics_flat[f'sensitivity_class_{i}'] = eval_metrics_nested['per_class']['recall'][i]
                eval_metrics_flat[f'precision_class_{i}'] = eval_metrics_nested['per_class']['precision'][i]
                eval_metrics_flat[f'f1_class_{i}'] = eval_metrics_nested['per_class']['f1'][i]

        # 保存预测
        pred_file, csv_file = save_predictions(
            predictions_dir, group_name, metabric_sample_ids,
            y_metabric, final_pred, final_probs, class_labels
        )

        all_groups_results[group_name] = {
            'weights': weights.tolist(),
            'model_indices': model_indices,
            'threshold': float(th),
            'metrics': eval_metrics_flat,
            'metrics_nested': eval_metrics_nested,
            'predictions_file': pred_file,
            'predictions_csv': csv_file,
        }

        print(f"  {group_name}: "
              f"Acc={eval_metrics_nested['accuracy']:.4f}, "
              f"F1={eval_metrics_nested['macro']['f1']:.4f}, "
              f"AUC={eval_metrics_nested['macro']['roc_auc']:.4f}", end='')
        if f'sensitivity_class_1' in eval_metrics_flat:
            print(f", HRD+ Sens={eval_metrics_flat['sensitivity_class_1']:.4f}")
        else:
            print()

    # ========== 6. 保存结果 ==========
    print("\n6. 保存结果...")

    flat_metrics_names = [
        'accuracy', 'log_loss', 'mse', 'mae',
        'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_macro', 'prauc_macro',
        'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_weighted', 'prauc_weighted',
        'precision_micro', 'recall_micro', 'f1_micro', 'roc_auc_micro', 'prauc_micro',
        'sensitivity_class_0', 'sensitivity_class_1',
        'precision_class_0', 'precision_class_1',
        'f1_class_0', 'f1_class_1'
    ]

    summary_rows = []
    for group_name, result in all_groups_results.items():
        row = {'Group': group_name, 'n_models': len(result['model_indices']),
               'threshold': result['threshold']}
        for metric in flat_metrics_names:
            if metric in result['metrics']:
                row[metric] = result['metrics'][metric]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, "metabric_external_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  结果保存至: {summary_path}")

    # 保存配置
    config = {
        'tcga_model_dir': args.tcga_model_dir,
        'metabric_dir': args.metabric_dir,
        'threshold_method': threshold_method,
        'n_classes': n_classes,
        'n_metabric_samples': len(y_metabric),
    }
    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # 简洁摘要
    print(f"\n{'=' * 80}")
    print("METABRIC 外部验证结果")
    print(f"阈值方法: {threshold_method}")
    print(f"{'=' * 80}")
    header = f"{'Group':<26} {'Acc':<8} {'AUC':<8} {'HRD+Sens':<10} {'HRD+Prec':<10} {'Thresh':<8}"
    print(header)
    print("-" * 80)
    for group_name in COMBINATION_GROUPS.keys():
        if group_name in all_groups_results:
            r = all_groups_results[group_name]
            m = r['metrics']
            acc = f"{m['accuracy']:.4f}"
            auc = f"{m['roc_auc_macro']:.4f}"
            sens1 = f"{m.get('sensitivity_class_1', 0):.4f}"
            prec1 = f"{m.get('precision_class_1', 0):.4f}"
            th = f"{r['threshold']:.4f}"
            print(f"{group_name:<26} {acc:<8} {auc:<8} {sens1:<10} {prec1:<10} {th:<8}")

    print(f"\n结果保存至: {args.output_dir}")
    print("完成外部验证！")


if __name__ == "__main__":
    main()
