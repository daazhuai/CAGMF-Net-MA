"""
MyBrCa 外部验证脚本
加载 TCGA 上已训练好的 CAGMF-Net 模型，在 MyBrCa 数据集上进行外部验证。

仅验证 4 种模态组合（MyBrCa 无 mRNA）：
  Clinical, Clinical+SNV, Clinical+CNA, Clinical+SNV+CNA

使用方式：
  1. 先运行 cagmf_net_ma_external_train_TCGA.py 完成 TCGA 训练
  2. 再运行本脚本：
     python cagmf_net_ma_external_validation_MyBrCa.py \
         --tcga_model_dir ./eval_results/tcga_hrd_trained \
         --mybrca_dir ./data/mybrca \
         --output_dir ./eval_results/mybrca_external
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
    def __init__(self, in_dim, hidden, n_class, dropout=0.2):
        super().__init__()
        self.mlp = MLP(in_dim, hidden, dropout)
        self.classifier = nn.Linear(hidden, n_class)

    def forward(self, x):
        return self.classifier(self.mlp(x))


# ======================== 常量 ========================
MODALITY_NAME_MAP = {'clin': 'Clinical', 'snv': 'SNV', 'cnv': 'CNA'}

# 仅包含无 mRNA 的 4 个候选模型
CANDIDATE_MODELS = [
    ['clin'],               # 0
    ['clin', 'snv'],        # 1
    ['clin', 'cnv'],        # 2
    ['clin', 'snv', 'cnv'], # 3
]

# 4 种模态组合
COMBINATION_GROUPS = {
    'Clinical':         [0],
    'Clinical+SNV':     [1],
    'Clinical+CNA':     [2],
    'Clinical+SNV+CNA': [1, 2, 3],
}


def get_model_display_name(modalities):
    return '+'.join([MODALITY_NAME_MAP[mod] for mod in modalities])


# ======================== MyBrCa 数据加载 ========================
def load_mybrca_data(data_dir, return_sample_ids=True):
    """
    加载 MyBrCa 数据（无 mRNA），编码临床特征以匹配 TCGA 格式。

    TCGA LabelEncoder 编码:
      ER:     ER+→0, ER-→1
      Her2:   Her2+→0, Her2-→1
      LN:     LN+→0, LN-→1

    MyBrCa LN: LN+/LN-，映射为 TCGA 规则 (LN+→0, LN-→1)
    """
    clin_path = os.path.join(data_dir, "mybrca_Clinical_HRD.csv")
    snv_path = os.path.join(data_dir, "mybrca_SNV.csv")
    cna_path = os.path.join(data_dir, "mybrca_CNV.csv")

    print(f"加载 MyBrCa 数据:")
    print(f"  Clinical: {clin_path}")
    print(f"  SNV: {snv_path}")
    print(f"  CNA: {cna_path}")

    clin = pd.read_csv(clin_path)
    snv = pd.read_csv(snv_path)
    cna = pd.read_csv(cna_path)

    def set_index(df):
        df = df.copy()
        if 'Sample_ID' in df.columns:
            df = df.set_index('Sample_ID')
        else:
            for col in ['SAMPLE_ID', 'sample_id', 'sample']:
                if col in df.columns:
                    df = df.set_index(col)
                    break
            else:
                df = df.set_index(df.columns[0])
        df = df[~df.index.duplicated(keep='first')]
        return df

    def clean_numeric(df):
        df = df.copy()
        bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        df = df.drop(columns=bad)
        return df.apply(pd.to_numeric, errors='coerce').fillna(0)

    clin = set_index(clin)
    snv = set_index(snv)
    cna = set_index(cna)

    if 'HRD_label' in clin.columns:
        y = clin['HRD_label'].astype(int)
    else:
        raise ValueError("临床数据中未找到 HRD_label 列")

    # 编码临床特征（匹配 TCGA 格式）
    clin_encoded = pd.DataFrame(index=clin.index)
    clin_encoded['AGE'] = pd.to_numeric(clin['AGE'], errors='coerce')
    clin_encoded['ER'] = clin['ER'].map({'ER+': 0, 'ER-': 1}).fillna(2).astype(int)
    clin_encoded['Her2'] = clin['HER2'].map({'HER2+': 0, 'HER2-': 1}).fillna(2).astype(int)
    # MyBrCa LN: LN+/LN-; TCGA: LN+→0, LN-→1
    clin_encoded['LN'] = clin['LN'].map({'LN+': 0, 'LN-': 1}).fillna(2).astype(int)
    clin_feat = clin_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

    snv = clean_numeric(snv)
    cna = clean_numeric(cna)

    common = sorted(list(
        set(snv.index) & set(cna.index) & set(clin_feat.index) & set(y.index)
    ))
    print(f"  匹配样本数: {len(common)}")
    if len(common) == 0:
        raise ValueError("无法找到共同样本")

    snv = snv.loc[common]
    cna = cna.loc[common]
    clin_feat = clin_feat.loc[common]
    y = y.loc[common]

    X_data = {
        "clin": clin_feat.values.astype(np.float32),
        "snv": snv.values.astype(np.float32),
        "cnv": cna.values.astype(np.float32),
    }
    n_classes = len(np.unique(y))
    feature_dims = {mod: X_data[mod].shape[1] for mod in X_data}

    print(f"  HRD: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  维度: SNV={X_data['snv'].shape}, CNA={X_data['cnv'].shape}, Clinical={X_data['clin'].shape}")

    if return_sample_ids:
        return X_data, y.values, feature_dims, common
    return X_data, y.values, feature_dims


# ======================== 模型预测 ========================
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
                outputs = model(batch_data[0].to(device))
            else:
                batch_X_dict = {mod: batch_data[i].to(device) for i, mod in enumerate(modalities)}
                outputs = model(batch_X_dict)
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
    return np.array(all_probs)


def predict_with_threshold(probs, threshold):
    preds = np.zeros(len(probs), dtype=int)
    preds[probs[:, 1] >= threshold] = 1
    return preds


# ======================== 结果保存 ========================
def save_predictions(predictions_dir, group_name, sample_ids, y_true, y_pred, y_probs, class_labels):
    os.makedirs(predictions_dir, exist_ok=True)
    safe_name = group_name.replace('+', '_')
    pred_file = os.path.join(predictions_dir, f"{safe_name}_predictions.npz")
    csv_file = os.path.join(predictions_dir, f"{safe_name}_predictions.csv")
    np.savez(pred_file, sample_ids=sample_ids, y_true=y_true, y_pred=y_pred,
             y_probs=y_probs, class_labels=class_labels,
             metadata=json.dumps({'group': group_name}))
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
    """加载 TCGA 训练好的模型（仅加载所需的候选模型）"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for modalities in candidate_models:
        model_name = get_model_display_name(modalities)
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
    parser = argparse.ArgumentParser(description='MyBrCa 外部验证 - 加载已训练的TCGA模型')
    parser.add_argument('--tcga_model_dir', type=str, default='./eval_results/tcga_hrd_trained',
                        help='TCGA 训练产物目录（cagmf_net_ma_external_train_TCGA.py 的输出）')
    parser.add_argument('--mybrca_dir', type=str, default='./data/mybrca')
    parser.add_argument('--output_dir', type=str, default='./eval_results/mybrca_external')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    models_dir = os.path.join(args.tcga_model_dir, "saved_models")

    print("=" * 80)
    print("MyBrCa 外部验证 - 加载 TCGA 预训练模型")
    print(f"  TCGA 模型目录: {args.tcga_model_dir}")
    print(f"  MyBrCa 数据: {args.mybrca_dir}")
    print(f"  模态组合: Clinical, Clinical+SNV, Clinical+CNA, Clinical+SNV+CNA")
    print(f"  设备: {device}")
    print("=" * 80)

    # ========== 1. 加载 TCGA 训练产物 ==========
    print("\n1. 加载 TCGA 训练产物...")

    metadata_path = os.path.join(models_dir, "tcga_training_metadata.pkl")
    with open(metadata_path, 'rb') as f:
        tcga_meta = pickle.load(f)
    n_classes = tcga_meta['n_classes']
    print(f"  TCGA 样本数: {len(tcga_meta['y_tcga'])}, 类别数: {n_classes}")

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

    # ========== 2. 加载 TCGA 模型（仅加载 MyBrCa 需要的 4 个） ==========
    print("\n2. 加载 TCGA 模型（仅无 mRNA 的 4 个候选模型）...")
    final_models = load_tcga_models(models_dir, CANDIDATE_MODELS, device)
    for modalities in CANDIDATE_MODELS:
        print(f"  {get_model_display_name(modalities)} 已加载")

    # ========== 3. 加载 MyBrCa ==========
    print("\n3. 加载 MyBrCa 数据...")
    X_mybrca, y_mybrca, mybrca_dims, mybrca_ids = load_mybrca_data(args.mybrca_dir, return_sample_ids=True)
    class_labels = ['0', '1'] if n_classes == 2 else [str(c) for c in range(n_classes)]

    # 标准化 (用TCGA scaler，MyBrCa无 mRNA 所以只用 clin/snv/cnv 的 scaler)
    X_mybrca_std = {}
    for mod in X_mybrca.keys():
        if mod in scalers:
            X_mybrca_std[mod] = scalers[mod].transform(X_mybrca[mod])
        else:
            X_mybrca_std[mod] = X_mybrca[mod]

    # ========== 4. MyBrCa 预测 ==========
    print("\n4. MyBrCa 预测...")
    mybrca_probs = []
    for m, modalities in enumerate(CANDIDATE_MODELS):
        probs = predict_model(final_models[m], {mod: X_mybrca_std[mod] for mod in modalities}, modalities, device)
        mybrca_probs.append(probs)
    mybrca_probs = np.array(mybrca_probs)

    # ========== 5. 评估 ==========
    print("\n5. 评估 MyBrCa...")
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
        group_probs = mybrca_probs[model_indices, :, :]
        final_probs = np.zeros_like(group_probs[0])
        for w, probs in zip(weights, group_probs):
            final_probs += w * probs

        th = thresholds.get(group_name, 0.5)
        if n_classes == 2 and threshold_method != 'argmax':
            final_pred = predict_with_threshold(final_probs, th)
        else:
            final_pred = np.argmax(final_probs, axis=1)

        eval_metrics_nested = evaluate_predictions(final_probs, final_pred, y_mybrca, n_classes)
        eval_metrics_flat = flatten_metrics(eval_metrics_nested)
        if 'per_class' in eval_metrics_nested:
            for i in range(len(eval_metrics_nested['per_class']['recall'])):
                eval_metrics_flat[f'sensitivity_class_{i}'] = eval_metrics_nested['per_class']['recall'][i]
                eval_metrics_flat[f'precision_class_{i}'] = eval_metrics_nested['per_class']['precision'][i]
                eval_metrics_flat[f'f1_class_{i}'] = eval_metrics_nested['per_class']['f1'][i]

        pred_file, csv_file = save_predictions(
            predictions_dir, group_name, mybrca_ids, y_mybrca, final_pred, final_probs, class_labels)

        all_groups_results[group_name] = {
            'weights': weights.tolist(), 'model_indices': model_indices,
            'threshold': float(th),
            'metrics': eval_metrics_flat,
            'metrics_nested': {k: v for k, v in eval_metrics_nested.items() if k != 'per_class'},
            'metrics_nested_per_class': eval_metrics_nested.get('per_class', {}),
            'predictions_file': pred_file, 'predictions_csv': csv_file,
        }

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
        row = {'Group': group_name, 'n_models': len(result['model_indices']), 'threshold': result['threshold']}
        for metric in flat_metrics_names:
            if metric in result['metrics']:
                row[metric] = result['metrics'][metric]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(results_dir, "mybrca_external_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  Summary CSV: {csv_path}")

    # 保存 JSON（完整指标 + 嵌套结构）
    json_path = os.path.join(results_dir, "mybrca_external_summary.json")
    json_results = {}
    for group_name, result in all_groups_results.items():
        json_results[group_name] = {
            'weights': result['weights'],
            'model_indices': result['model_indices'],
            'threshold': result['threshold'],
            'metrics': result['metrics'],
            'metrics_nested': result['metrics_nested'],
            'metrics_nested_per_class': result['metrics_nested_per_class'],
        }
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=float)
    print(f"  Summary JSON: {json_path}")

    # 配置
    config = {
        'tcga_model_dir': args.tcga_model_dir, 'mybrca_dir': args.mybrca_dir,
        'threshold_method': threshold_method,
        'n_classes': n_classes, 'n_mybrca_samples': int(len(y_mybrca)),
        'modality_combinations': list(COMBINATION_GROUPS.keys()),
    }
    with open(os.path.join(results_dir, "experiment_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # 完整指标表
    print(f"\n{'=' * 100}")
    print("MyBrCa 外部验证结果 - 完整指标")
    print(f"  阈值方法: {threshold_method}")
    print(f"{'=' * 100}")
    for group_name in COMBINATION_GROUPS.keys():
        if group_name not in all_groups_results:
            continue
        r = all_groups_results[group_name]
        m = r['metrics']
        n = r['metrics_nested']
        print(f"\n{'─' * 80}")
        print(f"  {group_name}  (阈值={r['threshold']:.4f})")
        print(f"  {'Accuracy:':<16} {m['accuracy']:.4f}    {'LogLoss:':<16} {m['log_loss']:.4f}    {'MSE:':<16} {m['mse']:.4f}    {'MAE:':<16} {m['mae']:.4f}")
        print(f"  {'':>16} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12} {'PR-AUC':<12}")
        for avg in ['macro', 'weighted', 'micro']:
            vals = n[avg]
            print(f"  {avg.capitalize()+':':>16} {vals['precision']:<12.4f} {vals['recall']:<12.4f} {vals['f1']:<12.4f} {vals['roc_auc']:<12.4f} {vals['prauc']:<12.4f}")
        print(f"  {'Per-Class:':>16}")
        print(f"  {'  Class 0 (HRD-):':>16} Sens={m.get('sensitivity_class_0',0):.4f}  Prec={m.get('precision_class_0',0):.4f}  F1={m.get('f1_class_0',0):.4f}")
        print(f"  {'  Class 1 (HRD+):':>16} Sens={m.get('sensitivity_class_1',0):.4f}  Prec={m.get('precision_class_1',0):.4f}  F1={m.get('f1_class_1',0):.4f}")

    print(f"\n结果保存至: {args.output_dir}")
    print("完成!")


if __name__ == "__main__":
    main()
