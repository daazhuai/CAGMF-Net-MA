"""
机器学习基线模型评估 - TCGA HRD数据专用版
使用TCGA_Clinical_HRD.csv，以patient为ID，HRD_label为标签（二分类）
评估6种ML模型（逻辑回归、随机森林、XGBoost、LightGBM、SVM、Lasso）在8种模态组合下的表现
支持SMOTE/Oversample数据增强和Youden阈值选取
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
import warnings
from tqdm import tqdm
import argparse

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

sys.path.append('.')
from utils import evaluate_predictions, set_seed

# 导入XGBoost和LightGBM
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    XGBClassifier = None
    print("警告: XGBoost未安装，将跳过XGBoost模型")

try:
    from lightgbm import LGBMClassifier
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    LGBMClassifier = None
    print("警告: LightGBM未安装，将跳过LightGBM模型")


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

MODEL_CONFIGS = {
    'LogisticRegression': {
        'class': LogisticRegression,
        'params': {'max_iter': 1000, 'multi_class': 'multinomial'},
    },
    'RandomForest': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': 100},
    },
    'XGBoost': {
        'class': XGBClassifier,
        'params': {'n_estimators': 100, 'learning_rate': 0.3, 'verbosity': 0},
        'requires': 'XGB_AVAILABLE',
    },
    'LightGBM': {
        'class': LGBMClassifier,
        'params': {'n_estimators': 100, 'learning_rate': 0.1, 'verbose': -1},
        'requires': 'LGB_AVAILABLE',
    },
    'SVM': {
        'class': SVC,
        'params': {'probability': True},
    },
    'Lasso': {
        'class': LogisticRegression,
        'params': {'penalty': 'l1', 'solver': 'saga', 'max_iter': 1000,
                   'multi_class': 'multinomial'},
    },
}


# ======================== 数据加载（复用HRD数据加载逻辑） ========================
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
def apply_data_augmentation(X_train, y_train, method, random_state=42):
    """
    对训练数据进行数据增强（SMOTE或随机过采样）

    Args:
        X_train: 拼接后的训练特征 (n_samples, n_features)
        y_train: 训练标签
        method: 'smote' 或 'oversample'
        random_state: 随机种子

    Returns:
        增强后的 X_train, y_train
    """
    if method is None:
        return X_train, y_train

    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            n_minority = np.sum(y_train == np.bincount(y_train).argmin())
            k_neighbors = min(5, max(1, n_minority - 1))
            smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            print(f"      SMOTE增强: {len(y_train)} -> {len(y_res)} 样本")
            return X_res, y_res
        except ImportError:
            print("      SMOTE不可用(需pip install imblearn)，使用随机过采样替代")
            method = 'oversample'

    if method == 'oversample':
        try:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=random_state)
            X_res, y_res = ros.fit_resample(X_train, y_train)
            print(f"      RandomOverSampler增强: {len(y_train)} -> {len(y_res)} 样本")
            return X_res, y_res
        except ImportError:
            classes, counts = np.unique(y_train, return_counts=True)
            max_count = counts.max()
            indices = []
            for cls in classes:
                cls_idx = np.where(y_train == cls)[0]
                if len(cls_idx) < max_count:
                    additional = np.random.RandomState(random_state).choice(
                        cls_idx, max_count - len(cls_idx), replace=True
                    )
                    indices.extend(cls_idx.tolist())
                    indices.extend(additional.tolist())
                else:
                    indices.extend(cls_idx.tolist())
            X_res = X_train[np.array(indices)]
            y_res = y_train[np.array(indices)]
            print(f"      手动过采样增强: {len(y_train)} -> {len(y_res)} 样本")
            return X_res, y_res

    return X_train, y_train


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


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(description='TCGA HRD数据 机器学习基线评估')
    parser.add_argument('--data_dir', type=str, default='./data/tcga',
                        help='TCGA数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./eval_results/tcga_hrd_ml_baselines',
                        help='输出结果目录')
    parser.add_argument('--n_splits', type=int, default=100,
                        help='随机划分次数，默认100')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例，默认0.2')
    parser.add_argument('--random_seed_base', type=int, default=42,
                        help='随机种子基数')
    parser.add_argument('--smote', action='store_true',
                        help='使用SMOTE数据增强')
    parser.add_argument('--oversample', action='store_true',
                        help='使用随机过采样数据增强')
    parser.add_argument('--youden', action='store_true',
                        help='使用Youden阈值选取方法（默认使用argmax）')
    parser.add_argument('--skip_split_generation', action='store_true',
                        help='跳过划分生成，使用已有划分')

    args = parser.parse_args()

    data_augmentation = None
    if args.smote:
        data_augmentation = 'smote'
    elif args.oversample:
        data_augmentation = 'oversample'

    print("=" * 80)
    print("TCGA HRD数据 机器学习基线模型评估")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"划分次数: {args.n_splits}, 测试比例: {args.test_size}")
    if data_augmentation:
        print(f"数据增强: {data_augmentation}")
    if args.youden:
        print(f"阈值方法: Youden指数")
    else:
        print(f"阈值方法: argmax (默认)")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # 确定可用的ML模型
    available_models = {}
    for model_name, config in MODEL_CONFIGS.items():
        if 'requires' in config:
            if config['requires'] == 'XGB_AVAILABLE' and not XGB_AVAILABLE:
                print(f"跳过 {model_name}（XGBoost未安装）")
                continue
            if config['requires'] == 'LGB_AVAILABLE' and not LGB_AVAILABLE:
                print(f"跳过 {model_name}（LightGBM未安装）")
                continue
        available_models[model_name] = config

    print(f"\n可用模型: {list(available_models.keys())}")
    print(f"模态组合数: {len(MODALITY_COMBINATIONS)}")
    print(f"总实验数: {len(MODALITY_COMBINATIONS)} × {len(available_models)} × {args.n_splits} = "
          f"{len(MODALITY_COMBINATIONS) * len(available_models) * args.n_splits}")

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
    split_dir_name = "split_indices_tcga_hrd_ml"
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
    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # 4. 遍历每个划分
    print(f"\n4. 开始实验，共 {args.n_splits} 个划分...")
    seeds = range(args.random_seed_base, args.random_seed_base + args.n_splits)

    # 存储结构: all_results[seed][combo_name][model_name] = metrics_flat
    all_results = {}
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
        sample_ids_test = [sample_ids[idx] for idx in test_idx] if sample_ids is not None else np.arange(len(y_test))

        seed_results = {}

        # 对每个模态组合
        for combo_name, modalities in MODALITY_COMBINATIONS.items():
            # 拼接特征
            X_tr = np.concatenate([X_train_raw[mod] for mod in modalities], axis=1)
            X_te = np.concatenate([X_test_raw[mod] for mod in modalities], axis=1)

            combo_results = {}

            for model_name, config in available_models.items():
                model_class = config['class']
                params = config['params'].copy()
                params['random_state'] = seed

                # 对XGBoost和LightGBM设置use_label_encoder/eval_metric
                if model_name == 'XGBoost' and XGB_AVAILABLE:
                    params.setdefault('use_label_encoder', False)
                    params.setdefault('eval_metric', 'logloss')
                if model_name == 'LightGBM' and LGB_AVAILABLE:
                    params.setdefault('verbose', -1)

                # 数据增强
                X_tr_aug, y_tr_aug = apply_data_augmentation(
                    X_tr, y_train, data_augmentation, random_state=seed
                )

                # 训练模型
                model = model_class(**params)
                model.fit(X_tr_aug, y_tr_aug)

                # 预测
                y_proba = model.predict_proba(X_te)
                y_pred_raw = np.argmax(y_proba, axis=1)

                # Youden阈值
                youden_threshold = None
                if args.youden and n_classes == 2:
                    # 从训练集OOF计算Youden阈值
                    # 注意：ML模型无OOF，直接使用训练集概率估计阈值
                    train_proba = model.predict_proba(X_tr_aug)
                    youden_threshold, youden_j = find_youden_threshold(y_tr_aug, train_proba[:, 1])
                    y_pred = predict_with_youden_threshold(y_proba, youden_threshold)
                else:
                    y_pred = y_pred_raw

                # 评估
                eval_metrics_nested = evaluate_predictions(y_proba, y_pred, y_test, n_classes)
                eval_metrics_flat = flatten_metrics(eval_metrics_nested)

                combo_results[model_name] = {
                    'metrics': eval_metrics_flat,
                    'metrics_nested': eval_metrics_nested,
                    'youden_threshold': float(youden_threshold) if youden_threshold is not None else None,
                }

            seed_results[combo_name] = combo_results

        all_results[seed] = seed_results

        # 每10个seed打印一次进度
        if (i + 1) % 10 == 0 or i == 0:
            sample_combo = list(MODALITY_COMBINATIONS.keys())[0]
            sample_model = list(available_models.keys())[0]
            if sample_combo in seed_results and sample_model in seed_results[sample_combo]:
                m = seed_results[sample_combo][sample_model]['metrics']
                print(f"\n  Seed {seed}: {sample_combo}/{sample_model} "
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
        all_summaries[combo_name] = {}
        for model_name in available_models.keys():
            metrics_accum = {metric: [] for metric in flat_metrics_names}
            thresholds_accum = []

            for seed in seeds:
                if seed in all_results:
                    seed_data = all_results[seed]
                    if combo_name in seed_data and model_name in seed_data[combo_name]:
                        m = seed_data[combo_name][model_name]['metrics']
                        for metric in flat_metrics_names:
                            if metric in m:
                                metrics_accum[metric].append(m[metric])
                        th = seed_data[combo_name][model_name].get('youden_threshold')
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

            all_summaries[combo_name][model_name] = summary

    # 打印汇总
    for combo_name in MODALITY_COMBINATIONS.keys():
        print(f"\n--- {combo_name} ---")
        for model_name in available_models.keys():
            s = all_summaries[combo_name][model_name]
            if 'accuracy' in s:
                print(f"  {model_name}: "
                      f"Acc={s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f}, "
                      f"F1={s['f1_macro']['mean']:.4f}±{s['f1_macro']['std']:.4f}, "
                      f"AUC={s['roc_auc_macro']['mean']:.4f}±{s['roc_auc_macro']['std']:.4f}")

    # 6. 保存结果
    print("\n6. 保存结果...")

    # 汇总表
    summary_rows = []
    for combo_name in MODALITY_COMBINATIONS.keys():
        for model_name in available_models.keys():
            s = all_summaries[combo_name][model_name]
            row = {'Modality': combo_name, 'Model': model_name}
            for metric in flat_metrics_names:
                if metric in s:
                    row[metric] = f"{s[metric]['mean']:.4f}±{s[metric]['std']:.4f}"
            if 'youden_threshold_mean' in s:
                row['youden_threshold'] = f"{s['youden_threshold_mean']:.4f}±{s['youden_threshold_std']:.4f}"
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, "ml_baselines_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  汇总结果保存至: {summary_path}")

    # 详细结果表（每个seed每个combo每个model一行）
    detailed_rows = []
    for seed in seeds:
        if seed not in all_results:
            continue
        for combo_name in MODALITY_COMBINATIONS.keys():
            if combo_name not in all_results[seed]:
                continue
            for model_name in available_models.keys():
                if model_name not in all_results[seed][combo_name]:
                    continue
                m = all_results[seed][combo_name][model_name]
                row = {'seed': seed, 'modality': combo_name, 'model': model_name}
                for metric, value in m['metrics'].items():
                    row[metric] = value
                row['youden_threshold'] = m.get('youden_threshold', '')
                detailed_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = os.path.join(results_dir, "ml_baselines_detailed.csv")
    detailed_df.to_csv(detailed_path, index=False)
    print(f"  详细结果保存至: {detailed_path}")

    # 所有汇总指标的mean/std以宽格式保存（便于快速查看）
    means_rows = []
    for combo_name in MODALITY_COMBINATIONS.keys():
        row = {'Modality': combo_name}
        for model_name in available_models.keys():
            s = all_summaries[combo_name][model_name]
            for metric in ['accuracy', 'f1_macro', 'roc_auc_macro', 'prauc_macro']:
                if metric in s:
                    row[f'{model_name}_{metric}'] = f"{s[metric]['mean']:.4f}±{s[metric]['std']:.4f}"
        means_rows.append(row)

    means_df = pd.DataFrame(means_rows)
    means_path = os.path.join(results_dir, "ml_baselines_wide_summary.csv")
    means_df.to_csv(means_path, index=False)
    print(f"  宽格式汇总保存至: {means_path}")

    # JSON格式完整结果
    results_json = {
        'modality_combinations': {k: v for k, v in MODALITY_COMBINATIONS.items()},
        'models': list(available_models.keys()),
        'summary': {
            combo: {
                model: {
                    metric: {'mean': v['mean'], 'std': v['std']}
                    for metric, v in model_summary.items()
                    if isinstance(v, dict) and 'mean' in v
                }
                for model, model_summary in combo_data.items()
            }
            for combo, combo_data in all_summaries.items()
        },
        'data_augmentation': data_augmentation,
        'threshold_method': 'youden' if args.youden else 'argmax',
        'n_splits': args.n_splits,
        'n_classes': n_classes,
        'feature_dims': feature_dims,
        'total_samples': len(y),
    }

    results_json_path = os.path.join(results_dir, "ml_baselines_results.json")
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
        'threshold_method': 'youden' if args.youden else 'argmax',
        'n_classes': n_classes,
        'feature_dims': feature_dims,
        'total_samples': len(y),
        'models': list(available_models.keys()),
        'modality_combinations': {k: v for k, v in MODALITY_COMBINATIONS.items()},
    }

    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  实验配置保存至: {config_path}")

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

    print(f"\n{'=' * 80}")
    print("TCGA HRD数据 机器学习基线评估完成！")
    print(f"结果目录: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
