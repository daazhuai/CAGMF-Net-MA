import pandas as pd
import torch
import torch.nn.functional as F
import pickle
import json
import os
import numpy as np

# 修复 OMP_NUM_THREADS 问题 - 在导入其他库之前设置
os.environ['OMP_NUM_THREADS'] = '1'  # 设置为有效数值


# ==================== 模型定义 ====================
class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, hidden_dim)

    def forward(self, x): return F.relu(self.fc(x))


class Gate(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = torch.nn.Linear(dim * 2, dim)

    def forward(self, z_k, z_ref): return torch.sigmoid(self.fc(torch.cat([z_k, z_ref], dim=1))) * z_k


class MultiOmicNet(torch.nn.Module):
    def __init__(self, dims, hidden, n_class):
        super().__init__()
        self.mlp = torch.nn.ModuleDict({k: MLP(dims[k], hidden) for k in dims})
        self.gate = torch.nn.ModuleDict({k: Gate(hidden) for k in dims if k != "clin"})
        self.classifier = torch.nn.Linear(hidden, n_class)
        self.used_modalities = list(dims.keys())

    def forward(self, xs):
        z = {k: self.mlp[k](xs[k]) for k in self.used_modalities}
        z_ref = z["clin"] if "clin" in z else z[list(z.keys())[0]]
        fused = z_ref
        for k in z:
            if k != "clin" and k != list(z.keys())[0]:
                fused = fused + self.gate[k](z[k], z_ref)
        return self.classifier(fused)


# ==================== 预测器类 ====================
class CancerPredictor:
    def __init__(self, model_dir):
        """初始化：加载所有模型组件"""
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载配置和编码器
        with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
            self.config = json.load(f)
        with open(os.path.join(model_dir, 'scalers.pkl'), 'rb') as f:
            self.scalers = pickle.load(f)
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        with open(os.path.join(model_dir, 'clinical_encoders.pkl'), 'rb') as f:
            self.clinical_encoders = pickle.load(f)

        print(f"模型信息:")
        print(f"  类别: {self.label_encoder.classes_}")
        print(f"  特征维度: {self.config['feature_dimensions']}")

        # 加载所有模型
        self.models = []
        self.weights = []
        print(f"加载 {len(self.config['candidate_models'])} 个模型...")

        for i, modalities in enumerate(self.config['candidate_models']):
            model_path = os.path.join(model_dir, f"model_{i + 1}_{'_'.join(modalities)}.pth")
            dims = {m: self.config['feature_dimensions'][m] for m in modalities}
            model = MultiOmicNet(dims, 128, self.config['n_classes'])

            # 兼容不同版本的PyTorch
            try:
                state_dict = torch.load(model_path, map_location=self.device)
            except RuntimeError:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)

            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.models.append(model)
            self.weights.append(self.config['model_weights'][i])

        self.weights = torch.tensor(self.weights) / sum(self.weights)
        print(f"模型加载完成！设备: {self.device}")

    def predict(self, clin_df=None, cnv_df=None, snv_df=None, mrna_df=None):
        """
        预测函数
        返回: (labels, probabilities) 元组
        """
        # 数据预处理
        processed = {}
        n_samples = None

        # 确定样本数
        for df in [clin_df, cnv_df, snv_df, mrna_df]:
            if df is not None:
                n_samples = len(df)
                break

        if n_samples is None:
            raise ValueError("至少需要提供一个模态的数据")

        for name, df in [('clin', clin_df), ('cnv', cnv_df), ('snv', snv_df), ('mrna', mrna_df)]:
            if df is not None:
                # 检查维度
                expected_dim = self.config['feature_dimensions'][name]
                if df.shape[1] != expected_dim:
                    print(f"警告: {name} 数据维度不匹配！期望 {expected_dim}，实际 {df.shape[1]}")
                    # 调整维度
                    if df.shape[1] < expected_dim:
                        # 填充随机列
                        for i in range(df.shape[1], expected_dim):
                            df[f'dummy_{i}'] = 0
                    else:
                        df = df.iloc[:, :expected_dim]

                if name == 'clin':
                    df = df.copy()
                    for col, enc in self.clinical_encoders.items():
                        if col in df.columns:
                            df[col] = df[col].astype(str).map(enc['mapping']).fillna(0)

                df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                processed[name] = torch.tensor(
                    self.scalers[name].transform(df.values.astype('float32')),
                    dtype=torch.float32
                ).to(self.device)

        # 集成预测
        preds = []
        used_weights = []

        for i, model in enumerate(self.models):
            mods = self.config['candidate_models'][i]
            if all(m in processed for m in mods):
                with torch.no_grad():
                    out = model({m: processed[m] for m in mods})
                    preds.append(F.softmax(out, dim=1).cpu() * self.weights[i])
                    used_weights.append(self.weights[i])

        if not preds:
            raise ValueError(f"没有可用的模型进行预测。提供的模态: {list(processed.keys())}")

        final_probs = sum(preds) / sum(used_weights)
        pred_indices = torch.argmax(final_probs, dim=1).numpy()
        pred_labels = self.label_encoder.inverse_transform(pred_indices)

        return pred_labels, final_probs.numpy()


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 极简调用示例
    import pandas as pd
    import numpy as np
    from predict import CancerPredictor

    # 1. 加载模型
    predictor = CancerPredictor("/root/METAtrain/saved_models_clin_cnv_snv_mrna")

    # 2. 准备数据 (6维临床 + 7维CNV + 15维SNV + 50维mRNA)
    # 以下为随机数示例
    clin_data = pd.DataFrame(np.random.randn(5, 6))  # 5个样本，6个特征
    cnv_data = pd.DataFrame(np.random.randn(5, 7))  # 5个样本，7个特征
    snv_data = pd.DataFrame(np.random.randn(5, 15))  # 二值数据
    mrna_data = pd.DataFrame(np.random.randn(5, 50))  # 5个样本，50个特征

    # 3. 进行预测
    print("\n进行预测...")
    labels, probabilities = predictor.predict(clin_data, cnv_data, snv_data, mrna_data)

    # 4. 输出结果
    print("\n" + "=" * 40)
    print("预测结果:")
    print("=" * 40)

    for i in range(5):
        print(f"\n样本 {i + 1}:")
        print(f"  预测类别: {labels[i]}")
        print(f"  类别概率:")
        for j, class_name in enumerate(predictor.label_encoder.classes_):
            print(f"    {class_name}: {probabilities[i][j]:.4f}")