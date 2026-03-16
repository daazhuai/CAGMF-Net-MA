import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


def read_csv_file(path):
    """读取CSV文件"""
    return pd.read_csv(path)


def normalize_sample_id(x):
    """标准化样本ID"""
    return str(x).split("-01")[0].split("-02")[0]


# 文件路径（请根据实际情况修改）
clinical_path = "./dataset/METABRIC/METABRIC_Clinical.csv"
snv_path = "./dataset/METABRIC/METABRIC_SNV.csv"
cnv_path = "./dataset/METABRIC/METABRIC_CNV.csv"
mrna_path = "./dataset/METABRIC/METABRIC_RNA.csv"


# 读取数据
clin = read_csv_file(clinical_path)
snv = read_csv_file(snv_path)
cnv = read_csv_file(cnv_path)
mrna = read_csv_file(mrna_path)


def set_index_and_clean(df, is_clinical=False):
    """设置索引并清理数据"""
    df = df.copy()

    # 1. 获取样本ID列并设置成索引
    # 处理样本ID列
    sample_id_columns = ['SAMPLE_ID', 'Sample_ID', 'sample_id', 'sample']
    sample_col = None

    # 查找样本ID列
    for col in sample_id_columns:
        if col in df.columns:
            sample_col = col
            break

    # 如果没有找到标准列名，使用第一列
    if sample_col is None and len(df.columns) > 0:
        sample_col = df.columns[0]
        print(f"警告: 未找到标准样本ID列名，使用第一列 '{sample_col}' 作为样本ID")

    if sample_col:
        # 标准化样本ID并设置为索引
        df[sample_col] = df[sample_col].astype(str).map(normalize_sample_id)
        df = df.set_index(sample_col)

        # 重置索引，确保索引成为列，然后再次设置（处理重复索引）
        df = df.reset_index()
        df = df.set_index(sample_col)

    # 2. 提取分类变量并编码
    # 对于clinical数据，除AGE外都是分类变量
    if is_clinical:
        # 提取分类变量列（排除AGE和可能的SUBTYPE）
        categorical_cols = []
        numerical_cols = []

        for col in df.columns:
            if col == 'AGE' or col == 'SUBTYPE':
                continue

            # 尝试转换为数值，如果失败则是分类变量
            try:
                # 尝试转换为数值
                pd.to_numeric(df[col], errors='raise')
                numerical_cols.append(col)
            except:
                categorical_cols.append(col)

        print(f"分类变量列: {categorical_cols}")
        print(f"数值变量列: {numerical_cols}")

        # 对分类变量进行标签编码
        for col in categorical_cols:
            try:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                print(f"已编码列: {col} (唯一值数量: {len(le.classes_)})")
            except Exception as e:
                print(f"警告: 无法编码列 {col}: {e}")
                # 如果编码失败，删除该列
                df = df.drop(columns=[col])

        # 确保数值列是浮点类型
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# 处理各数据表
print("处理clinical数据...")
clin = set_index_and_clean(clin, is_clinical=True)

print("\n处理SNV数据...")
snv = set_index_and_clean(snv)

print("\n处理CNV数据...")
cnv = set_index_and_clean(cnv)

print("\n处理mRNA数据...")
mrna = set_index_and_clean(mrna)

print("\n数据形状:")
print(f"Clinical: {clin.shape}")
print(f"SNV: {snv.shape}")
print(f"CNV: {cnv.shape}")
print(f"mRNA: {mrna.shape}")


# 检查并清理特征数据中的非数值列
def clean_numeric_data(df, name):
    """清理数据框中的非数值数据"""
    df = df.copy()

    # 找出所有非数值列
    non_numeric_cols = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='raise')
        except:
            non_numeric_cols.append(col)

    if non_numeric_cols:
        print(f"警告: {name} 数据中存在非数值列，将被删除: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)

    # 转换为数值类型，非数值转换为NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # 填充NaN值为0（或使用其他策略）
    df = df.fillna(0)

    return df


# 清理特征数据
snv = clean_numeric_data(snv, "SNV")
cnv = clean_numeric_data(cnv, "CNV")
mrna = clean_numeric_data(mrna, "mRNA")

# 提取标签和临床特征
if 'SUBTYPE' in clin.columns:
    y = clin['SUBTYPE']
    clin_feat = clin.drop(columns=['SUBTYPE'])
else:
    raise ValueError("Clinical data must contain 'SUBTYPE' column")

# 清理临床特征数据
clin_feat = clean_numeric_data(clin_feat, "Clinical features")

# 查找共同样本
common_samples = (
        set(snv.index) &
        set(cnv.index) &
        set(mrna.index) &
        set(clin_feat.index) &
        set(y.index)
)

print(f"\n匹配样本数量: {len(common_samples)}")

# 筛选共同样本
snv = snv.loc[common_samples]
cnv = cnv.loc[common_samples]
mrna = mrna.loc[common_samples]
clin_feat = clin_feat.loc[common_samples]
y = y.loc[common_samples]

print("\n筛选后数据形状:")
print(f"SNV: {snv.shape}")
print(f"CNV: {cnv.shape}")
print(f"mRNA: {mrna.shape}")
print(f"Clinical features: {clin_feat.shape}")
print(f"Labels: {y.shape}")

# 数据标准化
scaler = StandardScaler()

# 确保所有数据都是数值类型
X_snv = scaler.fit_transform(snv.values.astype(np.float32))
X_cnv = scaler.fit_transform(cnv.values.astype(np.float32))
X_mrna = scaler.fit_transform(mrna.values.astype(np.float32))
X_clin = scaler.fit_transform(clin_feat.values.astype(np.float32))

# ==================================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        return F.relu(self.fc(x))


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, z_k, z_ref):
        g = torch.sigmoid(self.fc(torch.cat([z_k, z_ref], dim=1)))
        return g * z_k


class MultiOmicNet(nn.Module):
    def __init__(self, dims, hidden, n_class):
        super().__init__()

        self.mlp = nn.ModuleDict({
            k: MLP(dims[k], hidden) for k in dims
        })

        self.gate = nn.ModuleDict({
            k: Gate(hidden) for k in dims if k != "clin"
        })

        self.classifier = nn.Linear(hidden, n_class)

    def forward(self, xs):
        z = {k: self.mlp[k](xs[k]) for k in xs}

        z_ref = z["clin"]
        fused = z_ref
        for k in z:
            if k != "clin":
                fused = fused + self.gate[k](z[k], z_ref)

        return self.classifier(fused)


# 编码标签
le_y = LabelEncoder()
y_enc = le_y.fit_transform(y)
n_classes = len(np.unique(y_enc))

print(f"\n类别数量: {n_classes}")
print(f"类别标签: {le_y.classes_}")

# 转换为PyTorch张量
X = {
    "snv": torch.tensor(X_snv, dtype=torch.float32),
    "cnv": torch.tensor(X_cnv, dtype=torch.float32),
    "mrna": torch.tensor(X_mrna, dtype=torch.float32),
    "clin": torch.tensor(X_clin, dtype=torch.float32),
}

y_t = torch.tensor(y_enc, dtype=torch.long)

print(f"\n特征维度:")
print(f"SNV: {X_snv.shape[1]}")
print(f"CNV: {X_cnv.shape[1]}")
print(f"mRNA: {X_mrna.shape[1]}")
print(f"Clinical: {X_clin.shape[1]}")

# 划分训练集和测试集
indices = np.arange(len(y_t))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_enc)

# 创建数据加载器
batch_size = 32

# 训练集
train_X = {k: X[k][train_idx] for k in X}
train_y = y_t[train_idx]
train_dataset = TensorDataset(*[train_X[k] for k in train_X], train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 测试集
test_X = {k: X[k][test_idx] for k in X}
test_y = y_t[test_idx]
test_dataset = TensorDataset(*[test_X[k] for k in test_X], test_y)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\n数据划分:")
print(f"训练集样本数: {len(train_idx)}")
print(f"测试集样本数: {len(test_idx)}")

# 初始化模型
model = MultiOmicNet(
    dims={
        "snv": X_snv.shape[1],
        "cnv": X_cnv.shape[1],
        "mrna": X_mrna.shape[1],
        "clin": X_clin.shape[1],
    },
    hidden=128,
    n_class=n_classes
)

# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n使用设备: {device}")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练循环
epochs = 50
train_losses = []
val_losses = []
val_accuracies = []

print("\n开始训练...")
print("=" * 60)

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for batch_X_snv, batch_X_cnv, batch_X_mrna, batch_X_clin, batch_y in train_loader:
        batch_X = {
            "snv": batch_X_snv.to(device),
            "cnv": batch_X_cnv.to(device),
            "mrna": batch_X_mrna.to(device),
            "clin": batch_X_clin.to(device),
        }
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_X_snv.size(0)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X_snv, batch_X_cnv, batch_X_mrna, batch_X_clin, batch_y in test_loader:
            batch_X = {
                "snv": batch_X_snv.to(device),
                "cnv": batch_X_cnv.to(device),
                "mrna": batch_X_mrna.to(device),
                "clin": batch_X_clin.to(device),
            }
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X_snv.size(0)

            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 计算指标
    train_loss = train_loss / len(train_idx)
    val_loss = val_loss / len(test_idx)
    val_accuracy = 100 * correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # 打印训练进度
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1:3d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}%")

    scheduler.step()

print("=" * 60)
print("训练完成！")

# 最终评估
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch_X_snv, batch_X_cnv, batch_X_mrna, batch_X_clin, batch_y in test_loader:
        batch_X = {
            "snv": batch_X_snv.to(device),
            "cnv": batch_X_cnv.to(device),
            "mrna": batch_X_mrna.to(device),
            "clin": batch_X_clin.to(device),
        }
        batch_y = batch_y.to(device)

        outputs = model(batch_X)
        probs = F.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# 计算评估指标
final_accuracy = accuracy_score(all_labels, all_preds)

print("\n" + "=" * 60)
print("模型评估结果:")
print("=" * 60)
print(f"最终测试准确率: {final_accuracy * 100:.2f}%")

# 多分类的ROC AUC（需要one-hot编码）
if n_classes > 2:
    try:
        from sklearn.preprocessing import label_binarize

        y_test_bin = label_binarize(all_labels, classes=range(n_classes))
        roc_auc = roc_auc_score(y_test_bin, all_probs, multi_class='ovr', average='weighted')
        print(f"ROC AUC (weighted): {roc_auc:.4f}")
    except:
        print("无法计算多分类ROC AUC")

# 分类报告
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=le_y.classes_))

# 混淆矩阵
print("\n混淆矩阵:")
conf_mat = confusion_matrix(all_labels, all_preds)
print(conf_mat)

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'label_encoder': le_y,
    'scaler': scaler,
    'feature_dims': {
        "snv": X_snv.shape[1],
        "cnv": X_cnv.shape[1],
        "mrna": X_mrna.shape[1],
        "clin": X_clin.shape[1],
    }
}, 'multiomic_model.pth')

print(f"\n模型已保存到: multiomic_model.pth")