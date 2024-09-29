import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import time
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv('train.csv', encoding='utf-8')

X, y = df.iloc[:, 2:202], df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # 变成列向量
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)  # 变成列向量

class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 增加第一层隐藏层的神经元数量
        self.fc2 = nn.Linear(128, 64)         # 第二层隐藏层
        self.fc3 = nn.Linear(64, 32)          # 第三层隐藏层
        self.fc4 = nn.Linear(32, 1)           # 输出层
        self.relu = nn.ReLU()                 # 使用ReLU作为激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)

# 初始化模型
input_dim = X_train.shape[1]
model = DeepNN(input_dim)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 200
train_begin = time.time()
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
train_end = time.time()
print(f"Training Time: {train_end - train_begin}")

# 评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predicted = (test_outputs >= 0.5).float()
    accuracy = accuracy_score(y_test, predicted)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Classification Report:\n{classification_report(y_test, predicted)}')

    # 计算AUC
    test_probs = torch.sigmoid(test_outputs).numpy()
    roc_auc = roc_auc_score(y_test, test_probs)
    print(f'AUC: {roc_auc:.4f}')

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, test_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f'DeepNN (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    # 特征重要性
    feature_importance = np.abs(model.fc1.weight.detach().numpy()).mean(axis=0)
    sorted_idx = np.argsort(feature_importance)[::-1][:6]  # 获取前六个最重要的特征索引
    top_features = df.columns[2:202][sorted_idx]

    plt.figure()
    plt.bar(range(6), feature_importance[sorted_idx], tick_label=top_features)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Top 6 Feature Importance')
    plt.show()
