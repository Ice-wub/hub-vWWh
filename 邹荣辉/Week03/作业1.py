import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
dataset = pd.read_csv("C:\\Users\\Administrator\\Desktop\\研究生\\ai2026\\AI大模型学习\\Week01\\dataset.csv", sep="\t",
                      header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40

class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# --- 定义三种模型 ---
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden_state = self.rnn(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)#LSTM与RNN、GRU的区别，它有一个细胞单元
        out = self.fc(hidden_state.squeeze(0))
        return out

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out


# --- 训练函数 ---
def train_model(model, model_name, dataloader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []  # 记录每个epoch的平均损失
    batch_losses = []  # 记录每个batch的损失

    print(f"\n{'=' * 50}")
    print(f"开始训练 {model_name}")
    print(f"{'=' * 50}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_losses.append(loss.item())
            batch_count += 1

            if idx % 20 == 0:
                print(f"{model_name} - Epoch [{epoch + 1}/{num_epochs}], Batch {idx}, Loss: {loss.item():.4f}")

        avg_epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        print(f"{model_name} - Epoch [{epoch + 1}/{num_epochs}], 平均Loss: {avg_epoch_loss:.4f}")

    return epoch_losses, batch_losses


# --- 分类函数 ---
def classify_text(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# --- 主程序 ---
if __name__ == "__main__":
    # 准备数据集
    dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型参数
    embedding_dim = 64
    hidden_dim = 128
    output_dim = len(label_to_index)
    num_epochs = 10

    # 创建三种模型
    models = {
        "RNN": RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
        "LSTM": LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
        "GRU": GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    }

    # 训练所有模型并收集损失
    all_epoch_losses = {}
    all_batch_losses = {}

    for model_name, model in models.items():
        epoch_losses, batch_losses = train_model(model, model_name, dataloader, num_epochs)
        all_epoch_losses[model_name] = epoch_losses
        all_batch_losses[model_name] = batch_losses

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 绘制每个epoch的平均损失
    ax1 = axes[0]
    for model_name, losses in all_epoch_losses.items():
        ax1.plot(range(1, len(losses) + 1), losses, marker='o', label=model_name, linewidth=2)

    ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('平均损失 (Loss)', fontsize=12)
    ax1.set_title('三种RNN变体的训练损失对比 (按Epoch)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 绘制batch损失曲线
    ax2 = axes[1]
    window_size = 20  # 移动平均窗口大小

    for model_name, batch_losses in all_batch_losses.items():
        smoothed_losses = []
        for i in range(len(batch_losses)):
            start = max(0, i - window_size)
            end = i + 1
            avg_loss = np.mean(batch_losses[start:end])
            smoothed_losses.append(avg_loss)

        ax2.plot(smoothed_losses, label=model_name, alpha=0.8, linewidth=1.5)

    ax2.set_xlabel('训练批次 (Batch)', fontsize=12)
    ax2.set_ylabel('损失(Loss)', fontsize=12)
    ax2.set_title('三种RNN变体的训练损失对比 (按Batch)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rnn_models_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印最终损失
    print("\n" + "=" * 60)
    print("最终损失对比:")
    print("=" * 60)
    for model_name, losses in all_epoch_losses.items():
        print(
            f"{model_name}: 初始损失 = {losses[0]:.4f}, 最终损失 = {losses[-1]:.4f}, 改善 = {(1 - losses[-1] / losses[0]) * 100:.1f}%")

    # 使用三个模型进行预测
    index_to_label = {i: label for label, i in label_to_index.items()}

    test_texts = ["帮我导航到北京", "查询明天北京的天气", "播放周杰伦的歌", "打开微信"]

    print("\n" + "=" * 60)
    print("模型预测结果对比:")
    print("=" * 60)
    print(f"{'输入文本':<20} {'RNN预测':<12} {'LSTM预测':<12} {'GRU预测':<12}")
    print("-" * 60)

    for test_text in test_texts:
        rnn_pred = classify_text(test_text, models["RNN"], char_to_index, max_len, index_to_label)
        lstm_pred = classify_text(test_text, models["LSTM"], char_to_index, max_len, index_to_label)
        gru_pred = classify_text(test_text, models["GRU"], char_to_index, max_len, index_to_label)
        print(f"{test_text:<20} {rnn_pred:<12} {lstm_pred:<12} {gru_pred:<12}")
