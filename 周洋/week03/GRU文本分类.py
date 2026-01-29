import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 读取数据
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签编码
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 字符编码
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# 最大文本长度
max_len = 40


# 自定义数据集
class CharGRUDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 截断或填充文本
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# GRU 分类器模型
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False):
        super(GRUClassifier, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # GRU层 (替换LSTM)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        # 如果使用双向GRU，隐藏维度需要加倍
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 全连接层
        self.fc = nn.Linear(gru_output_dim, output_dim)

        # 保存参数供forward使用
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    def forward(self, x):
        # 1. 嵌入层: batch_size * seq_len -> batch_size * seq_len * embedding_dim
        embedded = self.embedding(x)

        # 2. GRU层: batch_size * seq_len * embedding_dim -> batch_size * seq_len * hidden_dim
        # GRU只返回output和hn，没有细胞状态cn
        gru_out, hn = self.gru(embedded)

        # 3. 提取最后一个时间步的隐藏状态用于分类
        # hn的形状: (num_layers * num_directions, batch_size, hidden_dim)
        if self.bidirectional:
            # 双向GRU: 取最后两个方向的隐藏状态拼接
            forward_hidden = hn[-2, :, :]  # 前向最后层
            backward_hidden = hn[-1, :, :]  # 后向最后层
            hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            # 单向GRU: 取最后一层的隐藏状态
            hidden = hn[-1, :, :]

        # 4. 全连接层: hidden_dim -> output_dim
        out = self.fc(hidden)

        return out


# 创建数据集和数据加载器
gru_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

# 模型参数
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
num_layers = 1  # GRU层数
bidirectional = False  # 是否使用双向GRU

# 初始化模型、损失函数和优化器
model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Epoch {epoch + 1}, Batch {idx}, Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")


# 分类函数
def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
    # 文本编码
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # 添加batch维度

    # 预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    # 获取预测标签
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 创建反向标签映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试模型
test_texts = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "播放周杰伦的歌",
    "今天有什么新闻",
    "设置明天早上7点的闹钟"
]

print("\n" + "=" * 50)
print("GRU模型测试结果:")
print("=" * 50)

for text in test_texts:
    predicted_class = classify_text_gru(text, model, char_to_index, max_len, index_to_label)
    print(f"输入: '{text}'")
    print(f"预测: '{predicted_class}'")
    print("-" * 40)
