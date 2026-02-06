import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# -------------------------- 1. 数据加载与预处理（适配新情感数据集） --------------------------
# 加载修改后的情感数据集（注意：新数据集有表头，sep默认逗号，无需指定sep="\t"）
dataset_df = pd.read_csv("ChnSentiCorp_sentiment_3class_1000.csv")  # 新数据集含"review"和"情感标签"表头

# 查看数据集基本信息，确认数据格式正确
print("新情感数据集前5行：")
print(dataset_df.head())
print(f"\n数据集形状：{dataset_df.shape}")
print(f"情感标签分布：\n{dataset_df['情感标签'].value_counts()}")

# 初始化LabelEncoder，将"正向评论"/"负面评论"转换为数字标签（0和1）
lbl = LabelEncoder()
# 对情感标签列进行编码（新数据集标签列名为"情感标签"，非原代码的索引1）
labels = lbl.fit_transform(dataset_df["情感标签"].values)
# 提取评论内容列（新数据集评论列名为"review"，非原代码的索引0）
texts = list(dataset_df["review"].values)

# 打印标签映射关系，方便后续解读结果
print(f"\n标签编码映射：{dict(zip(lbl.classes_, lbl.transform(lbl.classes_)))}")

# 分割数据为训练集（80%）和测试集（20%），stratify确保正负样本分布一致
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,  # 评论文本数据
    labels,  # 编码后的数字标签
    test_size=0.2,  # 测试集比例
    stratify=labels,  # 保持标签分布与原数据一致
    random_state=42  # 固定随机种子，确保结果可复现
)

print(f"\n训练集样本数：{len(x_train)}，测试集样本数：{len(x_test)}")
print(f"训练集标签分布：{np.bincount(train_labels)}（对应{lbl.classes_}）")
print(f"测试集标签分布：{np.bincount(test_labels)}（对应{lbl.classes_}）")

# -------------------------- 2. 模型与分词器加载（适配二分类任务） --------------------------
# 加载bert-base-chinese分词器（若模型路径不同，需修改为实际路径，如'./models/bert-base-chinese'）
device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained('../models/google-bert/bert-base-chinese')  # 若本地有模型，替换为本地路径

# 加载Bert分类模型：新任务是3分类（正向/负面/中性），num_labels设为2（原代码17类）
model = BertForSequenceClassification.from_pretrained(
    '../models/google-bert/bert-base-chinese',  # 若本地有模型，替换为本地路径（如'../models/google-bert/bert-base-chinese'）
    num_labels=3
)
model.to(device)
# -------------------------- 3. 文本编码（与原逻辑一致，适配评论数据） --------------------------
# 对训练集文本进行编码：截断（truncation）、填充（padding）到固定长度64
train_encodings = tokenizer(
    x_train,
    truncation=True,  # 文本长度超过max_length时截断
    padding=True,  # 文本长度不足时填充到max_length
    max_length=64,  # 固定序列长度（根据评论长度调整，64足够覆盖多数酒店评论）
    return_tensors=None  # 不返回tensor，后续由Trainer自动处理
)

# 对测试集文本进行同样编码
test_encodings = tokenizer(
    x_test,
    truncation=True,
    padding=True,
    max_length=64,
    return_tensors=None
)

# -------------------------- 4. 转换为Hugging Face Dataset格式（适配Trainer） --------------------------
# 训练集转换：包含input_ids（token编码）、attention_mask（注意力掩码）、labels（标签）
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})

# 测试集转换
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})


# -------------------------- 5. 评估指标函数（与原逻辑一致） --------------------------
def compute_metrics(eval_pred):
    """定义评估指标：计算准确率（适用于二分类）"""
    # eval_pred是元组：(模型预测的logits, 真实标签)
    logits, true_labels = eval_pred
    # 取logits最大值索引作为预测结果（二分类时，索引0/1对应负面/正向）
    pred_labels = np.argmax(logits, axis=-1)
    # 计算准确率：预测正确的样本数 / 总样本数
    accuracy = (pred_labels == true_labels).mean()
    return {'accuracy': round(accuracy, 4)}  # 返回准确率，保留4位小数


# -------------------------- 6. 训练参数配置（微调适配二分类） --------------------------
training_args = TrainingArguments(
    output_dir='./sentiment_results',  # 训练结果保存目录（修改为情感分类专属目录，避免冲突）
    num_train_epochs=4,  # 调整epoch数：二分类任务简单，3轮足够（原4轮）
    per_device_train_batch_size=8,  # 批次大小：根据设备内存调整（CPU建议8，GPU建议16）
    per_device_eval_batch_size=8,  # 评估批次大小，与训练一致
    warmup_steps=50,  # 学习率预热步数：原500步过多（总步数少），改为100步
    weight_decay=0.01,  # 权重衰减：防止过拟合，保持不变
    logging_dir='./sentiment_logs',  # 日志保存目录（情感分类专属）
    logging_steps=10,  # 日志打印步数：原100步间隔大，改为20步更及时
    eval_strategy="epoch",  # 每轮epoch结束后评估（保持不变）
    save_strategy="epoch",  # 每轮epoch结束后保存模型（保持不变）
    load_best_model_at_end=True,  # 训练结束后加载最优模型（基于验证集准确率）
    use_cpu=True
)

# -------------------------- 7. 启动训练与评估 --------------------------
# 实例化Trainer：整合模型、参数、数据集、评估指标
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 开始训练（自动处理批次、梯度、参数更新）
print("\n开始训练模型...")
trainer.train()

# 训练结束后，在测试集上进行最终评估
print("\n开始测试集评估...")
eval_results = trainer.evaluate()
print(f"\n测试集最终结果：{eval_results}")


# -------------------------- 8. 简单预测示例（可选，验证模型效果） --------------------------
def predict_sentiment(text):
    """用训练好的模型预测单条评论的情感"""
    # 文本编码
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors='pt'  # 返回torch tensor
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}  # 新增：数据放到CPU
    # 模型推理（关闭梯度计算，加速）
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        # 计算概率（softmax）并取预测标签
        probs = torch.softmax(logits, dim=1)
        pred_label_idx = torch.argmax(probs, dim=1).item()
        pred_label = lbl.inverse_transform([pred_label_idx])[0]  # 还原为文本标签
        pred_prob = probs[0][pred_label_idx].item()  # 预测概率

    return f"评论：{text}\n预测情感：{pred_label}\n置信度：{round(pred_prob, 4)}"


# 测试2条评论的预测效果
test_text1 = "酒店环境很好，服务贴心，下次还来！"
test_text2 = "房间又脏又小，隔音差，再也不会住了。"
test_text3 = "周围有医院和商场，挺方便的。"
print("\n预测示例1：")
print(predict_sentiment(test_text1))#//正向评论
print("\n预测示例2：")
print(predict_sentiment(test_text2))#//负面评论
print("\n预测示例3：")
print(predict_sentiment(test_text3)) #//中性评论
