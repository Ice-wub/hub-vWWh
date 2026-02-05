import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
import json
import joblib
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler

def load_product_data():
    data_dir = "winwin_inc/product-classification-hiring-demo"
    # 读取训练数据
    train_data = []
    with open(f"{data_dir}/train.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                train_data.append({
                    'text': item['product_name'],
                    'label': item['category']
                })

    # 读取测试数据
    test_data = []
    with open(f"{data_dir}/test.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                test_data.append({
                    'text': item['product_name'],
                    'label': item['category']
                })

    # 合并数据用于标签编码
    all_data = train_data + test_data
    dataset_df = pd.DataFrame(all_data)

    print(f"总数据量: {len(dataset_df)} 条记录")
    print(f"商品类别: {dataset_df['label'].nunique()} 个")
    print(f"类别分布:\n{dataset_df['label'].value_counts()}")

    return train_data, test_data

# 加载和预处理数据
train_data, test_data = load_product_data()

# 转换为DataFrame格式
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

print(f"\n原始数据集划分:")
print(f"训练集: {len(train_df)} 条")
print(f"测试集: {len(test_df)} 条")

# 处理数据不平衡问题 - 使用过采样
print("\n处理数据不平衡问题...")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(
    train_df[['text']], 
    train_df['label']
)
train_df_balanced = pd.DataFrame({
    'text': X_resampled['text'].values,
    'label': y_resampled
})

print(f"平衡后训练集: {len(train_df_balanced)} 条")

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# 拟合所有标签数据
all_labels = pd.concat([train_df_balanced['label'], test_df['label']])
lbl.fit(all_labels.values)

# 转换训练集和测试集标签
train_labels = lbl.transform(train_df_balanced['label'].values)
test_labels = lbl.transform(test_df['label'].values)

# 提取文本内容
x_train = list(train_df_balanced['text'].values)
x_test = list(test_df['text'].values)

print(f"\n最终数据集划分:")
print(f"训练集: {len(x_train)} 条")
print(f"测试集: {len(x_test)} 条")
print(f"类别数: {len(lbl.classes_)}")
print(f"类别列表: {list(lbl.classes_)}")

# 使用更好的预训练模型
print("\n加载预训练模型...")
tokenizer = BertTokenizer.from_pretrained('models/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('models/bert-base-chinese', num_labels=len(lbl.classes_))

# 使用更长的序列长度以适应商品名称
print("对文本进行编码...")
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=128)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],  # 文本的token ID
    'attention_mask': train_encodings['attention_mask'],  # 注意力掩码
    'labels': train_labels  # 对应的标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    accuracy = (predictions == labels).mean()

    # 计算F1分数
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 配置优化的训练参数
print("配置训练参数...")
training_args = TrainingArguments(
    output_dir='./product_results_optimized',  # 训练输出目录
    num_train_epochs=2,  # 增加训练轮数
    per_device_train_batch_size=64,  # 批量大小
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,   # 减少梯度累积
    learning_rate=2e-5,  # 设置合适的学习率
    warmup_ratio=0.1,  # 使用比例而不是固定步数
    weight_decay=0.01,
    logging_dir='./logs_optimized',
    logging_steps=100,  # 更频繁的日志记录
    evaluation_strategy="steps",  # 每隔一定步数评估
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # 使用F1作为最优模型标准
    greater_is_better=True,
    seed=42,  # 设置随机种子确保可复现
)

# 实例化 Trainer 简化模型训练代码
print("初始化训练器...")
trainer = Trainer(
    model=model,  # 要训练的模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=test_dataset,  # 评估数据集
    compute_metrics=compute_metrics,  # 用于计算评估指标的函数
)

# 深度学习训练过程
print("🚀 开始训练商品分类模型...")
# 开始训练模型
trainer.train()

print("🎯 在测试集上进行最终评估...")
# 在测试集上进行最终评估
eval_results = trainer.evaluate()
print(f"\n📈 最终评估结果:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

# 保存最终模型
print("💾 保存训练好的模型...")
model.save_pretrained("./final_product_model_optimized")
tokenizer.save_pretrained("./final_product_model_optimized")
joblib.dump(lbl, "./final_product_model_optimized/label_encoder.pkl")
