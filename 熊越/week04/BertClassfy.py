import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# 代码使用trainer
dataset = pd.read_csv('train.csv', sep=',', header=0)
# 初始化 LabelEncoder，用于将文本标签转换为数字标签

lbl = LabelEncoder()
# 拟合数据并转换前500个标签，得到数字标签
labels = lbl.fit_transform(dataset["label"].values[:1000])
# 提取前500个文本内容
texts = list(dataset["text"].values[:1000])
number_layers = dataset["label"].nunique()
aa=set(dataset["label"])
# 加载本地bert模型
bert_path = "./bert-base-chinese"
tokenizers = AutoTokenizer.from_pretrained(bert_path)
model = AutoModelForSequenceClassification.from_pretrained(bert_path, num_labels=number_layers)

x_train, x_temp, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"训练: {len(x_train)}, 验证: {len(x_val)}, 测试: {len(x_test)}")
# 词袋模型，input_dis [1,1223,42213,555]代表每个字通过tokenizer分割后，文字所处在中文集中的下标
# attention_mask 数值为1和0,1代表有值，0代表填充符
max_len = 128
train_enc = tokenizers(x_train, max_length=max_len, truncation=True, padding=True)
val_enc = tokenizers(x_val, max_length=max_len, truncation=True, padding=True)
test_enc = tokenizers(x_test, max_length=max_len, truncation=True, padding=True)
# datasets下Dataset.from_dict组装词典
train_dataset = Dataset.from_dict({
    "input_ids": train_enc["input_ids"],
    "attention_mask": train_enc["attention_mask"],
    "labels": train_labels
})
val_dataset = Dataset.from_dict({
    "input_ids": val_enc["input_ids"],
    "attention_mask": val_enc["attention_mask"],
    "labels": y_val
})
test_dataset = Dataset.from_dict({
    "input_ids": test_enc["input_ids"],
    "attention_mask": test_enc["attention_mask"],
    "labels": y_test
})


# 7. 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,  # 增加轮数
    per_device_train_batch_size=16,  #  训练batch_size
    per_device_eval_batch_size=32,
    learning_rate=2e-5,  # 稍大学习率
    warmup_ratio=0.1,  # 预热比例更科学
    weight_decay=0.01,  # 权重衰退
    logging_dir='./logs',
    logging_steps=50,  # 日志保存
    eval_strategy="epoch",  # 每个epoch保存评估模型
    save_strategy="epoch",  # 每个epoch保存训练模型
    load_best_model_at_end=True,  # 保存最好模型
    metric_for_best_model="eval_accuracy", # 保存准确率最高模型
    greater_is_better=True,  # 越高越好，false为越低越好
    report_to="none",  # 禁用 wandb 等
)


# 8. 评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': np.mean(predictions == labels)
    }


# 9. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # 用验证集选模型
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 早停
)

trainer.train()

# 10. 最终测试集评估
print("\n=== 验证集最佳结果 ===")
best_results = trainer.evaluate(val_dataset)
print(f"验证集准确率: {best_results['eval_accuracy']:.2%}")

print("\n=== 最终测试集评估 ===")
test_results = trainer.evaluate(test_dataset)
print(f"测试集准确率: {test_results['eval_accuracy']:.2%}")

trainer.save_model("./results/model")
tokenizers.save_pretrained("./results/model")



