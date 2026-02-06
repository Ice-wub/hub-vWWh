from datasets import load_dataset
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from config import MODEL_PATH, BERT_MODEL_PKL_PATH, CATEGORY_NAME
import numpy as np

dataset = load_dataset("parquet", data_files=r'D:\八斗学院\test-00000-of-00001.parquet')

labels = dataset['train']['label'][:500]
# import pdb; pdb.set_trace()
texts = dataset['train']['text'][:500]

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=12)

x_train, x_test, labels_train, labels_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels,
)

train_Encoding = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_Encoding = tokenizer(x_test, truncation=True, padding=True, max_length=64)

train_dataset = Dataset.from_dict({
    'input_ids': train_Encoding['input_ids'],
    'attention_mask': train_Encoding['attention_mask'],
    'labels': labels_train,
})
test_dataset = Dataset.from_dict({
    'input_ids': test_Encoding['input_ids'],
    'attention_mask': test_Encoding['attention_mask'],
    'labels': labels_test,
})

def compute_metrics(pred):
    logits, label = pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == label).mean()}

arguments = TrainingArguments(
    output_dir='./results',
    logging_dir='./logs',
    logging_steps=100,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    best_model = BertForSequenceClassification.from_pretrained(best_model_path)
    torch.save(best_model.state_dict(), BERT_MODEL_PKL_PATH)
    print('Saved model to disk')
else:
    print('No saved model')