import numpy as np
from safetensors import torch
from torch import cuda
import torch
from config import BERT_MODEL_PKL_PATH, MODEL_PATH, CATEGORY_NAME
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# 自定义数据集 -》加载模型 -》 模型分类
device = 'cuda' if cuda.is_available() else 'cpu'
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=12)
model.load_state_dict(torch.load(BERT_MODEL_PKL_PATH))
model.eval()
model.to(device)

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

def model_for_bert(request_text: str) -> str:
    request_encoding = tokenizer(request_text, truncation=True, padding=True, max_length=64, return_tensors='pt')
    request_dataset = {key: torch.tensor(val) for key, val in request_encoding.items()}
    request_dataset['labels'] = torch.tensor(int(0))

    input_ids = request_dataset['input_ids'].to(device)
    attention_mask = request_dataset['attention_mask'].to(device)
    labels = request_dataset['labels'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs[1]
    logits = logits.detach().cpu().numpy()
    prediction = np.argmax(logits, axis=-1)[0]
    return CATEGORY_NAME[prediction]

request_text = 'i have been on a roller coaster of emotions over these supposed feelings that something unpleasant was coming'
print(model_for_bert(request_text))
