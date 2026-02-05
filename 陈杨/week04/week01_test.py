import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# 加载模型和分词器
model = BertForSequenceClassification.from_pretrained("./final_product_model_optimized")
tokenizer = BertTokenizer.from_pretrained("./final_product_model_optimized")
lbl = joblib.load("./final_product_model_optimized/label_encoder.pkl")

# 测试新样本分类效果
def predict_product_category(product_name):
    """预测商品类别"""
    inputs = tokenizer(product_name, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        confidence = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()
    predicted_label = lbl.inverse_transform([predicted_class_id])[0]
    return predicted_label, confidence

# 测试新的商品样本
print("\n商品分类测试:")
test_products = [
    "蒙牛纯牛奶250ml",
    "可口可乐330ml",
    "西凤酒55度"
]

print(f"{'商品名称':<25} {'预测类别':<15} {'置信度':<10}")
print("-" * 55)

for product in test_products:
    predicted_label, confidence = predict_product_category(product)
    print(f"{product:<25} {predicted_label:<15} {confidence:.4f}")
