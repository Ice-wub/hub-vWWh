本次作业主要实现BERT的Fine-Tune , 参考论文对应的github仓库为[Link](https://github.com/xuyige/BERT4doc-Classification)

修改后的代码见[HW04.py](https://github.com/Birchove/ai_learning/blob/main/%E7%8E%8B%E5%85%86%E7%82%AB/week04/HW04.py)

采用的数据集是TNEWS , 今日头条的一些title分类

demo输出示例 :

```bash
Text: 昨天国家队在比赛中表现出色夺得冠军
Pred label id: 3, confidence: 0.8964991569519043

Text: 如何优化深度学习模型的训练速度？
Pred label id: 8, confidence: 0.7621596455574036
```

代码实现了论文中提到的 "Single-Task Fine-Tuning" (单任务微调) 策略 , 即直接在预训练好的 BERT 模型后接一个分类层，使用目标领域的数据（这里是 TNEWS 新闻分类）进行端到端的训练。

代码解析 :
+ 序列截断 (Truncation)： 代码中设置 max_length=64 和 truncation=True。这是为了适应 BERT 的输入限制（最长 512）。
+ 论文中专门探讨了处理长文本的方法（如截断头部、尾部或分层处理） 。代码中直接采用了直接截断策略（保留头部），这在处理像 TNEWS 这样的短文本新闻标题时是非常高效的。

```python
training_args = TrainingArguments(
    learning_rate=LR,  # LR = 2e-5
    num_train_epochs=EPOCHS, # EPOCHS = 3
    # ...
)
```
+ 灾难性遗忘 (Catastrophic Forgetting)： 论文通过实验指出，较高的学习率（如 4e-4）会导致模型无法收敛，而较低的学习率（如 2e-5）对于克服“灾难性遗忘”问题至关重要 。代码直接采用了论文推荐的最佳实践。

```python
trainer = Trainer(...)
trainer.train()
# ...
load_best_model_at_end=True
```
+ 联合微调 (Joint Fine-Tuning)： trainer.train() 会同时更新 BERT 的预训练参数和新添加的分类层参数 $W$ 。
+ 防止过拟合： 代码中使用了 load_best_model_at_end=True，这意味着在验证集上表现最好的模型会被保存下来。论文中也提到了根据验证集保存最佳模型用于测试 。
