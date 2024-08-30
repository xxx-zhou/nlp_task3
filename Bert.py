import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForNextSentencePrediction, AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
# 检查是否有可用的GPU，如果有，使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的BERT模型和分词器
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForNextSentencePrediction.from_pretrained(model_name).to(device)


# SNLI数据集类
class SNLIDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    sentence1, sentence2, label = parts
                    data.append((sentence1, sentence2, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1, sentence2, label = self.data[idx]
        encoding = self.tokenizer(sentence1, sentence2, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        # 将标签转换为数值
        if label == 'entailment':
            num_label = 0
        elif label == 'contradiction':
            num_label = 1
        else:
            num_label = 2
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(num_label, dtype=torch.long)
        }
# 定义训练参数
epochs = 4
batch_size = 16
learning_rate = 2e-5
num_warmup_steps = 0

# 将数据集转换为DataLoader
train_dataset = SNLIDataset('snli_1.0_train.txt', tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
num_training_steps = epochs * len(train_dataloader)

# 优化器和调度器
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                            num_training_steps=num_training_steps)

# 将数据集转换为DataLoader
train_dataset = SNLIDataset('train.tsv', tokenizer)  # 假设你的训练数据为train.tsv
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        inputs, labels = batch
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}")

# 保存模型
model.save_pretrained('snli_bert_model')
