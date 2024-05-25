import torch
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Định nghĩa CustomBERTModel
class CustomBERTModel(torch.nn.Module):
    def __init__(self, config, num_classes):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel(config, output_attentions=True)
        self.classifier = torch.nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return outputs.attentions, self.classifier(pooled_output)

# Chuẩn bị dữ liệu
data = pd.read_csv('solidity_vulnerabilities.csv')

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenize dữ liệu
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SolidityDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

MAX_LEN = 128
BATCH_SIZE = 16

train_dataset = SolidityDataset(train_data, tokenizer, MAX_LEN)
test_dataset = SolidityDataset(test_data, tokenizer, MAX_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Thiết lập cấu hình tùy chỉnh cho BERT
config = BertConfig(
    hidden_size=768,            # Kích thước ẩn
    num_attention_heads=12,     # Số lượng heads của multi-head attention
    num_hidden_layers=12,       # Số lớp encoder
    intermediate_size=3072,     # Kích thước layer intermediate (feed-forward layer)
    hidden_dropout_prob=0.1,    # Tỷ lệ dropout
    attention_probs_dropout_prob=0.1,
)

# Số lớp phân loại
NUM_CLASSES = 4

model = CustomBERTModel(config, NUM_CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Các siêu tham số
EPOCHS = 3
LEARNING_RATE = 2e-5
EPSILON = 1e-8

# Optimizer và scheduler
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Hàm huấn luyện mô hình
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_attentions = []
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        attentions, outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        all_attentions.append(attentions)
    avg_attention_weights = [torch.mean(torch.stack([batch_att[layer] for batch_att in all_attentions]), dim=0).cpu().numpy() for layer in range(len(all_attentions[0]))]
    return total_loss / len(dataloader), avg_attention_weights

# Hàm đánh giá mô hình
def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_attentions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            attentions, outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            all_attentions.append(attentions)
    avg_attention_weights = [torch.mean(torch.stack([batch_att[layer] for batch_att in all_attentions]), dim=0).cpu().numpy() for layer in range(len(all_attentions[0]))]
    return total_loss / len(dataloader), correct_predictions.double() / len(dataloader.dataset), avg_attention_weights

# Huấn luyện mô hình và lưu trữ trọng số attention
train_attention_weights = []
val_attention_weights = []

for epoch in range(EPOCHS):
    train_loss, train_attentions = train_epoch(model, train_dataloader, criterion, optimizer, device)
    val_loss, val_acc, val_attentions = eval_model(model, test_dataloader, criterion, device)
    scheduler.step()
    train_attention_weights.append(train_attentions)
    val_attention_weights.append(val_attentions)
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print(f'Train Loss: {train_loss}')
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_acc}')

# Hàm trực quan hóa attention weights
def plot_avg_attention(avg_attention_weights, epoch, layer, head):
    attention = avg_attention_weights[epoch][layer][head]
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(attention, cmap='viridis')
    fig.colorbar(cax)
    plt.title(f'Epoch {epoch + 1} - Layer {layer + 1} - Head {head + 1}')
    plt.show()

# Ví dụ trực quan hóa
plot_avg_attention(train_attention_weights, epoch=0, layer=0, head=0)
