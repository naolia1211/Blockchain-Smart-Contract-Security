import torch
from transformers import BertModel, BertConfig
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix


class CustomBERTModel(nn.Module):
    def __init__(self, config, num_classes):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel(config)
        
        # Chia 12 lớp ẩn thành 4 phần, mỗi phần 3 lớp
        self.layer_chunks = nn.ModuleList([nn.ModuleList([self.bert.encoder.layer[i] for i in range(start, start+3)]) for start in range(0, config.num_hidden_layers, 3)])
        
        # Lớp phân loại cuối cùng
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Lấy embedding đầu vào từ BERT
        embedding_output = self.bert.embeddings(input_ids)
        
        all_layer_outputs = []
        
        for layer_chunk in self.layer_chunks:
            for layer_module in layer_chunk:
                embedding_output = layer_module(embedding_output, attention_mask)[0]
            all_layer_outputs.append(embedding_output[:, 0, :])  # Lấy đầu ra của token [CLS] cho mỗi chunk
        
        # Kết hợp đầu ra của các phần
        combined_output = torch.cat(all_layer_outputs, dim=-1)
        
        # Phân loại đầu ra
        logits = self.classifier(combined_output)
        
        return logits

# Thiết lập cấu hình tùy chỉnh cho BERT
config = BertConfig(
    hidden_size=768,
    num_attention_heads=4,
    num_hidden_layers=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
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
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Hàm huấn luyện mô hình
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({"Loss": loss.item()})
    return total_loss / len(dataloader)

# Hàm đánh giá mô hình
def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_labels = []
    all_preds = []
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            progress_bar.set_postfix({"Loss": loss.item(), "Accuracy": correct_predictions.item() / len(dataloader.dataset)})
    return total_loss / len(dataloader), correct_predictions.double() / len(dataloader.dataset), all_labels, all_preds

# Huấn luyện mô hình
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = eval_model(model, test_dataloader, criterion, device)
    scheduler.step()
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Đánh giá mô hình trên tập kiểm tra và vẽ confusion matrix
test_loss, test_acc, test_labels, test_preds = eval_model(model, test_dataloader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(set(test_labels)))
plt.xticks(tick_marks, set(test_labels), rotation=45)
plt.yticks(tick_marks, set(test_labels))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
