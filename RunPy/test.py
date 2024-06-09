from pymongo import MongoClient
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Kết nối tới MongoDB và truy vấn dữ liệu
client = MongoClient('mongodb://localhost:27017/')
db = client['Interaction_and_Contract_State_Vulnerabilities']

vulnerabilities = ['reentrancy', 'delegatecall', 'unchecked_external_call', 'unchecked_send']
labels = {vulnerability: i for i, vulnerability in enumerate(vulnerabilities)}
data = []

def fetch_data_from_collection(collection_name, label):
    collection = db[collection_name]
    documents = collection.find({})
    for doc in documents:
        if 'extract_feature' in doc:
            for feature in doc['extract_feature']:
                tokens = feature.get('tokens', [])
                if tokens:
                    data.append((' '.join(tokens), label))

for vulnerability in vulnerabilities:
    fetch_data_from_collection(vulnerability, labels[vulnerability])

print(f"Number of samples: {len(data)}")

# Định nghĩa lớp SolidityDataset
class SolidityDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Chuẩn bị dữ liệu và dataloader
MAX_LEN = 128
# chạy thống kê xem token dài bao nhiêu để setting lại
# lớn quá thì nhiều padding và training lâu

# thử với CNN, RNN, LSTM để phân loại head (phân loại)


BATCH_SIZE = 16 #tăng lên nếu nhỏ thì không đại diện mẫu. mỗi lần phải 
# bội của tổng số lỗ hổng
# không lấy theo thứ tự 
# 2 cách : radom tại bước training, *(random từ đầu vào)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

if data:
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = SolidityDataset(train_data, tokenizer, MAX_LEN)
    test_dataset = SolidityDataset(test_data, tokenizer, MAX_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
else:
    print("No data available to split into training and test sets.")

# Khởi tạo mô hình và các tham số huấn luyện
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = BertConfig(
    hidden_size=768,
    num_hidden_layers=12, #có thể tăng lên và chia nhóm ra 
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    initializer_range=0.02,
    num_labels=len(vulnerabilities)
)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config = config)
model = model.to(device)

EPOCHS = 13
LEARNING_RATE = 2e-5
EPSILON = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

# Định nghĩa hàm train_epoch và eval_model
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()

        # In ra giá trị gradient của các tham số mô hình
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: {param.grad.norm()}")

        optimizer.step()
        progress_bar.set_postfix({"Loss": loss.item()})
    return total_loss / len(dataloader)

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
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            progress_bar.set_postfix({"Loss": loss.item(), "Accuracy": correct_predictions.item() / len(dataloader.dataset)})
    return total_loss / len(dataloader), correct_predictions.double() / len(dataloader.dataset), all_labels, all_preds

def evaluate_model(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

# Huấn luyện mô hình
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
    scheduler.step()
    val_loss, val_acc, _, _ = eval_model(model, test_dataloader, criterion, device)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Đánh giá mô hình trên tập kiểm tra
test_loss, test_acc, test_labels, test_preds = eval_model(model, test_dataloader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Sử dụng hàm evaluate_model để tính các chỉ số đánh giá
print("\nEvaluation Metrics:")
evaluate_model(test_labels, test_preds)
