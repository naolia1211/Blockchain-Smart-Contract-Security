from pymongo import MongoClient
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import math
import os

# Connect MongoDB
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

# SolidityDataset
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

MAX_LEN = 128
BATCH_SIZE = 16

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

if data:
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = SolidityDataset(train_data, tokenizer, MAX_LEN)
    test_dataset = SolidityDataset(test_data, tokenizer, MAX_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
else:
    print("No data available to split into training and test sets.")

# Model and training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomBertConfig(BertConfig):
    def __init__(self, num_attention_groups, **kwargs):
        super().__init__(**kwargs)
        self.num_attention_groups = num_attention_groups

class CustomBertSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_attention_groups = config.num_attention_groups
        self.group_size = self.num_attention_heads // self.num_attention_groups

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

        self.out = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.pruned_heads = set()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.out(context_layer)

        return output

class CustomBertLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CustomBertSelfAttention(config)
        self.intermediate = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layer_norm(layer_output + attention_output)
        return layer_output

class CustomBertEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        num_layers = config.num_hidden_layers
        rnn_layer = num_layers // 3
        self.layers = torch.nn.ModuleList(
            [CustomBertLayer(config) if i % 3 == 0 else
             (torch.nn.RNN(config.hidden_size, config.hidden_size, batch_first=True) if i % 3 == 1 else
              torch.nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True))
             for i in range(num_layers)]
        )

    def forward(self, hidden_states, attention_mask=None):
        for i, layer_module in enumerate(self.layers):
            if isinstance(layer_module, CustomBertLayer):
                hidden_states = layer_module(hidden_states, attention_mask)
            else:
                hidden_states, _ = layer_module(hidden_states)
        return hidden_states

class CustomBertModelWithCNN(BertPreTrainedModel):
    config_class = CustomBertConfig
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertModel(config).embeddings
        self.encoder = CustomBertEncoder(config)
        self.pooler = BertModel(config).pooler
        
        # Add CNN layers
        self.conv1 = torch.nn.Conv1d(config.hidden_size, 128, kernel_size=3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Adjusted classifier input size
        cnn_output_dim = 64 * (config.max_position_embeddings // 4)  # 64 channels, seq_len / 4
        self.classifier = torch.nn.Linear(cnn_output_dim, config.num_labels)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        embedding_output = self.embeddings(input_ids=input_ids)
        encoder_output = self.encoder(embedding_output, attention_mask=attention_mask)
        
        # Transpose the output to have the channel dimension correct
        cnn_input = encoder_output.transpose(1, 2)  # Shape: (batch_size, hidden_size, seq_length)
        
        # Apply CNN layers
        cnn_output = self.conv1(cnn_input)
        cnn_output = self.relu1(cnn_output)
        cnn_output = self.pool1(cnn_output)
        cnn_output = self.conv2(cnn_output)
        cnn_output = self.relu2(cnn_output)
        cnn_output = self.pool2(cnn_output)

        # Flatten and classify
        flattened_output = cnn_output.view(cnn_output.size(0), -1)
        logits = self.classifier(flattened_output)

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs = (loss,) + outputs

        return outputs

config = CustomBertConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.2,
    max_position_embeddings=512,
    initializer_range=0.02,
    num_labels=len(vulnerabilities),  # Number of vulnerabilities to classify
    num_attention_groups=4
)

model = CustomBertModelWithCNN.from_pretrained('bert-base-uncased', config=config)
model = model.to(device)

EPOCHS = 15
LEARNING_RATE = 2e-4
EPSILON = 1e-7

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# train_epoch and eval_model
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({"Loss": loss.item()})
    return total_loss / len(dataloader)

def eval_model(model, dataloader, device):
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
            loss = outputs[0]
            logits = outputs[1]
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
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

# Directory to save the model and tokenizer
output_dir = './model_save/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Training
training_stats = []
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_dataloader, optimizer, device)
    scheduler.step()
    val_loss, val_acc, _, _ = eval_model(model, test_dataloader, device)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    
    # Save model, tokenizer, and training stats after each epoch
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    training_stats.append({
        'epoch': epoch + 1,
        'Training Loss': train_loss,
        'Validation Loss': val_loss,
        'Validation Accuracy': val_acc.item()
    })
    torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.bin"))

# Testing
test_loss, test_acc, test_labels, test_preds = eval_model(model, test_dataloader, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Evaluate model
print("\nEvaluation Metrics:")
evaluate_model(test_labels, test_preds)

# Save final model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save training statistics
import json
with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
    json.dump(training_stats, f, indent=4)
