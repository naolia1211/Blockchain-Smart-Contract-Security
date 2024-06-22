import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaConfig, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from collections import Counter
# Đường dẫn
JSON_FILE_PATH = r"/content/drive/MyDrive/model/codebert-base/Data.Data.json"
VOCAB_PATH = r"/content/drive/MyDrive/model/codebert-base/vocab.json"
TOKEN_FREQ_PATH = r"/content/drive/MyDrive/model/codebert-base/token_freq.json"
RESULTS_DIR = r"/content/drive/MyDrive/model/codebert-base/results"

# Các hằng số
MAX_VOCAB_SIZE = 50000
FREQUENCY_THRESHOLD = 3
MAX_LENGTH = 512
EPOCH = 201
LABEL_MAPPING = {
    "Authentication_and_Authorization_Vulnerabilities": 0,
    "Dependency_vulnerabilities": 1,
    "Interaction_and_constract_state_vulnerabilities": 2,
    "Resource_(Gas)_usage_vulnerabilities": 3,
    "Clean": 4
}
INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
class CustomTokenizerWrapper:
    def __init__(self, padding=True, truncation=True):
        self.tokenizer = RegexpTokenizer(r'\w+|\{|\}|\(|\)|\[|\]|\;|\=|\+|\-|\*|\/|\!|\%|<|>|\||\&|\.|,')
        self.truncation = truncation
        self.padding = padding
        self.vocabulary = {'<pad>': 0, '<unk>': 1, 'null': 2}
        self.features = [
            'external_call_function', 'state_changes_after_external_calls', 'error_handling',
            'fallback_function_interaction', 'ignored_return_value', 'state_variables_manipulation',
            'recursive_calls', 'use_of_msg_value_or_sender', 'use_of_delegatecall', 'library_safe_practices',
            'input_and_parameter_validation', 'gas_limitations_in_fallback_functions', 'usage_of_block_timestamp',
            'usage_of_block_number', 'usage_of_block_blockhash', 'miner_manipulation', 'usage_in_comparison_operations',
            'lack_of_private_seed_storage', 'historical_block_data', 'predictability_of_randomness_sources',
            'insufficient_input_and_parameter_validation', 'owner_variable', 'modifier_usage', 'function_visibility',
            'authentication_checks', 'state_changing_external_calls', 'selfdestruct_usage',
            'authorization_checks_using_tx_origin', 'dependency_on_function_order', 'high_gas_price_transactions',
            'require_statements', 'concurrent_function_invocations', 'gas_limit_checks',
            'state_changing_operations', 'recursive_function_calls', 'high_complexity_loops'
        ]
        self.max_length = MAX_LENGTH

    def train(self, data):
        token_counter = Counter()
        for item in data:
            for feature in self.features:
                if feature in item['functions_with_issues']:
                    for function in item['functions_with_issues'][feature]:
                        if 'function_content' in function:
                            tokens = self.tokenizer.tokenize(function['function_content'])
                            token_counter.update(tokens)

        with open(TOKEN_FREQ_PATH, 'w') as f:
            json.dump(token_counter, f)

        sorted_tokens = sorted(token_counter.items(), key=lambda item: item[1], reverse=True)
        cumulative_count = 0
        freq_threshold = 0

        for token, count in sorted_tokens:
            cumulative_count += 1
            if cumulative_count > MAX_VOCAB_SIZE - len(self.vocabulary):
                freq_threshold = count
                break

        filtered_tokens = [token for token, count in sorted_tokens if count >= freq_threshold]
        new_tokens = filtered_tokens[:MAX_VOCAB_SIZE - len(self.vocabulary)]
        self.vocabulary.update({token: idx + len(self.vocabulary) for idx, token in enumerate(new_tokens)})

        print(f"Selected frequency threshold: {freq_threshold}")

    def tokenize_and_pad(self, tokens):
        if self.truncation and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        if self.padding and len(tokens) < self.max_length:
            tokens.extend(['<pad>'] * (self.max_length - len(tokens)))
        return tokens

    def convert_token_to_ids(self, tokens):
        return [self.vocabulary.get(token, self.vocabulary['<unk>']) for token in tokens]

    def __call__(self, data):
        all_tokens = []
        for feature in self.features:
            all_tokens.append(feature)
            if feature in data['functions_with_issues']:
                function = data['functions_with_issues'][feature][0] if data['functions_with_issues'][feature] else None
                if function and 'function_content' in function:
                    tokens = self.tokenizer.tokenize(function['function_content'])
                    all_tokens.extend(tokens)
                else:
                    all_tokens.append('null')
            else:
                all_tokens.append('null')

        all_tokens = self.tokenize_and_pad(all_tokens)
        input_ids = self.convert_token_to_ids(all_tokens)
        attention_mask = [1 if token != '<pad>' else 0 for token in all_tokens]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': LABEL_MAPPING[data['label']]
        }

    def __len__(self):
        return len(self.vocabulary)

    def save_vocabulary(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.vocabulary, outfile)

    def load_vocabulary(self, filename):
        try:
            with open(filename, "r") as json_file:
                self.vocabulary = json.load(json_file)
        except FileNotFoundError:
            print(f"Error: File {filename} not found! Using default vocabulary.")

class CustomDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

class ThreatDetectionModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(ThreatDetectionModel, self).__init__()
        self.bert = RobertaModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        self.config = self.bert.config
        self.num_labels = num_labels

        self.attention_pooling = nn.Sequential(
            nn.Linear(self.config.hidden_size, 1),
            nn.Softmax(dim=1)
        )

        self.bilstm = nn.LSTM(self.config.hidden_size // 3, self.config.hidden_size // 6, batch_first=True, bidirectional=True)

        self.self_attention = nn.MultiheadAttention(self.config.hidden_size // 3, num_heads=4, batch_first=True)

        self.feedforward = nn.Sequential(
            nn.Linear(self.config.hidden_size // 3, self.config.hidden_size // 3),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size // 3, self.config.hidden_size // 3)
        )
        self.attention_layer = nn.MultiheadAttention(self.config.hidden_size // 3, num_heads=4, batch_first=True)

        self.dense = nn.Linear((self.config.hidden_size // 3) * 3, num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        encoder_outputs = outputs.last_hidden_state

        attention_weights = self.attention_pooling(encoder_outputs).transpose(1, 2)
        pooled_output = torch.bmm(attention_weights, encoder_outputs).squeeze(1)

        split_size = self.config.hidden_size // 3
        attention_heads = pooled_output.split(split_size, dim=-1)

        rnn_input = attention_heads[0].unsqueeze(1)
        rnn_output, _ = self.bilstm(rnn_input)
        rnn_output = rnn_output[:, 0, :]

        attn_input = attention_heads[1].unsqueeze(1).permute(1, 0, 2)
        attn_output, _ = self.self_attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.permute(1, 0, 2)[:, 0, :]

        ff_input = attention_heads[2].unsqueeze(1).permute(1, 0, 2)
        ff_output = self.feedforward(ff_input)
        attn_ff_output, _ = self.attention_layer(ff_output, ff_output, ff_output)
        attn_ff_output = attn_ff_output.permute(1, 0, 2)[:, 0, :]

        combined_output = torch.cat((rnn_output, attn_output, attn_ff_output), dim=-1)

        logits = self.dense(combined_output)
        probs = self.softmax(logits)

        return probs

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=EPOCH, patience=200):
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': {label: [] for label in LABEL_MAPPING},
        'val_recall': {label: [] for label in LABEL_MAPPING},
        'val_f1': {label: [] for label in LABEL_MAPPING},
        'val_roc_auc': {label: [] for label in LABEL_MAPPING},
    }

    checkpoint_filename = os.path.join(RESULTS_DIR, "latest_checkpoint.pth")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        avg_train_loss = total_loss / len(train_dataloader)
        train_accuracy = correct_predictions / total_predictions

        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, _, _ = evaluate_model(
            model, val_dataloader, criterion, device
        )

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        for i, label in enumerate(LABEL_MAPPING):
            history['val_precision'][label].append(val_precision[i])
            history['val_recall'][label].append(val_recall[i])
            history['val_f1'][label].append(val_f1[i])
            history['val_roc_auc'][label].append(val_roc_auc[i])

        if save_checkpoint(epoch + 1, model, optimizer, scheduler, val_loss, val_accuracy, checkpoint_filename):
            print(f"Checkpoint saved: {checkpoint_filename}")
        else:
            print("Failed to save checkpoint")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))
            print(f"New best model saved at epoch {epoch + 1}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping!")
                break

    return history
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=range(len(LABEL_MAPPING)), zero_division=0)

    roc_auc = []
    for i in range(len(LABEL_MAPPING)):
        binary_labels = [1 if label == i else 0 for label in all_labels]
        binary_probs = [prob[i] for prob in all_probs]
        roc_auc.append(roc_auc_score(binary_labels, binary_probs))

    return avg_loss, accuracy, precision, recall, f1, roc_auc, all_labels, all_preds

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    unique_labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def plot_metrics_per_label(history, metric_name, save_path):
    plt.figure(figsize=(12, 6))
    for label in LABEL_MAPPING:
        plt.plot(history['epoch'], history[metric_name][label], label=label)
    plt.title(f'{metric_name.capitalize()} per Label')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_checkpoint(epoch, model, optimizer, scheduler, val_loss, val_accuracy, filename):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }
    torch.save(state, filename)
    return True

def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def main():
    if torch.cuda.is_available():
        print(f"CUDA is available. Version: {torch.version.cuda}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available.")

    data = load_data_from_json(JSON_FILE_PATH)
    print(f"Loaded data structure: {json.dumps(data, indent=2)[:500]}...")

    custom_tokenizer = CustomTokenizerWrapper()
    custom_tokenizer.train(data)

    max_vocab_index = max(custom_tokenizer.vocabulary.values())
    if max_vocab_index >= MAX_VOCAB_SIZE:
        raise ValueError(f"Maximum vocabulary index {max_vocab_index} exceeds the limit of {MAX_VOCAB_SIZE - 1}")

    custom_tokenizer.save_vocabulary(VOCAB_PATH)

    tokenized_data = [custom_tokenizer(item) for item in data]

    train_data, val_data = train_test_split(tokenized_data, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64)

    config = RobertaConfig.from_pretrained("/content/drive/MyDrive/model/codebert-base/config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ThreatDetectionModel("microsoft/codebert-base", num_labels=len(LABEL_MAPPING))
    model.bert = AutoModel.from_pretrained("microsoft/codebert-base", config=config)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    total_steps = len(train_dataloader) * EPOCH
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    checkpoints = [f for f in os.listdir(RESULTS_DIR) if f.startswith("checkpoint_epoch_")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[2].split(".")[0]))
        checkpoint_path = os.path.join(RESULTS_DIR, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    history = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=EPOCH-start_epoch)

    criterion = nn.CrossEntropyLoss()
    val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, y_true, y_pred = evaluate_model(model, val_dataloader, criterion, device)
    print(f"Final Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    y_true = [INVERSE_LABEL_MAPPING[label] for label in y_true]
    y_pred = [INVERSE_LABEL_MAPPING[pred] for pred in y_pred]

    print(classification_report(y_true, y_pred))

    labels = list(LABEL_MAPPING.keys())
    save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, labels, save_path)
    print("Confusion matrix has been saved to:", save_path)

    # Plot and save metrics for each label
    metrics = ['val_precision', 'val_recall', 'val_f1', 'val_roc_auc']
    for metric in metrics:
        save_path = os.path.join(RESULTS_DIR, f"{metric}_plot.png")
        plot_metrics_per_label(history, metric, save_path)
        print(f"{metric.capitalize()} plot has been saved to:", save_path)

    # Plot validation accuracy and loss
    plt.figure(figsize=(12, 6))
    plt.plot(history['epoch'], history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend(loc='best')
    plt.savefig(os.path.join(RESULTS_DIR, "validation_metrics_plot.png"))
    plt.close()
    print("Validation metrics plot has been saved.")

if __name__ == "__main__":
    main()