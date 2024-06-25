import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaConfig, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
# Paths
TOKEN_FREQ_PATH = r"D:\GitHub\Blockchain-Smart-Contract-Security\model\codebert-base\token_freq.json"
RESULTS_DIR = r"D:\GitHub\Blockchain-Smart-Contract-Security\model\codebert-base\results"
CSV_FILE_PATH = r"D:\GitHub\Blockchain-Smart-Contract-Security\Dataset2\extracted_data.csv"  # Cập nhật đường dẫn này
VOCAB_PATH = r"D:\GitHub\Blockchain-Smart-Contract-Security\model\codebert-base\vocab.json"

# Constants
MAX_VOCAB_SIZE = 17000 # dounle check the size of your vocabulary
FREQUENCY_THRESHOLD = 2
MAX_LENGTH = 512
EPOCH = 30
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
        self.tokenizer = RegexpTokenizer(r'\w+|\{|\}|\(|\)|\[|\]|\;|\=|\+|\-|\*|\/|\!|\%|<|>|\||\&|\.|,|;')
        self.truncation = truncation
        self.padding = padding
        self.vocabulary = {'<pad>': 0, '<unk>': 1, 'null': 2}
        self.max_length = MAX_LENGTH

    def train(self, data):
        token_counter = Counter()
        for item in data:
            for feature, value in item['features'].items():
                token_counter[feature] += 1  # Đảm bảo tên đặc trưng được thêm vào
                tokens = self.tokenizer.tokenize(str(value))
                token_counter.update(tokens)
            for feature in data[0]['features'].keys():
                self.vocabulary[feature] = len(self.vocabulary)

        with open(TOKEN_FREQ_PATH, 'w') as f:
            json.dump(token_counter, f)

        sorted_tokens = sorted(token_counter.items(), key=lambda item: item[1], reverse=True)
        selected_tokens = [token for token, count in sorted_tokens if count >= FREQUENCY_THRESHOLD]
        new_tokens = selected_tokens[:MAX_VOCAB_SIZE - len(self.vocabulary)]
        
        self.vocabulary.update({token: idx + len(self.vocabulary) for idx, token in enumerate(new_tokens)})
        print(f"Selected tokens: {len(new_tokens)}")
        print(f"Final vocabulary size: {len(self.vocabulary)}")

    def tokenize_and_pad(self, tokens):
        if self.truncation and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        if self.padding and len(tokens) < self.max_length:
            tokens.extend(['<pad>'] * (self.max_length - len(tokens)))
        return tokens

    def convert_token_to_ids(self, tokens):
        vocab_size = len(self.vocabulary)
        return [min(self.vocabulary.get(token, self.vocabulary['<unk>']), vocab_size - 1) for token in tokens]

    def __call__(self, data):
        all_tokens = []
        for feature, value in data['features'].items():
            all_tokens.append(feature)
            feature_tokens = self.tokenizer.tokenize(str(value))
            if not feature_tokens:
                feature_tokens = ['nan']
            all_tokens.extend(feature_tokens)

        # Cắt bớt nếu vượt quá max_length
        if len(all_tokens) > self.max_length:
            all_tokens = all_tokens[:self.max_length]
        
        # Padding nếu cần
        if len(all_tokens) < self.max_length:
            all_tokens.extend(['<pad>'] * (self.max_length - len(all_tokens)))

        input_ids = self.convert_token_to_ids(all_tokens)
        attention_mask = [1 if token != '<pad>' else 0 for token in all_tokens]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': LABEL_MAPPING[data['label']]
        }

    def __len__(self):
        return len(self.vocabulary)

    def save_vocabulary(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.vocabulary, f)

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
    def __init__(self, config, num_labels):
        super(ThreatDetectionModel, self).__init__()
        self.bert = RobertaModel(config)
        self.num_labels = num_labels

        # Each group processes 1/3 of hidden_size
        group_size = config.hidden_size // 3

        # RNN for the first group
        self.rnn = nn.LSTM(group_size, 128, batch_first=True)

        # CNN for the second group - Adjust input channels to group_size
        self.cnn = nn.Conv1d(group_size, 128, kernel_size=3, padding=1)

        # BiLSTM for the third group
        self.bilstm = nn.LSTM(group_size, 128, batch_first=True, bidirectional=True)

        # Dense layer to combine outputs
        self.dense = nn.Linear(128 + 128 + 256, config.hidden_size)

        # Final classification layer
        self.final_dense = nn.Linear(config.hidden_size, num_labels)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        encoder_outputs = outputs.last_hidden_state

        # Split hidden_size into three groups
        group_size = encoder_outputs.size(2) // 3
        group1 = encoder_outputs[:, :, :group_size]
        group2 = encoder_outputs[:, :, group_size:2*group_size]
        group3 = encoder_outputs[:, :, 2*group_size:]

        # Process each group
        rnn_output, _ = self.rnn(group1)
        rnn_output = rnn_output[:, 0, :]  # Extracting the hidden state of the first token

        cnn_output = self.cnn(group2.permute(0, 2, 1)).permute(0, 2, 1)
        cnn_output = cnn_output[:, 0, :]  # Extracting the feature map of the first token

        bilstm_output, _ = self.bilstm(group3)
        bilstm_output = bilstm_output[:, 0, :]  # Extracting the hidden state of the first token

        # Combine outputs
        combined_output = torch.cat((rnn_output, cnn_output, bilstm_output), dim=-1)
        dense_output = self.dense(combined_output)

        # Final classification
        logits = self.final_dense(dense_output)
        probs = self.softmax(logits)

        return probs


# Define custom model configuration
vocab_size = 13603  # Define the vocab_size
config = RobertaConfig(
    vocab_size=vocab_size,
    hidden_size=384,  # Hidden size
    num_hidden_layers=12,  # Number of hidden layers
    num_attention_heads=12,  # Number of attention heads
    intermediate_size=3702,  # Intermediate size
    hidden_act="gelu_new",  # Activation function
    max_position_embeddings=512,  # Max position embeddings
    type_vocab_size=1,
    initializer_range=0.02,
    layer_norm_eps=1e-12
)

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

def calculate_roc_auc(y_true, y_score, average='macro', multi_class='ovr'):
    try:
        if len(np.unique(y_true)) > 2:
            # Đa lớp
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            return roc_auc_score(y_true_bin, y_score, average=average, multi_class=multi_class)
        else:
            # Nhị phân
            return roc_auc_score(y_true, y_score[:, 1])  # Giả sử y_score là ma trận xác suất
    except ValueError:
        print("Không thể tính ROC AUC. Có thể chỉ có một lớp trong dữ liệu.")
        return None
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
    roc_auc = calculate_roc_auc(all_labels, all_probs)

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

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    
    feature_columns = [
        'external_call_function', 'state_change_after_external_call', 'error_handling',
        'fallback_function_interaction', 'unchecked_external_call', 'use_of_msg_value_or_sender',
        'delegatecall_with_msg_data', 'dynamic_delegatecall', 'library_safe_practices',
        'gas_limitations_in_fallback_functions', 'balance_check_before_send', 'multiple_external_calls',
        'external_call_in_loop', 'reentrancy_guard_missing', 'low_level_call', 'library_usage_in_delegatecall',
        'state_variables_manipulation', 'state_update_without_verification', 'recursive_calls',
        'context_preservation', 'usage_of_block_timestamp', 'usage_of_block_number', 'usage_of_block_blockhash',
        'miner_manipulation', 'usage_in_comparison_operations', 'block_number_as_time', 'historical_block_data',
        'lack_of_private_seed_storage', 'predictability_of_randomness_sources', 'block_based_random',
        'keccak256_random', 'unsafe_random_generation', 'timestamp_arithmetic', 'timestamp_in_condition',
        'timestamp_assignment', 'timestamp_function_param', 'block_number_arithmetic', 'block_number_in_condition',
        'block_number_assignment', 'block_number_function_param', 'block_based_operations',
        'miner_susceptible_operations', 'input_and_parameter_validation', 'owner_variable', 'modifier_definition',
        'function_visibility', 'authentication_checks', 'state_changing_external_calls', 'selfdestruct_usage',
        'vulnerable_sensitive_data_handling', 'improper_input_padding', 'unchecked_address_variables',
        'constructor_ownership_assignment', 'transaction_state_variable', 'ownership_transfer', 'privileged_operation',
        'unchecked_parameter', 'address_parameter', 'state_change_after_transfer', 'multiple_transfers',
        'price_check_before_action', 'unsafe_type_inference', 'reentrancy_risk', 'unchecked_send_result',
        'assembly_block', 'costly_loop', 'unbounded_operation', 'gas_intensive_function', 'risky_external_call',
        'improper_exception_handling', 'arithmetic_in_loop', 'risky_fallback', 'loop_counter_vulnerability',
        'gas_limit_in_loop', 'arithmetic_operations', 'contract_creation', 'event_logging', 'state_variables',
        'modifiers', 'assertions', 'external_calls', 'ether_transfer', 'fallback_functions', 'dynamic_address_handling',
        'time_based_logic', 'complex_state_changes', 'unchecked_arithmetic', 'unconventional_control_flows',
        'unchecked_return_values'
    ]
    
    data = []
    for _, row in df.iterrows():
        item = {
            'features': {feature: row[feature] for feature in feature_columns},
            'label': row['Label'],
        }
        data.append(item)
    
    global LABEL_MAPPING
    unique_labels = df['Label'].unique()
    LABEL_MAPPING = {label: idx for idx, label in enumerate(unique_labels)}
    
    return data

def test_tokenizer(custom_tokenizer, data, num_tokens=1):
    # Tokenize a few examples
    tokenized_examples = [custom_tokenizer(item) for item in data[:num_tokens]]
    
    # Print the tokenized examples
    for i, example in enumerate(tokenized_examples):
        print(f"Example {i + 1}:")
        
        # Retrieve the original tokens
        original_tokens = custom_tokenizer.tokenizer.tokenize(' '.join([str(value) for value in data[i]['features'].values()]))
        padded_tokens = custom_tokenizer.tokenize_and_pad(original_tokens)
        
        print(f"Original Tokens: {padded_tokens[:10]}...")  # Print only the first 10 tokens for brevity
        print(f"Input IDs: {example['input_ids'][:10]}...")  # Print only the first 10 input IDs for brevity
        print(f"Attention Mask: {example['attention_mask'][:10]}...")  # Print only the first 10 attention mask values
        print(f"Label: {example['label']}\n")
def validate_input_ids(input_ids, vocab_size):
    invalid_ids = [idx for idx in input_ids if idx >= vocab_size]
    if invalid_ids:
        print(f"Invalid input IDs found: {invalid_ids}")
        return False
    return True
def inspect_dataset_input(data, num_samples=1):
    print("\nInspecting data before creating CustomDataset:")
    for i, sample in enumerate(data[:num_samples]):
        print(f"\nSample {i + 1}:")
        print(f"Keys: {sample.keys()}")
        print(f"Input IDs (first 10): {sample['input_ids'][:10]}...")
        print(f"Input IDs length: {len(sample['input_ids'])}")
        print(f"Attention Mask (first 10): {sample['attention_mask'][:10]}...")
        print(f"Attention Mask length: {len(sample['attention_mask'])}")
        print(f"Label: {sample['label']}")

    # Kiểm tra xem tất cả các mẫu có cùng chiều dài không
    input_lengths = [len(sample['input_ids']) for sample in data]
    attention_mask_lengths = [len(sample['attention_mask']) for sample in data]
    
    print(f"\nAll samples have the same input_ids length: {len(set(input_lengths)) == 1}")
    print(f"All samples have the same attention_mask length: {len(set(attention_mask_lengths)) == 1}")
    
    if len(set(input_lengths)) > 1:
        print(f"Input lengths vary. Min: {min(input_lengths)}, Max: {max(input_lengths)}")
    if len(set(attention_mask_lengths)) > 1:
        print(f"Attention mask lengths vary. Min: {min(attention_mask_lengths)}, Max: {max(attention_mask_lengths)}")
def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    if torch.cuda.is_available():
        print(f"CUDA is available. Version: {torch.version.cuda}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available.")

    data = load_data_from_csv(CSV_FILE_PATH)
    print(f"Loaded {len(data)} samples")

    custom_tokenizer = CustomTokenizerWrapper()
    custom_tokenizer.train(data)
    print("Vocabulary size:", len(custom_tokenizer.vocabulary))
    max_vocab_index = max(custom_tokenizer.vocabulary.values())
    if max_vocab_index >= MAX_VOCAB_SIZE:
        raise ValueError(f"Maximum vocabulary index {max_vocab_index} exceeds the limit of {MAX_VOCAB_SIZE - 1}")
    
    custom_tokenizer.save_vocabulary(VOCAB_PATH)

    tokenized_data = [custom_tokenizer(item) for item in data]
    inspect_dataset_input(tokenized_data)

    train_data, val_data = train_test_split(tokenized_data, test_size=0.2, random_state=42)
    print("\nInspecting train data:")
    inspect_dataset_input(train_data)

    # Kiểm tra dữ liệu validation
    print("\nInspecting validation data:")
    inspect_dataset_input(val_data)
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThreatDetectionModel(config, num_labels=len(LABEL_MAPPING))

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    total_steps = len(train_dataloader) * EPOCH
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    history = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=EPOCH)

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

    metrics = ['val_precision', 'val_recall', 'val_f1', 'val_roc_auc']
    for metric in metrics:
        save_path = os.path.join(RESULTS_DIR, f"{metric}_plot.png")
        plot_metrics_per_label(history, metric, save_path)
        print(f"{metric.capitalize()} plot has been saved to:", save_path)

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