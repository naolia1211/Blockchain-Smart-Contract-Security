import os
import json
import torch
from nltk.tokenize import RegexpTokenizer
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

# Import the tokenizer classes
from extractv8 import process_directory
from tockenizerv8 import tokenize_dataset

VOCAB_PATH = 'vocab.json'
TEMP_VOCAB_PATH = 'tempvocab.json'
DEBUG_PATH = 'debug.json'
MAX_VOCAB_SIZE = 50000

LABEL_MAPPING = {
    "Authentication_and_Authorization_Vulnerabilities": 0,
    "Dependency_vulnerabilities": 1,
    "Interaction_and_constract_state_vulnerabilities": 2,
    "Resource_(Gas)_usage_vulnerabilities": 3,
    "arithmetic" :4,
    "Clean": 5
}
INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
RESULTS_DIR = r'D:\GitHub\Blockchain-Smart-Contract-Security\results'
EPOCH = 30

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(filename=os.path.join(RESULTS_DIR, 'training.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ThreatDetectionModel(nn.Module):
    def __init__(self, config, num_labels):
        super(ThreatDetectionModel, self).__init__()
        self.config = config
        self.bert = RobertaModel(self.config)

        self.num_labels = num_labels
        self.bilstm = nn.LSTM(config.hidden_size // 3, config.hidden_size // 6, batch_first=True, bidirectional=True)

        self.cnn = nn.Conv1d(config.hidden_size // 3, config.hidden_size // 3, kernel_size=3, padding=1)
        self.self_attention = nn.MultiheadAttention(config.hidden_size // 3, num_heads=4, batch_first=True)

        self.feedforward = nn.Sequential(
            nn.Linear(config.hidden_size // 3, config.hidden_size // 3),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 3, config.hidden_size // 3)
        )

        self.dense = nn.Linear(config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Forward pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        encoder_outputs = outputs.last_hidden_state

        # Split the BERT output into 3 equal parts
        split_size = self.config.hidden_size // 3
        attention_heads = encoder_outputs.split(split_size, dim=-1)

        # Group 1: first head through BiLSTM
        rnn_input = attention_heads[0]
    
        rnn_output, _ = self.bilstm(rnn_input)
        rnn_output = rnn_output[:, -1, :]
        

        # Group 2: second head through CNN with self-attention
        cnn_input = attention_heads[1].transpose(1, 2)
        cnn_output = self.cnn(cnn_input).transpose(1, 2)
        attn_output, _ = self.self_attention(cnn_output, cnn_output, cnn_output)
        attn_output = attn_output[:, 0, :]
       

        # Group 3: third head through feedforward layers
        ff_input = attention_heads[2]
        ff_output = self.feedforward(ff_input)
        ff_output = ff_output[:, 0, :]

        # Combine outputs
        combined_output = torch.cat((rnn_output, attn_output, ff_output), dim=-1)
        # Final dense layer and softmax
        logits = self.dense(combined_output)
        probs = self.softmax(logits)

        return probs

class CustomDataset(Dataset):
    def __init__(self, tokenized_data, labels):
        self.tokenized_data = tokenized_data
        self.labels = labels

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=10, patience=200):
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

        logging.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

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
            logging.info(f"Checkpoint saved: {checkpoint_filename}")
        else:
            logging.error("Failed to save checkpoint")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))
            logging.info(f"New best model saved at epoch {epoch + 1}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                logging.info("Early stopping!")
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
        if len(set(binary_labels)) > 1:
            roc_auc.append(roc_auc_score(binary_labels, binary_probs))
        else:
            roc_auc.append(None)

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

class RecustomTokenizerWrapper:
    def __init__(self, vocab_path=VOCAB_PATH):
        self.vocab_path = vocab_path
        self.load_vocabulary()

    def load_vocabulary(self):
        try:
            with open(self.vocab_path, "r") as json_file:
                self.vocabulary = json.load(json_file)
        except FileNotFoundError:
            logging.warning(f"File {self.vocab_path} not found! Using default vocabulary.")
            self.vocabulary = {'<pad>': 0, '<unk>': 1}

    def tokenize(self, text):
        tokenizer = RegexpTokenizer(r'\w+|\{|\}|\(|\)|\[|\]|\;|\=|\+|\-|\*|\/|\!|\%|<|>|\||\&|\.|,|;')
        return tokenizer.tokenize(text)

    def convert_token_to_ids(self, tokens):
        return [self.vocabulary.get(token, self.vocabulary['<unk>']) for token in tokens]

    def __call__(self, row):
        combined_data = []
        for col in row.index:
            if col not in ['File Name', 'Vulnerability Label']:
                combined_data.extend(self.tokenize(str(row[col])))
        return combined_data

def main():
    if torch.cuda.is_available():
        logging.info(f"CUDA is available. Version: {torch.version.cuda}")
        logging.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        logging.info("CUDA is not available.")
    # Parameter for extracting data from dataset
    solidity_files_directory = r"D:\GitHub\Blockchain-Smart-Contract-Security\Dataset"
    csv_output_file = r"D:\GitHub\Blockchain-Smart-Contract-Security\Dataset\Dataset.csv"
    mongo_uri = r"mongodb://localhost:27017/"
    db_name = 'Data4'
    collection_name = 'Data'
    process_directory(solidity_files_directory, csv_output_file, mongo_uri, db_name, collection_name)

    # Parameters for tokenizer output
    input_csv = r"D:\GitHub\Blockchain-Smart-Contract-Security\Dataset\Dataset.csv"
    output_csv = r"D:\GitHub\Blockchain-Smart-Contract-Security\Dataset\tokenized_data.csv"
    max_position_embeddings, _, final_tokenized_df, final_vocab = tokenize_dataset(
        input_csv, output_csv, VOCAB_PATH, TEMP_VOCAB_PATH, DEBUG_PATH)
    final_tokenized_df.to_csv(output_csv.replace('.csv', '_final.csv'), index=False)
    
    # Re-tokenize the dataset from previous processes
    reinput_csv=output_csv.replace('.csv', '_final.csv')
    retokenized_df = pd.read_csv(reinput_csv)
    logging.info(f"Loaded data structure: {retokenized_df.head()}")

    with open(VOCAB_PATH, 'r') as vocab_file:
        vocab = json.load(vocab_file)
    logging.info(f"Loaded vocabulary: {list(vocab.keys())[:10]}...")

    # Re-tokenize the data using the loaded vocabulary
    tokenizer = RecustomTokenizerWrapper(vocab_path=VOCAB_PATH)
    tokenizer.vocabulary = vocab
    tokenized_data = []
    for _, row in retokenized_df.iterrows():
        tokens = tokenizer(row)
        input_ids = tokenizer.convert_token_to_ids(tokens)
        attention_mask = [1] * len(input_ids) + [0] * (max_position_embeddings - len(input_ids))
        input_ids.extend([tokenizer.vocabulary['<pad>']] * (max_position_embeddings - len(input_ids)))
        tokenized_data.append({'input_ids': input_ids[:max_position_embeddings], 'attention_mask': attention_mask[:max_position_embeddings]})

    labels = retokenized_df['Vulnerability Label'].apply(lambda x: LABEL_MAPPING[x]).values
    dataset = CustomDataset(tokenized_data, labels)

    train_data, val_data, train_labels, val_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
    config = RobertaConfig(
        vocab_size=50000,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu_new",
        max_position_embeddings=max_position_embeddings+2,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12
    )

    model = ThreatDetectionModel(config, num_labels=len(LABEL_MAPPING))

    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
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
        logging.info(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    history = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=EPOCH-start_epoch)

    criterion = nn.CrossEntropyLoss()
    val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, y_true, y_pred = evaluate_model(model, val_dataloader, criterion, device)
    logging.info(f"Final Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    y_true = [INVERSE_LABEL_MAPPING[label] for label in y_true]
    y_pred = [INVERSE_LABEL_MAPPING[pred] for pred in y_pred]

    logging.info("\n" + classification_report(y_true, y_pred))

    labels = list(LABEL_MAPPING.keys())
    save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, labels, save_path)
    logging.info("Confusion matrix has been saved to: " + save_path)

    metrics = ['val_precision', 'val_recall', 'val_f1', 'val_roc_auc']
    for metric in metrics:
        save_path = os.path.join(RESULTS_DIR, f"{metric}_plot.png")
        plot_metrics_per_label(history, metric, save_path)
        logging.info(f"{metric.capitalize()} plot has been saved to: " + save_path)

    plt.figure(figsize=(12, 6))
    plt.plot(history['epoch'], history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend(loc='best')
    plt.savefig(os.path.join(RESULTS_DIR, "validation_metrics_plot.png"))
    plt.close()
    logging.info("Validation metrics plot has been saved.")

if __name__ == "__main__":
    main()
