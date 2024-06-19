import re, os, json
import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
from transformers import RobertaModel, RobertaConfig, TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import RegexpTokenizer

MAX_VOCAB_SIZE = 100000
MAX_LENGTH = 52
VOCAB_PATH = "D:\\DLResearches\\giang\\Model\\vocabulary\\vocab.json"
DEBUG_PATH = "D:\\DLResearches\\giang\\Model\\debug_info.json"
MODEL_NAME = "roberta-base"
FREQUENCY_THRESHOLD = 7

class CustomTokenizerWrapper:
    def __init__(self, pattern=r'\w+|\{|\}|\(|\)|\[|\]|\;|\=|\+|\-|\*|\/|\!|\%|<|>|\||\&|\.|\,'):
        self.tokenizer = RegexpTokenizer(pattern)
        self.vocabulary = {}

    def train(self, dataset):
        token_counter = Counter()
        for _, row in dataset.iterrows():
            combined_data = self.tokenize_features(row)
            token_counter.update(combined_data)

        filtered_tokens = [token for token, count in token_counter.items() if count > FREQUENCY_THRESHOLD]
        new_tokens = filtered_tokens[:MAX_VOCAB_SIZE - len(self.vocabulary)]
        self.vocabulary = {token: idx for idx, token in enumerate(new_tokens)}

    def save_vocabulary(self, filename=VOCAB_PATH):
        with open(filename, 'w') as outfile:
            json.dump(self.vocabulary, outfile)

    def load_vocabulary(self, filename=VOCAB_PATH):
        try:
            with open(filename, "r") as json_file:
                self.vocabulary = json.load(json_file)
        except FileNotFoundError:
            print(f"Error: File {filename} not found! Using empty vocabulary.")
            self.vocabulary = {}

    def tokenize_features(self, row):
        tokens = []
        for feature_name in row.index:
            if pd.notna(row[feature_name]):
                feature_tokens = self.tokenizer.tokenize(str(row[feature_name]))
                tokens.extend(feature_tokens)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        ids = [self.vocabulary.get(token, len(self.vocabulary)) for token in tokens]
        return ids

    def __call__(self, row):
        tokens = self.tokenize_features(row)
        input_ids = self.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        if len(input_ids) < MAX_LENGTH:
            input_ids += [0] * (MAX_LENGTH - len(input_ids))
            attention_mask += [0] * (MAX_LENGTH - len(attention_mask))
        else:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, labels):
        self.tokenized_data = tokenized_data
        self.labels = labels

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class ThreatDetectionModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(ThreatDetectionModel, self).__init__()
        self.bert = RobertaModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        self.config = self.bert.config
        self.num_labels = num_labels

        self.pooling = nn.MaxPool1d(kernel_size=6, stride=6)
        self.rnn = nn.LSTM(self.config.hidden_size // 6, self.config.hidden_size // 6, batch_first=True)
        self.cnn = nn.Conv1d(self.config.hidden_size // 6, self.config.hidden_size // 6, kernel_size=3, padding=1)
        self.feedforward = nn.Sequential(
            nn.Linear(self.config.hidden_size // 6, self.config.hidden_size // 6),
            nn.ReLU()
        )
        self.dense = nn.Linear(self.config.hidden_size // 2, num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        encoder_outputs = outputs.last_hidden_state

        pooled_output = self.pooling(encoder_outputs.permute(0, 2, 1)).permute(0, 2, 1)
        split_size = self.config.hidden_size // 6
        attention_heads = pooled_output.split(split_size, dim=-1)

        rnn_output, _ = self.rnn(attention_heads[0])
        rnn_output = rnn_output[:, 0, :]  # Take the output of the first time step

        cnn_output = self.cnn(attention_heads[1].permute(0, 2, 1)).permute(0, 2, 1)
        cnn_output = cnn_output[:, 0, :]  # Take the output of the first time step

        ff_output = self.feedforward(attention_heads[2])
        ff_output = ff_output[:, 0, :]  # Take the output of the first time step

        combined_output = torch.cat((rnn_output, cnn_output, ff_output), dim=-1)
        logits = self.dense(combined_output)
        probs = self.softmax(logits)

        return probs

class SaveMetricsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        logs = state.log_history
        with open('training_metrics.json', 'w') as f:
            json.dump(logs, f)

# Load datasets
delegatecall_df = pd.read_csv("delegatecall.csv")
reentrancy_df = pd.read_csv("reentrancy.csv")
unchecked_external_call_df = pd.read_csv("unchecked_external_call.csv")
unchecked_send_df = pd.read_csv("unchecked_send.csv")

# Combine datasets into a single DataFrame
combined_df = pd.concat([delegatecall_df, reentrancy_df, unchecked_external_call_df, unchecked_send_df])

# Create a Dataset from the combined DataFrame
dataset = Dataset.from_pandas(combined_df)

# Initialize custom tokenizer
custom_tokenizer = CustomTokenizerWrapper()

# Train the tokenizer on the combined dataset
custom_tokenizer.train(combined_df)

# Save the trained vocabulary
custom_tokenizer.save_vocabulary(VOCAB_PATH)

# Tokenize the dataset
tokenized_data = dataset.map(custom_tokenizer, batched=True, remove_columns=dataset.column_names)

# Create labels
labels = combined_df['label'].tolist()

# Create CustomDataset
custom_dataset = CustomDataset(tokenized_data, labels)

# Split the dataset
train_size = int(0.8 * len(custom_dataset))
train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, len(custom_dataset) - train_size])

# Initialize the model
num_labels = len(combined_df['label'].unique())
model = ThreatDetectionModel(MODEL_NAME, num_labels=num_labels)

# Set device to use all available GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[SaveMetricsCallback],
)

trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine-tuned-vulnerability-detector")
custom_tokenizer.save_vocabulary("./fine-tuned-vulnerability-detector/vocab.json")