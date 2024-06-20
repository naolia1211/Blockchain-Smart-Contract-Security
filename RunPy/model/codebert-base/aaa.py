import re
import torch
import torch.nn as nn
import json
from pymongo import MongoClient
from transformers import RobertaModel, RobertaConfig, TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from datasets import ClassLabel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import RegexpTokenizer
from collections import Counter

CHECKPOINT = "model/codebert-base"
VOCAB_PATH = "model/codebert-base/results/vocab.json"
MAX_VOCAB_SIZE = 100000
FREQUENCY_THRESHOLD = 7

class CustomTokenizerWrapper:
    def __init__(self, max_length=128, padding=True, truncation=True):
        self.tokenizer = RegexpTokenizer(r'\w+|\{|\}|\(|\)|\[|\]|\;|\=|\+|\-|\*|\/|\!|\%|<|>|\||\&|\.|,')
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.vocabulary = {'<pad>': 0, '<unk>': 1}

    def train(self, dataset):
        token_counter = Counter()
        for item in dataset:
            tokens = self.tokenizer.tokenize(item['content'])
            token_counter.update(tokens)

        filtered_tokens = [token for token, count in token_counter.items() if count > FREQUENCY_THRESHOLD]
        new_tokens = filtered_tokens[:MAX_VOCAB_SIZE - len(self.vocabulary)]
        self.vocabulary.update({token: idx + len(self.vocabulary) for idx, token in enumerate(new_tokens)})

    def save_vocabulary(self, filename=VOCAB_PATH):
        with open(filename, 'w') as outfile:
            json.dump(self.vocabulary, outfile)

    def load_vocabulary(self, filename=VOCAB_PATH):
        try:
            with open(filename, "r") as json_file:
                self.vocabulary = json.load(json_file)
        except FileNotFoundError:
            print(f"Error: File {filename} not found! Using default vocabulary.")

    def convert_token_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token in self.vocabulary:
                ids.append(self.vocabulary[token])
            else:
                ids.append(self.vocabulary['<unk>'])
        return ids

    def __call__(self, data, **kwargs):
        content = data['content']
        tokens = self.tokenizer.tokenize(content)

        if self.padding and len(tokens) < self.max_length:
            tokens.extend(['<pad>'] * (self.max_length - len(tokens)))
        if self.truncation and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        input_ids = self.convert_token_to_ids(tokens)
        attention_mask = [1 if token != '<pad>' else 0 for token in tokens]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens,
        }

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
        rnn_output = rnn_output[:, 0, :]

        cnn_output = self.cnn(attention_heads[1].permute(0, 2, 1)).permute(0, 2, 1)
        cnn_output = cnn_output[:, 0, :]

        ff_output = self.feedforward(attention_heads[2])
        ff_output = ff_output[:, 0, :]

        combined_output = torch.cat((rnn_output, cnn_output, ff_output), dim=-1)
        logits = self.dense(combined_output)
        probs = self.softmax(logits)

        return probs

class SaveMetricsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        logs = state.log_history
        with open('training_metrics.json', 'w') as f:
            json.dump(logs, f)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    report = classification_report(labels, preds, zero_division=0)
    return {'accuracy': accuracy, 'report': report}

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['test']
collection = db['test']

# Initialize tokenizer
custom_tokenizer = CustomTokenizerWrapper(max_length=128)

# Retrieve data from MongoDB
data = list(collection.find())

# Train the tokenizer on the dataset
custom_tokenizer.train(data)
custom_tokenizer.save_vocabulary(VOCAB_PATH)

# Split data into train and validation sets
train_data = data[:int(0.8 * len(data))]
val_data = data[int(0.8 * len(data)):]

# Tokenize the datasets
train_tokenized_data = [custom_tokenizer(item) for item in train_data]
val_tokenized_data = [custom_tokenizer(item) for item in val_data]

train_labels = [item['label'] for item in train_data]
val_labels = [item['label'] for item in val_data]

train_dataset = CustomDataset(train_tokenized_data, train_labels)
val_dataset = CustomDataset(val_tokenized_data, val_labels)

config = RobertaConfig.from_pretrained(CHECKPOINT)
model_name = CHECKPOINT
num_labels = 4
model = ThreatDetectionModel(model_name, num_labels=num_labels)

# Resize the model's token embeddings to match the custom tokenizer's vocabulary size
model.bert.resize_token_embeddings(len(custom_tokenizer.vocabulary))

# Set device to use all available GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
model.to(device)

training_args = TrainingArguments(
    output_dir="D:\\DLResearches\\giang\\Model\\results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    callbacks=[SaveMetricsCallback]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("model/codebert-base/results")
custom_tokenizer.save_vocabulary("model/codebert-base/results/vocab.json")