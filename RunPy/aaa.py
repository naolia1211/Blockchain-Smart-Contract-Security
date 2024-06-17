# ThreatDetectionModel.py
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, Trainer, BertModel
from transformers.trainer_callback import TrainerCallback
import torch
import torch.nn as nn
import json
from torch_geometric.nn import GCNConv

# Download necessary NLTK data
nltk.download('punkt')

# Define a custom tokenizer using RegexpTokenizer
class CustomTokenizerWrapper:
    def __init__(self, pattern=r'\w+|[^\w\s]+'):
        self.tokenizer = RegexpTokenizer(pattern)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def __call__(self, text, **kwargs):
        tokens = self.tokenize(text)
        return {
            'input_ids': list(range(len(tokens))),  # Dummy input_ids for illustration
            'attention_mask': [1] * len(tokens),
        }

# Load dataset
data = pd.read_csv('http_attacks.csv')
dataset = Dataset.from_pandas(data)

# Initialize custom tokenizer
custom_tokenizer = CustomTokenizerWrapper()

# Save custom tokenizer configuration
tokenizer_file = "custom_tokenizer.json"
with open(tokenizer_file, "w") as f:
    json.dump({"tokenizer": "custom"}, f)

# Load custom tokenizer with PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, tokenizer_object=custom_tokenizer)

# Tokenize the dataset and save to file
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.save_to_disk("tokenized_dataset")

# Define training callback for saving metrics
class SaveMetricsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        logs = state.log_history
        with open('training_metrics.json', 'w') as f:
            json.dump(logs, f)

# Load pre-trained model
model_name = "microsoft/codebert-base"  # Change to any pre-trained model like "microsoft/codebert-base" or others
base_model = BertModel.from_pretrained(model_name)

# Custom model for smart contract detection/classification
class SmartContractDetectionModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(SmartContractDetectionModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.num_labels = num_labels

        # Divide BERT layers into groups and add custom layers
        self.group1 = nn.ModuleList(self.bert.encoder.layer[:4])
        self.group2 = nn.ModuleList(self.bert.encoder.layer[4:8])
        self.group3 = nn.ModuleList(self.bert.encoder.layer[8:12])

        self.rnn = nn.LSTM(self.config.hidden_size, self.config.hidden_size, batch_first=True)
        self.cnn = nn.Conv1d(self.config.hidden_size, self.config.hidden_size, kernel_size=3, padding=1)

        # GNN layers for graph-based features
        self.gnn1 = GCNConv(self.config.hidden_size, self.config.hidden_size)
        self.gnn2 = GCNConv(self.config.hidden_size, self.config.hidden_size)

        # Classification layer
        self.classifier = nn.Linear(self.config.hidden_size * 3, num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, edge_index=None, node_features=None):
        # Pass through embedding layers
        embedding_output = self.bert.embeddings(input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids)
        encoder_outputs = embedding_output

        # Group 1 with RNN
        for layer in self.group1:
            encoder_outputs = layer(encoder_outputs, attention_mask=attention_mask)[0]
        rnn_output, _ = self.rnn(encoder_outputs)

        # Group 2 with CNN
        encoder_outputs = rnn_output
        for layer in self.group2:
            encoder_outputs = layer(encoder_outputs, attention_mask=attention_mask)[0]
        cnn_output = self.cnn(encoder_outputs.permute(0, 2, 1)).permute(0, 2, 1)

        # Group 3
        encoder_outputs = cnn_output
        for layer in self.group3:
            encoder_outputs = layer(encoder_outputs, attention_mask=attention_mask)[0]

        # GNN layers for graph-based features
        gnn_output1 = self.gnn1(node_features, edge_index)
        gnn_output2 = self.gnn2(gnn_output1, edge_index)

        # Combine outputs
        combined_output = torch.cat((rnn_output[:, 0, :], cnn_output[:, 0, :], gnn_output2), dim=-1)

        # Classification
        logits = self.classifier(combined_output)
        probs = self.softmax(logits)

        return probs


# Example usage for smart contract detection
num_labels_smart_contract = 5  # Number of vulnerability types
model_smart_contract = SmartContractDetectionModel(model_name, num_labels_smart_contract)

# Split the dataset
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

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

trainer_smart_contract = Trainer(
    model=model_smart_contract,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[SaveMetricsCallback],
)

trainer_smart_contract.train()

# Save the fine-tuned models and tokenizer
model_smart_contract.save_pretrained("./fine-tuned-smart-contract-detector")
tokenizer.save_pretrained("./fine-tuned-tokenizer")
