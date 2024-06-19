import re, os, binascii, base64, json, urllib
import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
from random_string_detector import RandomStringDetector
from transformers import RobertaModel, RobertaConfig, AutoTokenizer, TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from datasets import ClassLabel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import RegexpTokenizer

MAX_VOCAB_SIZE = 100000  # Updated maximum vocabulary size
VOCAB_PATH = "D:\\DLResearches\\giang\\Model\\vocabulary\\vocab.json"
DEBUG_PATH = "D:\\DLResearches\\giang\\Model\\debug_info.json"
CHECKPOINT = "D:\\DLResearches\\Pretrains\\codebert-base"
FREQUENCY_THRESHOLD = 7  # Define a frequency threshold

class CustomTokenizerWrapper:
    def __init__(self, pattern=r'\w+|[^\w\s]+', max_length=128, padding=True, truncation=True):
        self.tokenizer = RegexpTokenizer(pattern)
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.detector = RandomStringDetector()
        self.vocabulary = {'<pad>': 0, '<unk>': 1}
        self.debug_data = []

    def train(self, dataset):
        token_counter = Counter()
        for _, row in dataset.iterrows():
            combined_data = (
                self.tokenize_method(row['method']) +
                self.tokenize_host(row['host']) +
                self.tokenize_path(row['path']) +
                self.tokenize_query_param(row['query_param']) +
                self.tokenize_user_agent(row['user-agent']) +
                self.tokenize_connection(row['connection']) +
                self.tokenize_accept(row['accept']) +
                self.tokenize_accept_language(row['accept-language']) +
                self.tokenize_cookie(row['cookie']) +
                self.tokenize_cache_control(row['cache-control']) +
                self.tokenize_sec_fetch_site(row['sec-fetch-site']) +
                self.tokenize_sec_fetch_mode(row['sec-fetch-mode']) +
                self.tokenize_sec_fetch_user(row['sec-fetch-user']) +
                self.tokenize_sec_fetch_dest(row['sec-fetch-dest']) +
                self.tokenize_referer(row['referer']) +
                self.tokenize_origin(row['origin']) +
                self.tokenize_body(row['body'])
            )
            token_counter.update(combined_data)

        # Filter tokens by frequency threshold
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
            print(f"Error: File {filename} not found! Using default vocabulary from pre-trained tokenizer.")
            self.load_default_vocabulary()

    def load_default_vocabulary(self):
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        self.vocabulary = tokenizer.get_vocab()

    def tokenize_method(self, method):
        return ['method', str(method)]

    def tokenize_host(self, host):
        return ["host"] + re.split(r'[.:]', str(host))

    def tokenize_user_agent(self, user_agent):
        tokens = ["user-agent"]
        user_agent_tokens = re.split(r'[\/;() ]+', str(user_agent))
        tokens.extend([token for token in user_agent_tokens if token])
        return tokens

    def tokenize_connection(self, connection):
        return ["connection"] + re.split(r'\s', str(connection))

    def tokenize_accept(self, accept):
        tokens = ["accept"] + re.split(r'[,\s]+', str(accept))
        return [token for token in tokens if token]

    def tokenize_accept_language(self, accept_language):
        try:
            return ["accept_language"] + str(accept_language).split('-')
        except Exception as e:
            print(e)
            return []

    def tokenize_cookie(self, cookies):
        tokens = []
        if cookies is None:
            return tokens
        tokens.append('cookie')
        cookies = re.split(';', str(cookies))
        for cookie in cookies:
            cookie = cookie.strip()
            index = cookie.find('=')
            value = cookie[index+1:]
            value = self.decode_url(self.decode_base64_to_hex(self.decode_url(cookie)))
            tokenizer = RegexpTokenizer(r'0?\\?x[\w\d]{2}|%[\w\d]{2}|\w[\w\d]+|[(){}<>+\-*/]')
            tokens_value = tokenizer.tokenize(value)
            tokens += ['cookiename'] + self.replace_random_strings(tokens_value)
        return tokens

    def tokenize_cache_control(self, cache_control):
        try:
            tokens = ["cache-control"] + re.split(r'[,\s]+', str(cache_control))
            return [token for token in tokens if token]
        except Exception as e:
            print(e)
            return []

    def tokenize_sec_fetch_site(self, sec_fetch_site):
        try:
            tokens = ["sec-fetch_site"] + re.split(r'([${}:/.])', str(sec_fetch_site))
            return [token for token in tokens if token]
        except Exception as e:
            print(e)
            return []

    def tokenize_sec_fetch_mode(self, sec_fetch_mode):
        try:
            return ["sec-fetch-mode"] + str(sec_fetch_mode).split('-')
        except Exception as e:
            print(e)
            return []

    def tokenize_sec_fetch_user(self, sec_fetch_user):
        return ['sec-fetch-user', str(sec_fetch_user)]

    def replace_random_strings(self, tokens):
        results = []
        for token in tokens:
            if len(token) > 20:
                results.append('randomstring')
            elif len(token) > 12 and self.detector(token):
                results.append('randomstring')
            else:
                try:
                    num = float(token)
                    if num > 99999:
                        results.append('bignum')
                    else:
                        results.append(token)
                except ValueError:
                    results.append(token)
        return results

    def tokenize_sec_fetch_dest(self, sec_fetch_dest):
        try:
            tokenizer = RegexpTokenizer(r'[\w\d]+|[=()${}?]|&|%[\w|\d]{2}|-')
            tokens = ['sec-fetch-dest'] + tokenizer.tokenize(str(sec_fetch_dest))
            return tokens
        except Exception as e:
            print(e)
            return []

    def tokenize_path(self, path):
        path = self.decode_url(str(path))
        paths = re.split(r'\/', path)
        tokens = ['path']
        tokenizer = RegexpTokenizer(r'%[\w\d]{2}|\.\.|\/|\\|[\w\d]+|[()\.]|-')
        for item in paths:
            list_token = tokenizer.tokenize(item)
            tokens.extend(self.replace_random_strings(list_token))
        return tokens

    def tokenize_query_param(self, query_param):
        try:
            query_param = self.decode_url(str(query_param))
            if query_param is None:
                return []
            params = re.split(r'&|=', query_param)
            tokenizer = RegexpTokenizer(r'[\w\d]+|[=()]|&|%[\w|\d]{2}|-')
            tokens = ['query-param'] + self.replace_random_strings(tokenizer.tokenize(query_param))
            return tokens
        except Exception as e:
            print(e)
            return []

    def tokenize_referer(self, referer):
        try:
            referer = self.decode_url(str(referer))
            tokens = ['referer'] + re.split(r'\/|\?|=', referer)
            return self.replace_random_strings(tokens)
        except Exception as e:
            print(e)
            return []

    def tokenize_origin(self, origin):
        origin = self.decode_url(str(origin))
        try:
            tokens = ["origin"] + re.split(r'[.:]', origin)
            return self.replace_random_strings(tokens)
        except Exception as e:
            print(e)
            return []

    def tokenize_body(self, body):
        body = str(body)
        tokens = []
        if body in [None, ""]:
            return tokens
        tokenizer = RegexpTokenizer(r'[\w\d]+|-|%[\w\d]{2}|0?x[\w\d]{2}|[<>\'"()]')
        if '&' in body:
            for item in body.split('&'):
                name, value = item.split('=')
                tokens.append(name)
                tokens.extend(tokenizer.tokenize(value))
        else:
            if '=' in body:
                name, value = body.split('=')
                tokens.append(name)
                tokens.extend(tokenizer.tokenize(value))
        return self.replace_random_strings(tokens)

    def decode_base64_to_hex(self, s):
        try:
            decoded = base64.b64decode(s)
            try:
                return decoded.decode('utf-8')
            except UnicodeDecodeError:
                return binascii.hexlify(decoded).decode('utf-8')
        except binascii.Error:
            return s

    def decode_url(self, text):
        try:
            while '%' in text:
                decode_text = urllib.parse.unquote(text)
                if decode_text == text:
                    return text
                text = decode_text
        except Exception as e:
            print(e)
        return text

    def convert_token_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token in self.vocabulary:
                ids.append(self.vocabulary[token])
            else:
                ids.append(self.vocabulary['<unk>'])
                self.debug_data.append(f"Token not found in vocabulary: {token}")
        return ids

    def save_debug_info(self, filename=DEBUG_PATH):
        with open(filename, 'w') as outfile:
            json.dump(self.debug_data, outfile, indent=4)

    def __call__(self, row, **kwargs):
        combined_data = (
            self.tokenize_method(row['method']) +
            self.tokenize_host(row['host']) +
            self.tokenize_path(row['path']) +
            self.tokenize_query_param(row['query_param']) +
            self.tokenize_user_agent(row['user-agent']) +
            self.tokenize_connection(row['connection']) +
            self.tokenize_accept(row['accept']) +
            self.tokenize_accept_language(row['accept-language']) +
            self.tokenize_cookie(row['cookie']) +
            self.tokenize_cache_control(row['cache-control']) +
            self.tokenize_sec_fetch_site(row['sec-fetch-site']) +
            self.tokenize_sec_fetch_mode(row['sec-fetch-mode']) +
            self.tokenize_sec_fetch_user(row['sec-fetch-user']) +
            self.tokenize_sec_fetch_dest(row['sec-fetch-dest']) +
            self.tokenize_referer(row['referer']) +
            self.tokenize_origin(row['origin']) +
            self.tokenize_body(row['body'])
        )
        if self.padding and len(combined_data) < self.max_length:
            combined_data.extend(['<pad>'] * (self.max_length - len(combined_data)))
        if self.truncation and len(combined_data) > self.max_length:
            combined_data = combined_data[:self.max_length]
        input_ids = self.convert_token_to_ids(combined_data)
        attention_mask = [1 if token != '<pad>' else 0 for token in combined_data]

        # Save debug information
        debug_info = {
            "combined_data_tokens": combined_data,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vocab_size": len(self.vocabulary)  # Log vocabulary size
        }
        self.debug_data.append(debug_info)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': combined_data,
        }

    def __len__(self):
        return len(self.vocabulary)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, labels):
        self.tokenized_data = tokenized_data
        self.labels = labels

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        item = self.tokenized_data.iloc[idx]
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

# Initialize tokenizer
custom_tokenizer = CustomTokenizerWrapper(max_length=128)
custom_tokenizer.load_vocabulary()

# Read CSV
train_df = pd.read_csv("D:\\DLResearches\\giang\\Model\\dataset\\dataset3_train.csv")
val_df = pd.read_csv("D:\\DLResearches\\giang\\Model\\dataset\\dataset3_val.csv")

# Analyze the dataset and update vocabulary
print("Analyzing the dataset to update vocabulary...")
custom_tokenizer.train(train_df)
custom_tokenizer.save_vocabulary(VOCAB_PATH)
print(f"Updated vocabulary saved to {VOCAB_PATH}")

# Tokenize the datasets
train_tokenized_data = train_df.apply(custom_tokenizer, axis=1)
val_tokenized_data = val_df.apply(custom_tokenizer, axis=1)

# Save debug information
custom_tokenizer.save_debug_info(DEBUG_PATH)

train_labels = train_df['attack_tag']
val_labels = val_df['attack_tag']
train_dataframe = pd.DataFrame(list(train_tokenized_data))
val_dataframe = pd.DataFrame(list(val_tokenized_data))

def label2int(batch):
    tmp = ClassLabel(num_classes=8, names=['benign', 'cookie injection', 'directory traversal', 'log4j', 'log forging', 'rce', 'sql injection', 'xss'])
    return tmp.str2int(batch)

train_labels = train_labels.map(label2int)
val_labels = val_labels.map(label2int)

train_dataset = CustomDataset(train_dataframe, train_labels)
val_dataset = CustomDataset(val_dataframe, val_labels)

config = RobertaConfig.from_pretrained("D:\\DLResearches\\Pretrains\\codebert-base\\config.json")
model_name = "D:\\DLResearches\\Pretrains\\codebert-base"
num_labels = 8
# Compute maximum token ID across the entire dataset
max_token_id = 0
for dataset in [train_dataset, val_dataset]:
    for item in dataset:
        max_id = item['input_ids'].max().item()
        if max_id > max_token_id:
            max_token_id = max_id

# Ensure model's embedding matrix can handle the largest token ID
vocab_size = max(max_token_id + 1, len(custom_tokenizer))
print(f"Resizing embeddings to vocab_size={vocab_size}")
model = ThreatDetectionModel(model_name, num_labels=num_labels)
model.bert.resize_token_embeddings(vocab_size)

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
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
criterion = nn.CrossEntropyLoss()

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Enable CUDA_LAUNCH_BLOCKING for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

for epoch in range(training_args.num_train_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        try:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            with open(DEBUG_PATH, 'a') as debug_file:
                debug_file.write(f"Error during forward pass: {e}\n")
                debug_file.write(f"input_ids: {input_ids}\n")
                debug_file.write(f"attention_mask: {attention_mask}\n")
                debug_file.write(f"labels: {labels}\n")
            raise e

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} | Training Loss: {avg_train_loss:.4f}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch + 1} - Validation Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, zero_division=0))

model.save_pretrained("D:\\DLResearches\\giang\\Model\\results")
custom_tokenizer.save_vocabulary("D:\\DLResearches\\giang\\Model\\results\\vocab.json")
