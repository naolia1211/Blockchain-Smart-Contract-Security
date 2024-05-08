from transformers import BertTokenizer, BertModel
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import os

def load_filtered_tokens(file_path):
    filtered_token_lists = []
    with open(file_path, 'r') as file:
        for line in file:
            filtered_tokens = eval(line.strip())
            filtered_token_lists.append(filtered_tokens)
    return filtered_token_lists

def preprocess_tokens(filtered_tokens, tokenizer, max_length):
    # Tokenize bằng tokenizer của BERT
    tokens = tokenizer.tokenize(' '.join(filtered_tokens))
    
    # Cắt ngắn hoặc đệm token_type_ids và attention_mask
    token_type_ids = [0] * len(tokens)
    attention_mask = [1] * len(tokens)
    
    # Đảm bảo độ dài không vượt quá max_length
    if len(tokens) > max_length - 2:
        tokens = tokens[:max_length - 2]
        token_type_ids = token_type_ids[:max_length - 2]
        attention_mask = attention_mask[:max_length - 2]
    
    # Thêm các token đặc biệt vào đầu và cuối chuỗi
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_type_ids = [0] + token_type_ids + [0]
    attention_mask = [1] + attention_mask + [1]
    
    # Chuyển đổi các token thành dạng số (token IDs)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    return input_ids, token_type_ids, attention_mask

def collate_fn(batch):
    input_ids, token_type_ids, attention_mask = zip(*batch)
    input_ids = pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=0)
    token_type_ids = pad_sequence([torch.tensor(ids) for ids in token_type_ids], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([torch.tensor(mask) for mask in attention_mask], batch_first=True, padding_value=0)
    return input_ids, token_type_ids, attention_mask

# Khởi tạo tokenizer và model của BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Đường dẫn đến file output đã tokenize
file_path = r'C:\Users\Admin\Documents\GitHub\Blockchain-Smart-Contract-Security\RunPy\filtered_tokens.txt'

# Độ dài tối đa của chuỗi token
max_length = 128

# Load các token đã lọc từ file
filtered_token_lists = load_filtered_tokens(file_path)

# Preprocess các danh sách token
preprocessed_data = [preprocess_tokens(filtered_tokens, tokenizer, max_length) for filtered_tokens in filtered_token_lists]

# Tạo DataLoader
batch_size = 32
data_loader = DataLoader(preprocessed_data, batch_size=batch_size, collate_fn=collate_fn)

# Kiểm tra GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Chuyển đổi và lưu các vector
vectors = []
with torch.no_grad():
    for batch in data_loader:
        input_ids, token_type_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        batch_vectors = last_hidden_state[:, 0, :].cpu().numpy().tolist()
        vectors.extend(batch_vectors)

# Đường dẫn lưu file tensor.txt
output_path = r'C:\Users\Admin\Documents\GitHub\Blockchain-Smart-Contract-Security\RunPy\tensor1.txt'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Lưu các vector vào file tensor.txt
with open(output_path, 'w') as file:
    for vector in vectors:
        file.write(str(vector) + '\n')