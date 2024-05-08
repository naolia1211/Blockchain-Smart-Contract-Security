from transformers import BertTokenizer, BertModel
import torch

def load_filtered_tokens(file_path):
    filtered_token_lists = []
    with open(file_path, 'r') as file:
        for line in file:
            filtered_tokens = eval(line.strip())
            filtered_token_lists.append(filtered_tokens)
    return filtered_token_lists

def preprocess_tokens(filtered_tokens, max_length):
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
    
    # Đệm thêm các giá trị 0 cho attention_mask và token_type_ids
    padding_length = max_length - len(tokens)
    token_type_ids = token_type_ids + [0] * padding_length
    attention_mask = attention_mask + [0] * padding_length
    
    # Chuyển đổi các token thành dạng số (token IDs)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Đệm thêm các giá trị 0 cho input_ids
    input_ids = input_ids + [0] * padding_length
    
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

# Khởi tạo list để lưu trữ các vector
vectors = []

# Preprocess và chuyển đổi từng danh sách token thành vector
for filtered_tokens in filtered_token_lists:
    input_ids, token_type_ids, attention_mask = preprocess_tokens(filtered_tokens, max_length)
    
    # Chuyển đổi thành tensor
    input_ids = torch.tensor([input_ids])
    token_type_ids = torch.tensor([token_type_ids])
    attention_mask = torch.tensor([attention_mask])
    
    # Đưa vào mô hình BERT
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        vector = last_hidden_state[0][0].tolist()  # Lấy vector của token [CLS]
    
    vectors.append(vector)

# In ra danh sách các vector
for vector in vectors:
    print(vector)