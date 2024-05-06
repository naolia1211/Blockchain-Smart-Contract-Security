from transformers import BertTokenizer, BertModel
import torch

def get_token_embeddings(tokens):
    # Tạo BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Chuyển token thành ID
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids)

    # Kiểm tra hình dạng của input_ids
    print("Hình dạng của input_ids:", input_ids.shape)

    # Nếu input_ids là tensor 1 chiều, thêm một chiều bổ sung để tạo tensor 2 chiều
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)

    # Tạo input mask
    input_mask = torch.ones(input_ids.shape, dtype=torch.long)

    # Chuyển input_ids và input_mask vào thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)

    # Tạo BertModel và nhúng token
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=input_mask)

    # Lấy vector embedding
    token_embeddings = outputs.last_hidden_state
    print(token_embeddings.shape)
    print(token_embeddings)

    return token_embeddings

# Danh sách các token
tokens = ['vulnerable', '##bank', '.', 'withdraw', '##balance', 'function', 'withdraw', '##balance', '(', ')', 'public', '{', 'ui', '##nt', 'amount', '##tow', '##ith', '##dra', '##w', '=', 'user', '##balance', '##s', '[', 'ms', '##g', '.', 'send', '##er', ']', ';', 'if', '(', '!', '(', 'ms', '##g', '.', 'send', '##er', '.', 'call', '.', 'value', '(', 'amount', '##tow', '##ith', '##dra', '##w', ')', '(', ')', ')', ')', '{', 'throw', ';', '}', 'user', '##balance', '##s', '[', 'ms', '##g', '.', 'send', '##er', ']', '=', '<', 'number', '>', ';', '}']

token_embeddings = get_token_embeddings(tokens)