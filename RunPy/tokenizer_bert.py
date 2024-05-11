import json
from pymongo import MongoClient
from transformers import RobertaTokenizer

# Kết nối với MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["your_database_name"]

# Các collection tương ứng với 3 lỗ hổng
collections = ["delegatecall", "reentrancy", "unchecked_external_call"]

# Khởi tạo tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Đường dẫn đến thư mục output
output_dir = "C:\\Users\\Admin\\Documents\\GitHub\\Blockchain-Smart-Contract-Security\\RunPy\\output_tokenize"

for collection_name in collections:
    collection = db[collection_name]
    
    # Lấy tất cả các document trong collection
    documents = collection.find()
    
    # Tokenize và lưu vào file txt
    output_file = f"{output_dir}/{collection_name}_tokens.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in documents:
            code = doc["code"]
            tokens = tokenizer.tokenize(code)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            f.write(json.dumps(token_ids) + "\n")

print("Tokenization completed.") 