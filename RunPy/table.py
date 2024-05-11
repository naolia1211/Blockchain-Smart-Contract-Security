import os
import csv
import json
import textwrap

# Đường dẫn đến thư mục chứa các file TXT
txt_folder = "C:\\Users\\Admin\\Documents\\GitHub\\Blockchain-Smart-Contract-Security\\RunPy\\output_tokenize"

# Đường dẫn đến thư mục lưu file CSV đầu ra
output_folder = "C:\\Users\\Admin\\Documents\\GitHub\\Blockchain-Smart-Contract-Security\\RunPy\\output_table"

# Kết nối đến MongoDB
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
db = client['Interaction_and_Contract_State_Vulnerabilities']

# Danh sách các loại lỗ hổng
vulnerability_types = ['delegatecall', 'reentrancy', 'unchecked_external_call']

# Tạo đường dẫn đầy đủ cho file CSV đầu ra
output_file = os.path.join(output_folder, 'output.csv')

# Tạo file CSV
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Vulnerabilities', 'Smart Contract', 'Feature Extraction', 'Tokenizer']
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL, delimiter=',', lineterminator='\r\n')
    writer.writerow(fieldnames)

    # Duyệt qua từng loại lỗ hổng
    for vulnerability_type in vulnerability_types:
        collection = db[vulnerability_type]
        
        # Đọc nội dung file TXT tương ứng
        txt_file = os.path.join(txt_folder, f"{vulnerability_type}_filtered_tokens.txt")
        tokenizers = []
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8') as file:
                    tokenizers = file.read().strip().split('\n')
            except UnicodeDecodeError:
                print(f"Lỗi khi đọc file {txt_file}. Bỏ qua file này.")
        
        # Lấy dữ liệu từ các file JSON trong MongoDB (giới hạn 10 file)
        docs = list(collection.find().limit(10))
        
        # Ghi dữ liệu vào file CSV
        for i in range(10):
            if i < len(docs):
                filename = docs[i].get('filename', '')
                extract_feature = docs[i].get('extract_feature', [])
                extract_feature_str = f"Array ({len(extract_feature)})\n" + "\n".join([f"{j}: {textwrap.fill(item, width=100)}" for j, item in enumerate(extract_feature)])
            else:
                filename = ''
                extract_feature_str = ''
            
            if i < len(tokenizers):
                tokenizer = tokenizers[i]
            else:
                tokenizer = ''
            
            writer.writerow([vulnerability_type, filename, extract_feature_str, tokenizer])

print("Đã tạo file CSV thành công")