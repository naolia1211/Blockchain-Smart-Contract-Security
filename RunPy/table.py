import csv
import os
from pymongo import MongoClient

# Kết nối tới MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Interaction_and_Contract_State_Vulnerabilities']

# Các nhóm lỗ hổng
vulnerability_groups = ['reentrancy', 'delegatecall', 'unchecked_send']

# Đường dẫn tới thư mục chứa các tệp tin tokenize và đầu ra CSV
tokenize_dir = r'C:\Users\hao30\Documents\GitHub\Blockchain-Smart-Contract-Security\RunPy\output_tokenize'

# Tên file CSV đầu ra
csv_filename = os.path.join(tokenize_dir, 'vulnerability_data.csv')

# Tạo file CSV và ghi dữ liệu vào
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['vulnerability_group', 'smart_contract', 'feature_extraction', 'tokenizer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for group in vulnerability_groups:
        # Lấy dữ liệu từ collection tương ứng
        collection = db[group]
        data = collection.find({}, {'filename': 1, 'extract_feature': 1})

        for item in data:
            filename = item['filename']
            extract_feature = item['extract_feature']

            # Tìm tệp tin tương ứng trong thư mục tokenize
            tokenize_file = os.path.join(tokenize_dir, f"{os.path.splitext(filename)[0]}.txt")

            if os.path.exists(tokenize_file):
                with open(tokenize_file, 'r', encoding='utf-8') as file:
                    tokenizer = file.read()
            else:
                tokenizer = ''

            writer.writerow({'vulnerability_group': group, 'smart_contract': filename, 'feature_extraction': extract_feature, 'tokenizer': tokenizer})

print("CSV file created successfully.")