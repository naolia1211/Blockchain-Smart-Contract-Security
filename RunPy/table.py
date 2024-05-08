from pymongo import MongoClient
import pandas as pd

# Kết nối đến MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['vul']
collection = db['a']

# Đọc dữ liệu từ MongoDB
data = []
for document in collection.find():
    if 'extract_feature' in document:
        feature_extractions = document['extract_feature']
        data.extend([[feat] for feat in feature_extractions])

# Đường dẫn đến các file
filtered_tokens_file = r'C:\Users\Admin\Documents\GitHub\Blockchain-Smart-Contract-Security\RunPy\filtered_tokens.txt'
tensor_file = r'C:\Users\Admin\Documents\GitHub\Blockchain-Smart-Contract-Security\RunPy\tensor1.txt'

# Đọc các token đã lọc từ file filtered_tokens.txt
with open(filtered_tokens_file, 'r') as file:
    filtered_tokens = [eval(line.strip()) for line in file]

# Đọc các vector từ file tensor1.txt
with open(tensor_file, 'r') as file:
    vectors = [eval(line.strip()) for line in file]

# Tạo DataFrame từ dữ liệu
df = pd.DataFrame(data, columns=['Feature Extraction'])
df['Filtered Tokens'] = filtered_tokens
df['Vector'] = vectors

# Lưu DataFrame vào file CSV
output_file = r'C:\Users\Admin\Documents\GitHub\Blockchain-Smart-Contract-Security\RunPy\output_table.csv'
df.to_csv(output_file, index=False)