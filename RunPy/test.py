from pymongo import MongoClient
import pandas as pd

# Kết nối tới MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Interaction_and_Contract_State_Vulnerabilities']

# Các collection tương ứng với các loại lỗ hổng
collections = {
    'delegatecall': 0,
    'reentrancy': 1,
    'uncheck_external_call': 2,
    'uncheck_send': 3
}

# Lấy dữ liệu từ các collection và gán nhãn
data = []
for collection_name, label in collections.items():
    collection = db[collection_name]
    documents = list(collection.find())
    for doc in documents:
        for feature in doc['extract_feature']:
            data.append({
                'tokens': feature['tokens'],
                'label': label
            })

# Chuyển đổi dữ liệu thành DataFrame
df = pd.DataFrame(data)

# Lưu dữ liệu đã chuẩn bị thành tệp CSV
df.to_csv('solidity_vulnerabilities.csv', index=False)
