from datasets import load_dataset
import requests
import os
import pandas as pd

# Thay thế 'YOUR_API_KEY' bằng API key thực của bạn từ Etherscan
ETHERSCAN_API_KEY = 'YOUR_API_KEY'

def get_source_code(contract_address):
    url = f"https://api.etherscan.io/api?module=contract&action=getsourcecode&address={contract_address}&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        source_code = data['result'][0]['SourceCode']
        return source_code
    else:
        return None

# Đường dẫn thư mục để lưu source code
source_code_dir = 'D:\\Github\\Blockchain-Smart-Contract-Security\\source'

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(source_code_dir):
    os.makedirs(source_code_dir)

# Load dataset
malicious = load_dataset("forta/malicious-smart-contract-dataset")
#slither = load_dataset("mwritescode/slither-audited-smart-contracts",'all-plain-text')
malicious_df = malicious['train'].to_pandas()

# Tải source code và lưu vào file
for index, row in malicious_df.iterrows():
    code = get_source_code(row['contract_address'])
    if code:
        file_name = row['contract_name'] if pd.notna(row['contract_name']) else row['contract_address']
        # Xử lý các ký tự không hợp lệ trong tên file
        file_name = "".join([c for c in file_name if c.isalpha() or c.isdigit() or c in [' ', '_', '-']]).rstrip()
        file_path = os.path.join(source_code_dir, f'{file_name}.sol')
        with open(file_path, 'w') as file:
            file.write(code)
