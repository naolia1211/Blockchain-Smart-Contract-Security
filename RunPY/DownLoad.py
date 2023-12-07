from datasets import load_dataset
import requests
import os
import pandas as pd
from tqdm import tqdm

# Thay thế 'YOUR_API_KEY' bằng API key thực của bạn từ Etherscan
ETHERSCAN_API_KEY = 'VSYSX5BCCMYTIFHP2BR7EBY5U12TG3I6X5'

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
malicious_df = malicious['train'].to_pandas()

# Tải và lưu source code cho 10 hợp đồng đầu tiên, với thanh tiến trình
for index, row in tqdm(malicious_df.head(10).iterrows(), total=10, desc="Downloading"):
    file_name = row['contract_name'] if pd.notna(row['contract_name']) else row['contract_address']
    # Xử lý các ký tự không hợp lệ trong tên file
    file_name = "".join([c for c in file_name if c.isalpha() or c.isdigit() or c in [' ', '_', '-']]).rstrip()
    file_path = os.path.join(source_code_dir, f'{file_name}.sol')
    if os.path.exists(file_path):
        continue
    code = get_source_code(row['contract_address'])
    if code:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(code)
