import os
import subprocess
import requests
from pymongo import MongoClient, DESCENDING
from tqdm import tqdm
from datasets import load_dataset
import re
from packaging import version
import json
import pandas as pd

# Kết nối tới MongoDB
client = MongoClient('localhost', 27017)
db = client['blockchain']
collection = db['call-graph']

ETHERSCAN_API_KEY = 'VSYSX5BCCMYTIFHP2BR7EBY5U12TG3I6X5'
source = r'C:\Users\hao30\Documents\GitHub\Blockchain-Smart-Contract-Security\source'

if not os.path.exists(source):
    os.makedirs(source)

def get_source_code(contract_address, file_path):
    url = f"https://api.etherscan.io/api?module=contract&action=getsourcecode&address={contract_address}&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        source_code = data['result'][0]['SourceCode']
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(source_code)

def run_slither_and_save(filename, source_directory, db_collection):
    file_path = os.path.join(source_directory, filename)

    # Xác định phiên bản Solidity phù hợp
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pragma_lines = [line for line in lines if line.startswith('pragma solidity')]
    versions = []
    try:
        for pragma_line in pragma_lines:
            versions.extend(re.findall(r'\d+\.\d+\.\d+', pragma_line))
        max_version = str(max(version.parse(v) for v in versions if version.parse(v) < version.parse('0.9.0') and version.parse(v) >= version.parse('0.4.1'))) if versions else 'latest'
    except Exception as e:
        print(f"Error parsing Solidity version for {filename}: {str(e)}. Skipping...")
        return

    # Sử dụng phiên bản Solidity phù hợp
    subprocess.run(['solc-select', 'use', max_version], cwd=source_directory)

    # Chạy Slither
    try:
        subprocess.run(["slither", file_path, "--print", "call-graph"], cwd=source_directory)
    except subprocess.CalledProcessError as e:
        print(f"Slither failed to run on {filename}: {e}")
        return

   # Duyệt qua các file để tìm và xử lý các file "all_contracts"
    for file in os.listdir(source_directory):
        if "all_contracts" in file:
            file_to_save = os.path.join(source_directory, file)
            
            # Đọc và lưu nội dung của file "all_contracts" vào MongoDB
            with open(file_to_save, 'r', encoding='utf-8') as f:
                file_content = f.read()
                db_collection.insert_one({'filename': filename, 'content': file_content})
                print(f"Saved {file} content to MongoDB")

    # Xóa tất cả các file trong thư mục source
    for file in os.listdir(source_directory):
        os.remove(os.path.join(source_directory, file))
        print(f"Deleted {file}")


def save_last_processed(last_processed):
    with open('last_processed.txt', 'w') as file:
        file.write(last_processed)\
            
def get_last_processed():
    try:
        with open('last_processed.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None


# Load dataset
def get_last_processed_file_from_db():
    last_entry = collection.find_one(sort=[('_id', DESCENDING)])
    return last_entry['sol_file'] if last_entry else None

def main():
    malicious = load_dataset("forta/malicious-smart-contract-dataset")
    malicious_df = malicious['train'].to_pandas()

    last_processed = get_last_processed_file_from_db()
    start_processing = last_processed is None

    processed_count = 0
    max_process = 1000  # Số lượng tối đa để xử lý mỗi lần

    for index, row in tqdm(malicious_df.iterrows(), total=malicious_df.shape[0], desc="Downloading and Analyzing"):
        file_name = row['contract_name'] if pd.notna(row['contract_name']) else row['contract_address']
        file_name = "".join([c for c in file_name if c.isalpha() or c.isdigit() or c in [' ', '_', '-']]).rstrip()

        if processed_count >= max_process:
            break

        if not start_processing:
            if file_name == last_processed:
                start_processing = True
            continue

        file_path = os.path.join(source, f'{file_name}.sol')
        
        if not os.path.exists(file_path):
            get_source_code(row['contract_address'], file_path)

        run_slither_and_save(f'{file_name}.sol', source, collection)
        processed_count += 1
        save_last_processed(file_name)

main()
