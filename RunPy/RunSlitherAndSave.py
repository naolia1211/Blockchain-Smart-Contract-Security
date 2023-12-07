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
collection = db['slither_analyze']

ETHERSCAN_API_KEY = 'VSYSX5BCCMYTIFHP2BR7EBY5U12TG3I6X5'
source = r'D:\Github\Blockchain-Smart-Contract-Security\source'
source = r'D:\Github\Blockchain-Smart-Contract-Security\source'

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

def run_slither_and_save(filename, source_directory, source):
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

    # Tạo đường dẫn cho file json
    json_file_path = os.path.join(source, f"{os.path.splitext(filename)[0]}.json")

    # Chạy Slither và đợi cho đến khi hoàn thành
    try:
        subprocess.run(["slither", filename, "--json", json_file_path], cwd=source_directory)
        os.remove(file_path)
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                json_data = json.load(file)
                collection.insert_one({'sol_file': filename, 'output_json': json_data})
                print(f"Saved {filename} analysis to MongoDB")
            os.remove(json_file_path)
            print(f"Deleted {filename} JSON file")
    except subprocess.CalledProcessError as e:
        print(f"Slither failed to run on {filename}: {e}")
# Load dataset
def get_last_processed_file_from_db():
    last_entry = collection.find_one(sort=[('_id', DESCENDING)])
    return last_entry['sol_file'] if last_entry else None

def main():
    malicious = load_dataset("forta/malicious-smart-contract-dataset")
    malicious_df = malicious['train'].to_pandas()

    last_processed = get_last_processed_file_from_db()
    start_processing = False if last_processed else True

    for index, row in tqdm(malicious_df.iterrows(), total=malicious_df.shape[0], desc="Downloading and Analyzing"):
        file_name = row['contract_name'] if pd.notna(row['contract_name']) else row['contract_address']
        file_name = "".join([c for c in file_name if c.isalpha() or c.isdigit() or c in [' ', '_', '-']]).rstrip()

        if not start_processing:
            if file_name == last_processed:
                start_processing = True
            continue

        file_path = os.path.join(source, f'{file_name}.sol')
        
        if not os.path.exists(file_path):
            get_source_code(row['contract_address'], file_path)
        
        run_slither_and_save(f'{file_name}.sol', source, source)

main()