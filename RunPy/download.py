from smart_contracts import SmartContracts
import pandas as pd
import os

def save_contract_to_file(contract_name, contract_address, source_code, output_dir):
    safe_name = "".join([c for c in contract_name if c.isalnum() or c in (' ', '-', '_')]).rstrip()
    filename = f"{safe_name}_{contract_address}.sol"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(source_code)
    
    return filepath

csv_file = r'D:\GitHub\Blockchain-Smart-Contract-Security\RunPy\contract_addresses_3000.csv'

# Thư mục để lưu các file .sol
output_directory = r'D:\GitHub\Blockchain-Smart-Contract-Security\test\delegatecall'

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_directory, exist_ok=True)

# Đọc file CSV và lấy danh sách địa chỉ contract
df = pd.read_csv(csv_file)
contract_addresses = df['address'].tolist()

# Tạo instance của SmartContracts
smart_contracts = SmartContracts("raw")

# Download và chuẩn bị dữ liệu
smart_contracts.download_and_prepare()

# Tạo dataset
dataset = smart_contracts.as_dataset()

# Lọc dataset
filtered_dataset = dataset['train'].filter(lambda example: example['contract_address'] in contract_addresses)

# Xử lý và lưu các contract đã lọc
for contract in filtered_dataset:
    address = contract['contract_address']
    name = contract['contract_name']
    source_code = contract['source_code']
    
    # Lưu contract vào file .sol
    file_path = save_contract_to_file(name, address, source_code, output_directory)
    
    print(f"Saved contract: {name}")
    print(f"Address: {address}")
    print(f"File: {file_path}")
    print("---")

print(f"All contracts have been saved to {output_directory}")