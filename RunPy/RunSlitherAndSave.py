import os
import subprocess
import re

# Đường dẫn đến thư mục chứa các file Solidity
solidity_dir = r"D:\GitHub\Blockchain-Smart-Contract-Security\Data\Interaction and Contract State Vulnerabilities\Unchecked_external_call\source"

# Đường dẫn đến thư mục đầu ra cho bytecode
output_dir = r"D:\GitHub\Blockchain-Smart-Contract-Security\Data\Interaction and Contract State Vulnerabilities\Unchecked_external_call\bytecode"

# Tạo thư mục đầu ra nếu chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Duyệt qua từng file Solidity trong thư mục
for file_name in os.listdir(solidity_dir):
    if file_name.endswith(".sol"):
        file_path = os.path.join(solidity_dir, file_name)
        
        # Đọc nội dung của file Solidity
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
        
        # Tìm phiên bản Solidity từ pragma statement
        match = re.search(r"pragma solidity\s+(\^|>=|>|<=|<)?\s*(\d+\.\d+\.\d+);", content)
        
        if match:
            solc_version = match.group(2)
        else:
            solc_version = "0.8.0"  # Giả sử phiên bản mặc định là 0.8.0 nếu không tìm thấy pragma
        
        # Chọn phiên bản Solidity compiler bằng solc-select
        command = f"solc-select use {solc_version}"
        subprocess.run(command, shell=True, check=True)
        
        # Biên dịch file Solidity thành bytecode
        output_file = file_name.replace(".sol", ".bin")
        output_path = os.path.join(output_dir, output_file)
        
        command = f'solc --bin "{file_path}" -o "{output_dir}"'
        
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Biên dịch {file_name} thành công. Bytecode được lưu trong {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Lỗi biên dịch {file_name}: {e}")
