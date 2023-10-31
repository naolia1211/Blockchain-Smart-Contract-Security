import os
import json
import subprocess
import re
from packaging import version

def run_slither():
    source_directory = r'C:\Users\20520\Blockchain-Smart-Contract-Security\get_source\dataset'
    output_directory = r'C:\Users\20520\Blockchain-Smart-Contract-Security\slither_analyis\extract_feature'
    for filename in os.listdir(source_directory):
        if filename.endswith(".sol"):
            file_path = os.path.join(source_directory, filename)
            
            # Đọc nội dung tệp và tìm dòng chứa 'pragma solidity'
            with open(file_path, 'r',encoding='utf-8') as f:
                lines = f.readlines()
            pragma_lines = [line for line in lines if line.startswith('pragma solidity')]
            
            # Tách phiên bản Solidity từ dòng 'pragma solidity'
            versions = []
            try:
                for pragma_line in pragma_lines:
                    versions.extend(re.findall(r'\d+\.\d+\.\d+', pragma_line))  # chỉ lấy số phiên bản
                
                # Chọn phiên bản 0.8 lớn nhất
                max_version = str(max(version.parse(v) for v in versions if version.parse(v) < version.parse('0.9.0') and version.parse(v) >= version.parse('0.4.0'))) if versions else 'latest'
            except Exception as e:
                print(f"Error parsing Solidity version for {filename}: {str(e)}. Skipping...")
                continue
            
            # Sử dụng solc-select để chọn phiên bản Solidity phù hợp
            subprocess.run(['solc-select', 'use', max_version], cwd=source_directory)
            
            json_file = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.json")
            
            # Kiểm tra xem tệp JSON đã tồn tại hay chưa
            if os.path.exists(json_file):
                print(f"{filename}'s json already exists.....")
                continue
            
            print(f"Running slither on {filename} with Solidity version {max_version}")
            subprocess.run(["slither", filename, "--json", json_file], cwd=source_directory)
            
            # Đọc tệp JSON đầu ra của Slither
            with open(json_file, 'r') as f:
                data = json.load(f)
            
run_slither()
