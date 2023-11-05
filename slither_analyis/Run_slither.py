import os
import json
import subprocess
from packaging import version

def run_slither():
    source_directory = r'C:\Users\Admin\Downloads\Blockchain-Smart-Contract-Security\get_source\dataset'
    output_directory = r'C:\Users\Admin\Documents\GitHub\Blockchain-Smart-Contract-Security\slither_analyis\extract_feature'
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
                    if '^=' in pragma_line:
                        versions.append(pragma_line.split('^=')[1].strip('; \n'))
                    elif '^' in pragma_line:
                        versions.append(pragma_line.split('^')[1].strip('; \n'))
                    elif '=' in pragma_line:
                        versions.append(pragma_line.split('=')[1].strip('; \n'))
                    elif '>=' in pragma_line:
                        versions.extend([v.strip('; \n') for v in pragma_line.split('>=')[1:]])
                    elif '<' in pragma_line:
                        versions.extend([v.strip('; \n') for v in pragma_line.split('<')[1:]])
                
                # Chọn phiên bản lớn nhất nhưng nhỏ hơn 0.9.0
                max_version = str(max(version.parse(v) for v in versions if version.parse(v) < version.parse('0.9.0'))) if versions else 'latest'
            except Exception as e:
                print(f"Error parsing Solidity version for {filename}: {str(e)}. Skipping...")
                continue
            
            # Sử dụng solc-select để chọn phiên bản Solidity phù hợp
            subprocess.run(['solc-select', 'use', max_version])
            
            json_file = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.json")
            
            # Kiểm tra xem tệp JSON đã tồn tại hay chưa
            if os.path.exists(json_file):
                print(f"JSON file for {filename} already exists. Skipping...")
                continue
            
            print(f"Running slither on {filename} with Solidity version {max_version}")
            subprocess.run(["slither", file_path, "--json", json_file])
            
            # Đọc tệp JSON đầu ra của Slither
            with open(json_file, 'r',encoding='utf-8') as f:
                data = json.load(f)
            
run_slither()
