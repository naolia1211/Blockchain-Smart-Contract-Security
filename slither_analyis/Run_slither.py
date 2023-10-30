import os
import json
import subprocess
from packaging import version

def run_slither():
    source_directory = "C:\\Users\\Admin\\OneDrive\\Documents\\GitHub\\Blockchain-Smart-Contract-Security\\get_source\\dataset"
    output_directory = "C:\\Users\\Admin\\OneDrive\\Documents\\GitHub\\Blockchain-Smart-Contract-Security\\slither_analyis\\extract_feature"
    for filename in os.listdir(source_directory):
        if filename.endswith(".sol"):
            file_path = os.path.join(source_directory, filename)
            
            # Đọc nội dung tệp và tìm dòng chứa 'pragma solidity'
            with open(file_path, 'r') as f:
                lines = f.readlines()
            pragma_lines = [line for line in lines if line.startswith('pragma solidity')]
            
            # Tách phiên bản Solidity từ dòng 'pragma solidity'
            versions = []
            for pragma_line in pragma_lines:
                if '^=' in pragma_line:
                    versions.append(pragma_line.split('^=')[1].strip('; \n'))
                elif '^' in pragma_line:
                    versions.append(pragma_line.split('^')[1].strip('; \n'))
                elif '=' in pragma_line:
                    versions.append(pragma_line.split('=')[1].strip('; \n'))
            
            # Chọn phiên bản lớn nhất
            max_version = str(max(version.parse(v) for v in versions)) if versions else 'latest'
            
            # Sử dụng solc-select để chọn phiên bản Solidity phù hợp
            subprocess.run(['solc-select', 'use', max_version])
            
            print(f"Running slither on {filename} with Solidity version {max_version}")
            json_file = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.json")
            subprocess.run(["slither", file_path, "--json", json_file])
            
            # Đọc tệp JSON đầu ra của Slither
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Lọc chỉ những lỗi có mức độ nghiêm trọng là 'High' hoặc 'Medium'
            high_and_medium_errors = [result for result in data['results'] if result['severity'] in ['High', 'Medium']]
            
            # Lưu những lỗi có mức độ nghiêm trọng là 'High' hoặc 'Medium' vào một tệp JSON mới
            with open(os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_high_medium.json"), 'w') as f:
                json.dump(high_and_medium_errors, f)

run_slither()
