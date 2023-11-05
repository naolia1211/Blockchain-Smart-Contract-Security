import os
import json
import subprocess
import re
from packaging import version
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

<<<<<<< HEAD
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
=======
lock = Lock()

def run_slither(filename, source_directory, output_directory):
    file_path = os.path.join(source_directory, filename)
    with open(file_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
    pragma_lines = [line for line in lines if line.startswith('pragma solidity')]
    versions = []
    unsupported_versions = ['0.8.20']  # Thêm vào đây các phiên bản không hỗ trợ khác
    try:
        for pragma_line in pragma_lines:
            versions.extend(re.findall(r'\d+\.\d+\.\d+', pragma_line))
        max_version = str(max(version.parse(v) for v in versions if version.parse(v) < version.parse('0.9.0') and version.parse(v) >= version.parse('0.4.0'))) if versions else 'latest'
        if max_version in unsupported_versions:
            max_version = '0.8.9'  # hoặc phiên bản gần nhất mà bạn có
    except Exception as e:
        print(f"Error parsing Solidity version for {filename}: {str(e)}. Skipping...")
        return
    subprocess.run(['solc-select', 'use', max_version], cwd=source_directory)
    json_file = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.json")
    
    with lock:
        if os.path.exists(json_file):
            print(f"{filename}'s json already exists.....")
            return
    
    print(f"Running slither on {filename} with Solidity version {max_version}")
    subprocess.run(["slither", filename, "--json", json_file], cwd=source_directory)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for detector in data['results']['detectors']:
        if 'elements' in detector:
            del detector['elements']
    
    with open(json_file, 'w') as f:
        json.dump(data, f,indent=4)

def main():
    source_directory = r'C:\Users\Admin\OneDrive\Documents\GitHub\Blockchain-Smart-Contract-Security\get_source\dataset'
    output_directory = r'C:\Users\Admin\OneDrive\Documents\GitHub\Blockchain-Smart-Contract-Security\slither_analyis\extract_feature'
    filenames = [filename for filename in os.listdir(source_directory) if filename.endswith(".sol")]
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(run_slither, filenames, [source_directory]*len(filenames), [output_directory]*len(filenames))

main()
>>>>>>> 671267b3af256008f51f9c715632d4493adda470
