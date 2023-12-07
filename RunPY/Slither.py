import os
import json
import subprocess
import re
from packaging import version
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

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
    source_directory = r'D:\Github\Blockchain-Smart-Contract-Security\source'
    output_directory = r'D:\Github\Blockchain-Smart-Contract-Security\extract_feature_from_slither'
    filenames = [filename for filename in os.listdir(source_directory) if filename.endswith(".sol")]
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(run_slither, filenames, [source_directory]*len(filenames), [output_directory]*len(filenames))

main()