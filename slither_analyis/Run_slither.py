import os
import json
import subprocess

# Đường dẫn đến thư mục chứa các file .sol
folder_path = r'C:\Users\Admin\OneDrive\Documents\GitHub\Blockchain-Smart-Contract-Security\get_source\dataset'

# Tạo một dictionary để lưu kết quả
results = {}

# Duyệt qua tất cả các file trong thư mục
for filename in os.listdir(folder_path):
    if filename.endswith('.sol'):
        # Tạo đường dẫn đầy đủ đến file
        file_path = os.path.join(folder_path, filename)
        
        # Lấy phiên bản Solidity từ file .sol
        with open(file_path, 'r',encoding='utf-8') as f:
            lines = f.readlines()
        pragma_line = next((line for line in lines if line.startswith('pragma solidity')), None)
        split_line = pragma_line.split('^')
        if len(split_line) > 1:
            version = split_line[1].strip('; \n')
        else:
            version = 'latest'

        
        # Chọn phiên bản Solidity phù hợp
        subprocess.run(['solc-select', 'use', version], capture_output=True, text=True)
        
        # Chạy lệnh slither và lấy kết quả
        result = subprocess.run(['slither', file_path], capture_output=True, text=True)
        
        # Kiểm tra xem liệu Slither có hoạt động đúng hay không
        if result.returncode != 0:
            print(f"An error occurred while running slither on {file_path}.")
            results[filename] = result.stderr
        else:
            # Lấy phần kết quả xuất ra sau khi chạy lệnh slither
            output_lines = result.stdout.split('\n')
            info_lines = [line for line in output_lines if line.startswith('INFO:')]
            results[filename] = '\n'.join(info_lines)

# Lưu kết quả vào file JSON
with open('results.json', 'w') as f:
    json.dump(results, f)
