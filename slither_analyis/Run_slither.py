import os
import subprocess
import json

# Thay đổi đường dẫn này thành thư mục chứa các tệp .sol của bạn
directory = r'C:\Users\20520\Blockchain-Smart-Contract-Security\get_source\dataset'

results = []

for filename in os.listdir(directory):
    if filename.endswith('.sol'):
        # Lấy phiên bản solc từ tệp .sol
        with open(os.path.join(directory, filename), 'r') as file:
            for line in file:
                if line.startswith('pragma solidity'):
                    version = line.split('^')[1].split(';')[0].strip()

                    # Chọn phiên bản solc phù hợp
                    subprocess.run(['solc-select', 'use', version], shell=True)

                    # Chạy slither cho tệp .sol
                    result = subprocess.run("slither '.\$MLG.sol'", shell=True, capture_output=True, text=True)

                    # Lưu kết quả vào danh sách
                    results.append({'filename': filename, 'result': result.stdout})
                    break

# Lưu kết quả vào tệp JSON
with open('results.json', 'w') as file:
    json.dump(results, file)
