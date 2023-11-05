import os
import json
import subprocess

def extract_features(dataset_path, output_path):
    # Lấy danh sách tất cả các file .sol trong thư mục dataset
    sol_files = [f for f in os.listdir(dataset_path) if f.endswith('.sol')]

    for sol_file in sol_files:
        # Đường dẫn đầy đủ tới file .sol
        sol_file_path = os.path.join(dataset_path, sol_file)

        # Tên file .json tương ứng
        json_file = sol_file.replace('.sol', '.json')
        json_file_path = os.path.join(output_path, json_file)

        # Chạy lệnh mythril để trích xuất features
        cmd = f'myth analyze {sol_file_path} -o json'
        result = subprocess.run(cmd, shell=True, capture_output=True)

        # Lưu kết quả vào file .json
        with open(json_file_path, 'w') as f:
            json.dump(json.loads(result.stdout), f)

#Sử dụng hàm
extract_features('/home/loanxinhdep/dataset', '/home/loanxinhdep/output')