import csv
import requests
import os

# Đường dẫn tới file CSV
csv_file = r'C:\Users\20520\Blockchain-Smart-Contract-Security\get_source\export-verified-contractaddress-opensource-license.csv'

# Đường dẫn tới thư mục chứa các file .sol
source_dir = r'C:\Users\20520\Blockchain-Smart-Contract-Security\get_source\dataset'
count = 0
# Đọc file CSV và lấy danh sách các hợp đồng
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    contracts = list(reader)

# Duyệt qua danh sách hợp đồng
for contract in contracts:
    if count == 10:
        break
    address = contract['ContractAddress']
    name = contract['ContractName']

    # Kiểm tra xem file .sol đã tồn tại hay chưa
    sol_file = os.path.join(source_dir, f'{name}.sol')
    if not os.path.exists(sol_file):
        # Tải về mã nguồn hợp đồng nếu file .sol chưa tồn tại
        url = f'https://api.etherscan.io/api?module=contract&action=getsourcecode&address={address}&apikey=VSYSX5BCCMYTIFHP2BR7EBY5U12TG3I6X5'
        response = requests.get(url).json()

        # Kiểm tra xem yêu cầu có thành công hay không
        if response['status'] == '1':
            # Lấy mã nguồn từ phản hồi và kiểm tra xem có chứa chuỗi "content" hay không
            source_code = response['result'][0]['SourceCode']
            if 'content' not in source_code:
                # Nếu mã nguồn không chứa chuỗi "content", lưu vào file .sol
                with open(sol_file, 'w',encoding='utf-8') as f:
                    f.write(source_code)
