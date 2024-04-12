from pymongo import MongoClient
import os
import chardet
import re

# Hàm tiền xử lý code
def preprocess_code(code):
    # Loại bỏ comments
    code = re.sub(re.compile(r"/\*.*?\*/", re.DOTALL), "", code)  # loại bỏ comment block
    code = re.sub(re.compile(r"//.*?\n"), "", code)  # loại bỏ comment inline
    
    code = re.sub(r'\d+','<number>', code)  # Thay thế số bằng <number>
    code = re.sub(r'".*?"', '<string>', code)  # Thay thế chuỗi bằng <string>

    code = re.sub(r'\t', '    ', code)  # Chuyển tất cả tabs thành 4 khoảng trắng
    code = re.sub(r' {2,}', ' ', code)  # Chuyển tất cả khoảng trắng liên tiếp thành một khoảng trắng
    code = re.sub(r' *\n', '\n', code)  # Loại bỏ khoảng trắng thừa ở cuối mỗi dòng

    # Loại bỏ dòng trống
    code = re.sub(r'\n\s*\n', '\n', code)
    return code.strip()


# Thiết lập kết nối đến MongoDB
client = MongoClient('localhost', 27017)
db = client['database']
contracts_collection = db['reentrancy']

# Đường dẫn tới thư mục dataset của bạn
dataset_path = r'D:\GitHub\Blockchain-Smart-Contract-Security\Dataset\reentrancy'

# Duyệt qua các thư mục và file trong dataset
for vulnerability_type in os.listdir(dataset_path):
    vulnerability_path = os.path.join(dataset_path, vulnerability_type)
    if os.path.isdir(vulnerability_path):
        for contract_file in os.listdir(vulnerability_path):
            if contract_file.endswith('.sol'):  # Only process .sol files
                contract_path = os.path.join(vulnerability_path, contract_file)
                with open(contract_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']
                
                with open(contract_path, 'r', encoding=encoding) as file:
                    code = file.read()
                    
                    preprocessed_code = preprocess_code(code)
                    
                    contract_document = {
                        'name': contract_file,
                        'code': preprocessed_code,
                        'vulnerability': vulnerability_type
                    }
                    
                    # Chèn document vào collection
                    contracts_collection.insert_one(contract_document)
