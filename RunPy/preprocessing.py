import os
import re
import logging
from pathlib import Path

# Thiết lập logging
logging.getLogger().setLevel(logging.INFO)

# Định nghĩa các mẫu regex để tìm và loại bỏ comment
MULTILINE_COMMENT_PATTERN = re.compile(r'/\*[\s\S]*?\*/')
SINGLELINE_COMMENT_PATTERN = re.compile(r'//.*')
ATHUR_COMMENT_PATTERN = re.compile(r'///.*')

def preprocess_code(code):
    code = MULTILINE_COMMENT_PATTERN.sub('', code)
    code = SINGLELINE_COMMENT_PATTERN.sub('', code)
    code = ATHUR_COMMENT_PATTERN.sub('', code)
    return code

def process_file(file_path, output_directory):
    logging.info(f"Processing file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        logging.info("File read successfully")
        
        # Tiền xử lý mã
        preprocessed_content = preprocess_code(content)
        
        # Lưu mã đã tiền xử lý vào thư mục mới
        output_path = os.path.join(output_directory, os.path.basename(file_path))
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(preprocessed_content)
        logging.info(f"Preprocessed content saved to {output_path}")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

def process_directory(input_directory, output_directory):
    # Tạo thư mục mới nếu chưa tồn tại
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(input_directory):
        if filename.endswith(".sol"):
            file_path = os.path.join(input_directory, filename)
            process_file(file_path, output_directory)

# Đường dẫn thư mục chứa các file Solidity
current_dir = Path(__file__).resolve().parent
input_directory =  current_dir / '../Dataset2/Resource_(Gas)_usage_vulnerabilities/denial_of_service'

# Đường dẫn thư mục để lưu các file đã tiền xử lý
output_directory = current_dir / '../Dataset2/Resource_(Gas)_usage_vulnerabilities/denial_of_service'

# Xử lý thư mục
process_directory(input_directory, output_directory)
