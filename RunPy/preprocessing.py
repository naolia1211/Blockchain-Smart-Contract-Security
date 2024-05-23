import os
import re
import logging

# Thiết lập logging
logging.getLogger().setLevel(logging.INFO)

# Định nghĩa các mẫu regex để tìm và loại bỏ comment
MULTILINE_COMMENT_PATTERN = re.compile(r'/\*[\s\S]*?\*/')
SINGLELINE_COMMENT_PATTERN = re.compile(r'//.*')
ATHUR_COMMENT_PATTERN = re.compile(r'///.*')
EXCESS_WHITESPACE_PATTERN = re.compile(r'\s{2,}')
LEADING_ASTERISK_PATTERN = re.compile(r'^\s*\*', re.MULTILINE)
BLANK_LINES_PATTERN = re.compile(r'^\s*$', re.MULTILINE)

def preprocess_code(code):
    code = MULTILINE_COMMENT_PATTERN.sub('', code)
    code = SINGLELINE_COMMENT_PATTERN.sub('', code)
    code = ATHUR_COMMENT_PATTERN.sub('', code)
    code = code.replace('\t', '')
    code = EXCESS_WHITESPACE_PATTERN.sub(' ', code)
    code = LEADING_ASTERISK_PATTERN.sub('', code)
    code = BLANK_LINES_PATTERN.sub('', code)
    return code.strip()

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
input_directory = r'C:\Users\hao30\Documents\GitHub\Blockchain-Smart-Contract-Security\Smart_Contracts\3_unauthorized_send'

# Đường dẫn thư mục để lưu các file đã tiền xử lý
output_directory = r'C:\Users\hao30\Documents\GitHub\Blockchain-Smart-Contract-Security\Data\Interaction and Contract State Vulnerabilities\unchecked send'

# Xử lý thư mục
process_directory(input_directory, output_directory)
