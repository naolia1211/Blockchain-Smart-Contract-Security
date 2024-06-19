import os
import re

def process_files(directory):
    # Duyệt qua tất cả các file trong thư mục
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Kiểm tra nếu file là file thường
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='latin-1') as file:
                    lines = file.readlines()
                
                new_lines = []
                skip_line = False
                for line in lines:
                    # Kiểm tra và bỏ qua các dòng không cần thiết
                    if re.match(r"^======= delegatecall/sourcecode/\d+\.sol:.* =======$", line):
                        skip_line = True
                    elif line.strip() == "Binary:":
                        skip_line = True
                    else:
                        skip_line = False
                    
                    # Thêm dòng vào new_lines nếu không phải dòng bỏ qua
                    if not skip_line:
                        new_lines.append(line)
                
                # Ghi lại các dòng cần giữ vào file
                with open(filepath, 'w', encoding='latin-1') as file:
                    file.writelines(new_lines)
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")

# Đường dẫn đến thư mục chứa các file
directory = r'D:\GitHub\Blockchain-Smart-Contract-Security\test\bin'  # Thay đổi đường dẫn này nếu cần

# Gọi hàm để xử lý các file
process_files(directory)
