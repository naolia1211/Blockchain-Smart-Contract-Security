import re

# Đọc nội dung từ file
with open('a.txt', 'r') as file:
    content = file.read()

# Biểu thức chính quy để tìm các hàm trong phần "External calls"
# Điều chỉnh biểu thức chính quy để phù hợp với định dạng dữ liệu của bạn
pattern = r'External calls:(?:\n\t- [^\n]+\n\t\t- ([^\(]+)\()'

# Tìm tất cả các kết quả phù hợp
matches = re.findall(pattern, content)

# In ra tên các hàm
for match in matches:
    print(match.strip())
