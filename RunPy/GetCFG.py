import os
import subprocess
import pymongo
import gridfs
from bson.objectid import ObjectId

def convert_sol_to_cfg_png(sol_file_path, output_dir):
    # Kiểm tra tệp Solidity tồn tại hay không
    if not os.path.isfile(sol_file_path):
        raise FileNotFoundError(f"Tệp {sol_file_path} không tồn tại.")

    # Tạo thư mục output nếu nó chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Tên tệp output CFG và PNG
    cfg_file_path = os.path.join(output_dir, os.path.basename(sol_file_path).replace('.sol', '.cfg'))
    png_file_path = os.path.join(output_dir, os.path.basename(sol_file_path).replace('.sol', '.png'))

    # Chạy lệnh Slither để phân tích và chuyển đổi sang CFG
    slither_cmd = f"slither {sol_file_path} --print cfg > {cfg_file_path}"
    subprocess.run(slither_cmd, shell=True, check=True)

    # Chạy lệnh Graphviz để chuyển đổi CFG sang PNG
    graphviz_cmd = f"dot -Tpng {cfg_file_path} -o {png_file_path}"
    subprocess.run(graphviz_cmd, shell=True, check=True)
    
    print(f"Đã chuyển đổi tệp {sol_file_path} sang {cfg_file_path} và {png_file_path}")
    
    return cfg_file_path, png_file_path

def store_png_in_mongodb(png_file_path, mongo_uri, db_name, collection_name, doc_id):
    # Kết nối đến MongoDB
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    fs = gridfs.GridFS(db)

    # Đọc tệp PNG và lưu trữ vào MongoDB
    with open(png_file_path, 'rb') as f:
        png_data = f.read()
        png_id = fs.put(png_data, filename=os.path.basename(png_file_path))

    # Cập nhật tài liệu trong MongoDB với trường extract_feature_graph
    collection.update_one({'_id': ObjectId(doc_id)}, {'$set': {'extract_feature_graph': png_id}})
    
    print(f"Đã lưu trữ tệp {png_file_path} vào MongoDB với ObjectId {png_id}")

def clean_up_files(*file_paths):
    for file_path in file_paths:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Đã xóa tệp {file_path}")

# Đường dẫn tệp Solidity và thư mục output
sol_file_path = '/mnt/data/32137.sol'  # Thay thế bằng đường dẫn đến tệp Solidity của bạn
output_dir = '/mnt/data/output'  # Thay thế bằng thư mục bạn muốn lưu tệp CFG và PNG

# Thông tin kết nối MongoDB
mongo_uri = 'mongodb://localhost:27017'  # Thay thế bằng URI kết nối đến MongoDB của bạn
db_name = 'your_db_name'  # Thay thế bằng tên database của bạn
collection_name = 'your_collection_name'  # Thay thế bằng tên collection của bạn
doc_id = '6658a36f57ab...'  # Thay thế bằng ObjectId của tài liệu bạn muốn cập nhật

# Chuyển đổi và lưu trữ
cfg_file_path, png_file_path = convert_sol_to_cfg_png(sol_file_path, output_dir)
store_png_in_mongodb(png_file_path, mongo_uri, db_name, collection_name, doc_id)
clean_up_files(cfg_file_path, png_file_path)
