import os
from slither import Slither
from pathlib import Path
import torch
from torch_geometric.data import Data

def analyze_contract_to_cfg(file_path):
    # Phân tích hợp đồng sử dụng Slither
    slither = Slither(file_path)
    
    # Danh sách các cạnh cho CFG
    edges = []
    node_idx = 0
    node_map = {}

    # Duyệt qua các hợp đồng trong file
    for contract in slither.contracts:
        contract_node = contract.name
        if contract_node not in node_map:
            node_map[contract_node] = node_idx
            node_idx += 1
        
        # Duyệt qua các hàm của hợp đồng
        for function in contract.functions:
            function_node = f"{contract_node}.{function.name}"
            if function_node not in node_map:
                node_map[function_node] = node_idx
                node_idx += 1
            edges.append((node_map[contract_node], node_map[function_node]))

            # Tạo các nút cho mỗi khối mã trong hàm
            for node in function.nodes:
                block_label = f"{function_node}.{node.node_id}"
                if block_label not in node_map:
                    node_map[block_label] = node_idx
                    node_idx += 1
                
                # Tạo các cạnh giữa các khối mã
                for successor in node.sons:
                    successor_label = f"{function_node}.{successor.node_id}"
                    if successor_label not in node_map:
                        node_map[successor_label] = node_idx
                        node_idx += 1
                    edges.append((node_map[block_label], node_map[successor_label]))

    return edges, len(node_map)

def convert_to_pyg_data(edges, num_nodes):
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.ones((num_nodes, 1), dtype=torch.float)  # Đặc trưng nút
    data = Data(x=x, edge_index=edge_index)
    return data

def main():
    current_dir = Path(__file__).resolve().parent
    folder_path = current_dir / "../Data/Interaction and Contract State Vulnerabilities/delegatecall"   
    output_folder = current_dir / "../Data/Interaction and Contract State Vulnerabilities/graph"  # Thư mục để lưu các hình ảnh
    os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".sol"):
            file_path = os.path.join(folder_path, file_name)
            edges, num_nodes = analyze_contract_to_cfg(file_path)
            data = convert_to_pyg_data(edges, num_nodes)
            output_file = output_folder / f"{file_name}."
            torch.save(data, output_file)
            print(f"Saved CFG data for {file_name} to {output_file}")

if __name__ == "__main__":
    main()
