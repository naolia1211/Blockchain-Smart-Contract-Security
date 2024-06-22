import os
import re
from solcx import compile_source, install_solc, set_solc_version, get_installed_solc_versions
import json
import networkx as nx
import torch
from torch_geometric.data import Data
from multiprocessing import Pool
from pymongo import MongoClient
from bson.binary import Binary
import pickle

def ensure_solc_versions():
    versions_to_install = ['0.4.24', '0.4.25', '0.5.0', '0.5.1', '0.6.0', '0.7.0', '0.8.0']
    for version in versions_to_install:
        try:
            if version not in get_installed_solc_versions():
                install_solc(version)
                print(f"Installed Solidity version {version}")
        except Exception as e:
            print(f"Could not install Solidity version {version}: {str(e)}")

ensure_solc_versions()

def read_solidity_files(directory):
    solidity_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.sol'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                solidity_files.append((file_path, content))
    return solidity_files

def detect_solidity_version(content):
    version_pattern = r'pragma solidity[\s\^>=<]*([\d.]+);'
    match = re.search(version_pattern, content)
    if match:
        return match.group(1)
    return None

def compile_with_fallback(source_code, file_path):
    detected_version = detect_solidity_version(source_code)
    installed_versions = sorted(get_installed_solc_versions(), reverse=True)
    
    if detected_version:
        versions_to_try = [v for v in installed_versions if v.startswith(detected_version)]
        if not versions_to_try:
            versions_to_try = installed_versions
    else:
        versions_to_try = installed_versions

    for version in versions_to_try:
        try:
            set_solc_version(version)
            return compile_source(source_code, output_values=['abi', 'ast'])
        except Exception as e:
            print(f"Failed to compile {file_path} with version {version}: {str(e)}")
    
    raise Exception(f"Failed to compile {file_path} with all available versions")

def extract_features(ast_node):
    def extract_node_features(node):
        features = {
            'type': node.get('nodeType', 'Unknown'),
            'name': node.get('name', ''),
            'line_number': int(node.get('src', '0:0:0').split(':')[0])
        }
        if features['type'] == 'FunctionDefinition':
            features['visibility'] = node.get('visibility', 'default')
            features['state_mutability'] = node.get('stateMutability', 'nonpayable')
        return features

    def traverse_ast(node):
        nodes = []
        edges = []
        if isinstance(node, list):
            for item in node:
                n, e = traverse_ast(item)
                nodes.extend(n)
                edges.extend(e)
        elif isinstance(node, dict):
            node_features = extract_node_features(node)
            nodes.append(node_features)
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    n, e = traverse_ast(value)
                    nodes.extend(n)
                    edges.extend(e)
                    if node_features['name'] and any(n['name'] for n in n):
                        edges.append((node_features['name'], n[0]['name'], {'type': key}))
        return nodes, edges

    return traverse_ast(ast_node)

def build_graph(nodes, edges):
    graph = nx.MultiDiGraph()
    for node in nodes:
        graph.add_node(node['name'], **node)
    for edge in edges:
        graph.add_edge(edge[0], edge[1], **edge[2])
    return graph

def convert_to_pyg(graph):
    node_features = []
    for node, data in graph.nodes(data=True):
        feature_vector = [
            1 if data['type'] == 'FunctionDefinition' else 0,
            1 if data['type'] == 'VariableDeclaration' else 0,
            int(data['line_number'])
        ]
        node_features.append(feature_vector)

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    edge_attr = []
    for _, _, data in graph.edges(data=True):
        edge_feature_vector = [
            1 if data.get('type', '') == 'control_flow' else 0,
            1 if data.get('type', '') == 'data_flow' else 0,
            1 if data.get('type', '') == 'call_relation' else 0
        ]
        edge_attr.append(edge_feature_vector)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def process_contract(file_path, source_code):
    try:
        compiled_sol = compile_with_fallback(source_code, file_path)
        contract_name = list(compiled_sol.keys())[0].split(':')[-1]
        ast = compiled_sol[f'<stdin>:{contract_name}']['ast']

        nodes, edges = extract_features(ast)
        graph = build_graph(nodes, edges)
        pyg_data = convert_to_pyg(graph)
    
        return file_path, pyg_data
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return file_path, None

def save_to_mongodb(results, db_name, collection_name):
    client = MongoClient('mongodb://localhost:27017/')
    db = client[db_name]
    collection = db[collection_name]

    for file_path, pyg_data in results.items():
        if pyg_data is not None:
            serialized_data = Binary(pickle.dumps(pyg_data))
            document = {
                "file_path": file_path,
                "pyg_data": serialized_data
            }
            collection.insert_one(document)

    print(f"Saved {len([v for v in results.values() if v is not None])} documents to MongoDB")
    client.close()

def main(directory):
    try:
        solidity_files = read_solidity_files(directory)
        with Pool() as p:
            results = p.starmap(process_contract, solidity_files)
        return {file_path: result for file_path, result in results if result is not None}
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        return {}

if __name__ == "__main__":
    directory = r"D:\GitHub\Blockchain-Smart-Contract-Security\test\delegatecall"
    results = main(directory)
    if results:
        save_to_mongodb(results, "solidity_analysis", "contract_graphs")
        for file_path, pyg_data in results.items():
            print(f"Successfully processed: {file_path}")
    else:
        print("No contracts were successfully processed.")