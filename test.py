import networkx as nx
from torch_geometric.utils import from_networkx
import torch
from torch_geometric.data import Data

def parse_ast(ast_content):
    nodes = []
    edges = []
    current_node = None

    for line in ast_content.splitlines():
        if "ContractDefinition" in line or "FunctionDefinition" in line or "VariableDeclaration" in line:
            node_type = line.split()[0]
            node_name = line.split('"')[1]
            nodes.append((node_name, {'type': node_type}))
            if current_node:
                edges.append((current_node, node_name))
            current_node = node_name
        elif "Source:" in line:
            continue
        else:
            current_node = None

    return nodes, edges

def create_networkx_graph(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def convert_to_pytorch_geometric(graph):
    data = from_networkx(graph)
    return data

if __name__ == "__main__":
    with open(r'D:\GitHub\Blockchain-Smart-Contract-Security\Data\Interaction and Contract State Vulnerabilities\reentrancy\test.txt', 'r') as f:
        ast_content = f.read()
    
    nodes, edges = parse_ast(ast_content)
    nx_graph = create_networkx_graph(nodes, edges)
    data = convert_to_pytorch_geometric(nx_graph)

    print(data)
