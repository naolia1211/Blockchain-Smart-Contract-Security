import json
import networkx as nx
import matplotlib.pyplot as plt
from solcx import compile_standard
from pathlib import Path
# Install py-solc-x library using: pip install py-solc-x
# Install networkx library using: pip install networkx
# Install matplotlib library using: pip install matplotlib

def compile_solidity(file_path):
    with open(file_path, 'r') as file:
        source_code = file.read()

    compiled_sol = compile_standard({
        "language": "Solidity",
        "sources": {
            "Contract.sol": {
                "content": source_code
            }
        },
        "settings": {
            "outputSelection": {
                "*": {
                    "*": ["metadata", "evm.bytecode", "evm.sourceMap"]
                }
            }
        }
    })

    return compiled_sol

def extract_ast(compiled_sol):
    contract_name = list(compiled_sol['contracts']['Contract.sol'].keys())[0]
    metadata = json.loads(compiled_sol['contracts']['Contract.sol'][contract_name]['metadata'])
    ast = metadata['output']['sources']['Contract.sol']['ast']
    return ast

def generate_cfg(ast):
    cfg = nx.DiGraph()

    def traverse(node, parent=None):
        if 'name' in node:
            node_id = node['id']
            cfg.add_node(node_id, label=node['name'])
            if parent:
                cfg.add_edge(parent, node_id)

            for child in node.get('children', []):
                traverse(child, node_id)
        else:
            for child in node.get('children', []):
                traverse(child, parent)

    traverse(ast)
    return cfg

def plot_cfg(cfg):
    pos = nx.spring_layout(cfg)
    labels = nx.get_node_attributes(cfg, 'label')
    nx.draw(cfg, pos, labels=labels, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    plt.show()

def main():
    current_dir = Path(__file__).resolve().parent
    solidity_file_path = current_dir / "../Data/Interaction and Contract State Vulnerabilities/delegatecall/1.sol"
    compiled_sol = compile_solidity(solidity_file_path)
    ast = extract_ast(compiled_sol)
    cfg = generate_cfg(ast)
    plot_cfg(cfg)

if __name__ == "__main__":
    main()
