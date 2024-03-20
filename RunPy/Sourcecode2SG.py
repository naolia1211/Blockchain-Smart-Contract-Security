from solidity_parser import parser
import json
import os
from graphviz import Digraph

def convert_sol_to_json(sol_file_path):
    with open(sol_file_path, 'r') as file:
        source_code = file.read()
    parsed_data = parser.parse(source_code)
    return json.dumps(parsed_data, indent=4)

def classify_node(label):
    statement_nodes = ["FunctionDefinition", "IfStatement", "FunctionCall", "ExpressionStatement"]
    variable_nodes = ["Identifier"]
    operator_nodes = ["BinaryOperation"]
    extended_library_nodes = ["SafeMath"]

    if any(node_type in label for node_type in statement_nodes):
        return 'statement'
    elif any(node_type in label for node_type in variable_nodes):
        return 'variable'
    elif any(node_type in label for node_type in operator_nodes):
        return 'operator'
    elif any(node_type in label for node_type in extended_library_nodes):
        return 'extended_library'
    else:
        return 'other'

def node_type_color(node_type):
    return {
        'variable': 'blue',
        'statement': 'orange',
        'operator': 'brown',
        'extended_library': 'green',
        'other': 'black'
    }.get(node_type, 'black')

def traverse_ast(node, graph_components, existing_ids, parent=None):
    if isinstance(node, dict):
        node_id = generate_unique_id(node, existing_ids)
        existing_ids.add(node_id)
        node_label = node.get('name', node.get('type', 'Node'))
        graph_components['nodes'].add(f'"{node_id}" [label="{node_label}", color="{node_type_color(classify_node(node_label))}"]')
        if parent:
            graph_components['edges'].add(f'"{parent}" -> "{node_id}"')
        for key, value in node.items():
            if isinstance(value, (list, dict)):
                traverse_ast(value, graph_components, existing_ids, parent=node_id)
    elif isinstance(node, list):
        for item in node:
            traverse_ast(item, graph_components, existing_ids, parent)

def generate_unique_id(node, existing_ids):
    base_id = f"{node.get('type', 'Node')}_{node.get('name', 'Unnamed')}"
    unique_id = base_id
    counter = 0
    while unique_id in existing_ids:
        counter += 1
        unique_id = f"{base_id}_{counter}"
    return unique_id

def json_ast_to_dot(ast_json, output_file):
    graph_components = {'nodes': set(), 'edges': set()}
    existing_ids = set()
    traverse_ast(json.loads(ast_json), graph_components, existing_ids)

    with open(output_file, 'w') as file:
        file.write('strict digraph {\n')
        for node in sorted(graph_components['nodes']):
            file.write(f'    {node};\n')
        for edge in sorted(graph_components['edges']):
            file.write(f'    {edge};\n')
        file.write('}\n')

def process_file(sol_file_path, output_file_path):
    # Convert Solidity to JSON AST
    json_ast = convert_sol_to_json(sol_file_path)

    # Convert JSON AST to DOT and create semantic graph
    json_ast_to_dot(json_ast, output_file_path)

if __name__ == '__main__':
    test_file_path = r'C:\Users\Admin\Documents\GitHub\Blockchain-Smart-Contract-Security\SimpleTest.sol'  # Đường dẫn đến file Solidity của bạn
    output_dot_file_path = r'C:\Users\Admin\Documents\GitHub\Blockchain-Smart-Contract-Security\test\semantic.dot'  # Đường dẫn đầu ra cho file DOT
    process_file(test_file_path, output_dot_file_path)
    print(f'Semantic graph saved to {output_dot_file_path}')
