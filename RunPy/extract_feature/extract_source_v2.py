import os
import re
from pymongo import MongoClient
from pathlib import Path

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['Data']
collection = db['Data']

current_dir = Path(__file__).resolve().parent
solidity_files_directory = r"D:\GitHub\Blockchain-Smart-Contract-Security\BigData"

def extract_functions(contract_code):
    function_pattern = re.compile(
        r'\bfunction\s+\w+\s*\([^)]*\)\s*(?:public|private|internal|external|view|pure|payable|constant)?\s*(?:returns\s*\(.*?\))?\s*\{[\s\S]*?\}',
        re.MULTILINE | re.DOTALL
    )
    return [match.group(0) for match in function_pattern.finditer(contract_code)]

def extract_info(function_content):
    patterns = {
        #Interaction and Contract State Vulnerabilities.
        'external_call_function': r'\.(call|send|delegatecall|transfer)\s*\(',
        'state_changes_after_external_calls': r'\.(call|send|delegatecall|transfer)\s*\(.*?\)\s*;[\s\S]*?\w+\s*=',
        'error_handling': r'\b(require|revert|assert)\s*\(.*?\);',
        'fallback_function_interaction': r'function\s+(fallback|receive)\s*\(\s*\)\s*(external|public)?',
        'ignored_return_value': r'\.(call|send|delegatecall|transfer)\s*\(.*?\)\s*;',
        'state_variables_manipulation': r'\b\w+\s*=\s*.*?;',
        'recursive_calls': r'function\s+(\w+)\s*\(.*?\)\s*{(?:(?!\bfunction\b).)*\b\1\s*\(',
        'use_of_msg_value_or_sender': r'\b(msg\.value|msg\.sender)\b',
        'use_of_delegatecall': r'\bdelegatecall\s*\(',
        'library_safe_practices': r'\blibrary\s+\w+',
        'input_and_parameter_validation': r'\brequire\s*\(.*?\)',
        'gas_limitations_in_fallback_functions': r'function\s+(fallback|receive)\s*\(\s*\)\s*(external|public)?\s*{(?:(?!\bgasleft\s*\(\s*\)).)*gasleft\s*\(\s*\).*?;',
        
        # Dependency Vulnerabilities
        'usage_of_block_timestamp': r'\bblock\.timestamp\b',
        'usage_of_block_number': r'\bblock\.number\b',
        'usage_of_block_blockhash': r'\bblock\.blockhash\s*\(',
        'miner_manipulation': r'\b(block\.timestamp|block\.number)\b',
        'usage_in_comparison_operations': r'\bblock\.number\s*(==|!=|>|<|>=|<=)',
        'lack_of_private_seed_storage': r'\bprivate\s+\w+\s+seed\b',
        'historical_block_data': r'\b(block\.number|block\.timestamp|block\.blockhash)\b',
        'predictability_of_randomness_sources': r'\b(block\.timestamp|block\.blockhash|block\.number)\b',
        
        # Authentication and Authorization Vulnerabilities
        'insufficient_input_and_parameter_validation': r'\b(require|assert|revert)\s*\(.*?\)',
        'owner_variable': r'\baddress\s+(public|private|internal)?\s*owner\b',
        'modifier_usage': r'\bmodifier\s+\w+',
        'function_visibility': r'\bfunction\s+\w+\s*\(.*?\)\s*(public|external|internal|private)',
        'authentication_checks': r'\b(require|assert)\s*\(\s*(msg\.sender|tx\.origin)\s*==\s*\w+\s*\)',
        'state_changing_external_calls': r'\b(address\s*\(\s*\w+\s*\)|address\s+\w+)\.(call|delegatecall|transfer)\s*\(',
        'selfdestruct_usage': r'\bselfdestruct\s*\(',
        'authorization_checks_using_tx_origin': r'\b(require|assert)\s*\(\s*tx\.origin\s*==\s*\w+\s*\)',
       
        # Resource (Gas) usage vulnerabilities
        'dependency_on_function_order': r'function\s+\w+\s*\(.*?\)\s*{(?:(?!\bfunction\b).)*\b\w+\s*\(',
        'high_gas_price_transactions': r'\.gas\s*\(\s*\d+\s*\)',
        'require_statements': r'\brequire\s*\(.*?\)',
        'state_variables_manipulation': r'\b(storage|mapping|array)\b',
        'concurrent_function_invocations': r'function\s+\w+\s*\(.*?\)\s*{(?:(?!\bfunction\b).)*\b\w+\s*\(',
        'gas_limit_checks': r'\bgasleft\s*\(\s*\)',
        'state_changing_operations': r'\b(storage|mapping|array)\b',
        'recursive_function_calls': r'function\s+(\w+)\s*\(.*?\)\s*{(?:(?!\bfunction\b).)*\b\1\s*\(',
        'high_complexity_loops': r'\b(for|while)\s*\(.*?\)\s*{',
    }

    results = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, function_content, re.MULTILINE | re.DOTALL)
        if matches:
            if isinstance(matches[0], tuple):
                matches = [''.join(match) for match in matches]
            results[key] = "\n".join(matches)
        else:
            results[key] = ""
    return results

def extract_functions_with_issues(contract_code):
    # Initialize the dictionary with all features
    functions_with_issues = {key: [] for key in [
        'external_call_function', 'state_changes_after_external_calls', 'error_handling', 
        'fallback_function_interaction', 'ignored_return_value', 'state_variables_manipulation', 
        'recursive_calls', 'use_of_msg_value_or_sender', 'use_of_delegatecall', 'library_safe_practices', 
        'input_and_parameter_validation', 'gas_limitations_in_fallback_functions', 'usage_of_block_timestamp', 
        'usage_of_block_number', 'usage_of_block_blockhash', 'miner_manipulation', 'usage_in_comparison_operations', 
        'lack_of_private_seed_storage', 'historical_block_data', 'predictability_of_randomness_sources', 
        'insufficient_input_and_parameter_validation', 'owner_variable', 'modifier_usage', 'function_visibility', 
        'authentication_checks', 'state_changing_external_calls', 'selfdestruct_usage', 
        'authorization_checks_using_tx_origin', 'dependency_on_function_order', 'high_gas_price_transactions', 
        'require_statements', 'state_variables_manipulation', 'concurrent_function_invocations', 'gas_limit_checks', 
        'state_changing_operations', 'recursive_function_calls', 'high_complexity_loops'
    ]}
    
    functions = extract_functions(contract_code)

    for function_content in functions:
        issues = extract_info(function_content)
        for key, value in issues.items():
            if value:
                functions_with_issues[key].append({
                    "function_name": function_content.split('(')[0].split()[-1],
                    "function_content": function_content,
                })
    return functions_with_issues

# Dictionary để ánh xạ từ tên nhóm lỗ hổng sang giá trị số nguyên
label_map = {
    "Authentication_and_Authorization_Vulnerabilities": 1,
    "Dependency_vulnerabilities": 2,
    "Interaction_and_constract_state_vulnerabilities": 3,
    "Resource_(Gas)_usage_vulnerabilities": 4
}

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    functions_with_issues = extract_functions_with_issues(content)
    
    # Xác định label dựa trên thư mục con ở lớp đầu tiên
    label = None
    for folder in label_map.keys():
        if folder in file_path:
            label = folder
            break
    
    document = {
        "filename": os.path.basename(file_path),
        "content": content,
        "functions_with_issues": functions_with_issues,
        "label": label
    }
    result = collection.insert_one(document)
    if result.acknowledged:
        print(f"Processed and stored findings for {file_path} in MongoDB.")
    else:
        print(f"Failed to store findings for {file_path}.")

def process_directory(directory_path):
    file_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".sol"):
                file_list.append(os.path.join(root, file))

    for file_path in file_list:
        process_file(file_path)

# Process the directory
process_directory(solidity_files_directory)