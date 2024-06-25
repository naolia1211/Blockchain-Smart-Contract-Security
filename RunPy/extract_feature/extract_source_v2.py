import os
import re
import csv
from pymongo import MongoClient
from pathlib import Path

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['Data6']
collection = db['Data']

current_dir = Path(__file__).resolve().parent

patterns = {
    'external_call_function': r'\.(call|send|delegatecall|transfer)\s*\(',
    'state_change_after_external_call': r'\.(call|send|transfer|delegatecall)\s*\([^;]+\);\s*\w+\s*[=+-]=',
    'error_handling': r'\b(require|revert|assert)\s*\(.*?\);',
    'fallback_function_interaction': r'function\s+(fallback|receive)\s*\(\s*\)\s*(external|public)?',
    'unchecked_external_call': r'\.(call|send|transfer|delegatecall)\s*\([^;]+\)\s*(?!;?\s*require|\s*if|\s*assert)',
    'use_of_msg_value_or_sender': r'\b(msg\.value|msg\.sender)\b',
    'delegatecall_with_msg_data': r'\.delegatecall\s*\(\s*msg\.data',
    'dynamic_delegatecall': r'address\s*\(\s*\w+\s*\)\.delegatecall',
    'library_safe_practices': r'\blibrary\s+\w+',
    'gas_limitations_in_fallback_functions': r'function\s+(fallback|receive)\s*\(\s*\)\s*(external|public)?\s*{(?:(?!\bgasleft\s*\(\s*\)).)*gasleft\s*\(\s*\).*?;',
    'balance_check_before_send': r'require\s*\([^)]*balance[^)]*\);\s*\.(send|transfer)',
    'multiple_external_calls': r'(\.(call|send|transfer|delegatecall)\s*\([^;]+\);\s*){2,}',
    'external_call_in_loop': r'for\s*\([^)]+\)\s*\{[^}]*\.(call|send|transfer|delegatecall)',
    'reentrancy_guard_missing': r'function\s+\w+\s*\([^)]*\)\s*public\s*(?!nonReentrant)',
    'low_level_call': r'\.call\s*\{[^}]*\}\s*\([^\)]*\)',
    'library_usage_in_delegatecall': r'\w+Library\.delegatecall',
    'state_variables_manipulation': r'\b(storage|mapping|array)\s+\w+\s*=',
    'state_update_without_verification': r'\b\w+\s*=\s*.*?;(?!.*(?:require|assert))',
    'recursive_calls': r'function\s+(\w+).*?{[^}]*\b\1\s*\(',
    'context_preservation': r'\.call\s*\{.*?value:.*?\}\s*\(',

    'usage_of_block_timestamp': r'\b(block\.timestamp|now)\b',
    'usage_of_block_number': r'\bblock\.number\b',
    'usage_of_block_blockhash': r'\bblock\.blockhash\s*\(',
    'miner_manipulation': r'\b(block\.timestamp|block\.number|block\.difficulty|block\.coinbase|block\.gaslimit)\b',
    'usage_in_comparison_operations': r'\b(block\.number|block\.timestamp|now)\s*(==|!=|>|<|>=|<=)',
    'block_number_as_time': r'\bblock\.number\b.*?[*+]\s*\d+',
    'historical_block_data': r'blockhash\s*\(\s*block\.number\s*-\s*\d+\s*\)',
    'lack_of_private_seed_storage': r'\bprivate\s+\w+\s+seed\b',
    'predictability_of_randomness_sources': r'(?:random|rand|seed).*?\b(block\.timestamp|block\.blockhash|block\.number|now)\b',
    'block_based_random': r'(block\.(timestamp|number|difficulty|coinbase|gaslimit|hash)|now)\s*.*?(random|seed)',
    'keccak256_random': r'keccak256\s*\(.*?(block\.(timestamp|number|difficulty|coinbase|gaslimit|hash)|now)',
    'unsafe_random_generation': r'function\s+\w+\s*\(.*?\)\s*(public|external).*?(random|seed)',
    'timestamp_arithmetic': r'(block\.timestamp|now)\s*[\+\-\*\/]',
    'timestamp_in_condition': r'if\s*\(\s*(block\.timestamp|now)',
    'timestamp_assignment': r'\w+\s*=\s*(block\.timestamp|now)',
    'timestamp_function_param': r'function\s+\w+\s*\([^)]*\b(block\.timestamp|now)\b[^)]*\)',
    'block_number_arithmetic': r'block\.number\s*[\+\-\*\/]',
    'block_number_in_condition': r'if\s*\(\s*block\.number',
    'block_number_assignment': r'\w+\s*=\s*block\.number',
    'block_number_function_param': r'function\s+\w+\s*\([^)]*\bblock\.number\b[^)]*\)',
    'block_based_operations': r'block\.(timestamp|number|difficulty|coinbase|gaslimit|hash)',
    'miner_susceptible_operations': r'(block\.(timestamp|number|difficulty|coinbase|gaslimit|hash)|now)\s*.*?(if|require|assert)',

    'input_and_parameter_validation': r'\b(require|assert|revert)\s*\(.*?\)',
    'owner_variable': r'\baddress\s+(public|private|internal)?\s*owner\b',
    'modifier_definition': r'modifier\s+(\w+)\s*\([^)]*\)',
    'function_visibility': r'function\s+(\w+)\s*\([^)]*\)\s*(public|external|internal|private)',
    'authentication_checks': r'(require|assert)\s*\(\s*(msg\.sender|tx\.origin)\s*==\s*(\w+|owner)\s*\)',
    'state_changing_external_calls': r'\.(call|delegatecall|transfer|send|call\.value)\s*\(',
    'selfdestruct_usage': r'selfdestruct\s*\(',
    'vulnerable_sensitive_data_handling': r'function\s+\w+\s*\(.*?\)\s*(public|external)(?=.*?{.*?\b(password|secret|key)\b)',
    'improper_input_padding': r'\b(abi\.encodePacked|keccak256)\s*\((?:[^()]*,\s*){2,}[^()]*\)',
    'unchecked_address_variables': r'\baddress\s+\w+\s*(?!=\s*address\(0\))',
    'constructor_ownership_assignment': r'constructor\s*\(.*?\)\s*{[^}]*\bowner\s*=',
    'transaction_state_variable': r'\btx\.(origin|gasprice|gas)\b',
    'ownership_transfer': r'(\w+|owner)\s*=\s*(msg\.sender|tx\.origin)',
    'privileged_operation': r'function\s+\w+\s*\([^)]*\)\s*(public|external)\s*(onlyOwner|require\(.*?==\s*owner\))',
    'unchecked_parameter': r'function\s+\w+\s*\([^)]*\)\s*(public|external)(?![^{]*require)',
    'address_parameter': r'function\s+\w+\s*\([^)]*address[^)]*\)',
    'state_change_after_transfer': r'\.(transfer|send|call\.value)\s*\([^;]+\);\s*\w+\s*[=+-]=',
    'multiple_transfers': r'(\.(transfer|send|call\.value)\s*\([^;]+\);\s*){2,}',
    'price_check_before_action': r'(require|assert)\s*\(\s*\w+\s*(==|<=|>=)\s*[^;]+\);\s*\.(transfer|send|call)',
    'unsafe_type_inference': r'\bvar\s+\w+\s*=',
    'reentrancy_risk': r'\.(transfer|send|call)\s*\([^;]+\);\s*\w+\s*[=+-]=',
    'unchecked_send_result': r'\.(send|call)\s*\([^;]+\)(?!\s*require)',
    'low_level_call': r'\.call\s*\{[^}]*\}\s*\([^\)]*\)',
    'assembly_block': r'assembly\s*\{[^}]*\}',

    'costly_loop': r'(for|while)\s*\([^)]*\)\s*\{[^}]*\}',
    'unbounded_operation': r'for\s*\([^;]*;\s*[^;]*<\s*(\w+)\.length;\s*[^)]*\)',
    'gas_intensive_function': r'function\s+\w+\s*\([^)]*\)\s*(external|public)[^{]*\{[^}]*\}',
    'risky_external_call': r'\.(call|send|transfer)\{?[^}]*\}?\(',
    'improper_exception_handling': r'(throw|revert)\s*\(',
    'unsafe_type_inference': r'(var|uint8|byte)\s+\w+\s*=',
    'arithmetic_in_loop': r'for\s*\([^;]*;\s*[^;]*;\s*\w+\s*[+\-*/]=',
    'unchecked_send_result': r'\.(send|transfer)\([^)]*\)\s*;(?!\s*require)',
    'risky_fallback': r'(fallback|receive)\s*\(\)\s*(external|public)\s*(payable)?\s*\{[^}]*\}',
    'loop_counter_vulnerability': r'for\s*\([^;]*;\s*\w+\s*([<>]=?|==)\s*\d+;\s*[^)]*\)',
    'gas_limit_in_loop': r'for\s*\([^)]*\)\s*\{[^}]*\.(send|transfer|call)\(',

    'arithmetic_operations': r'\b(\+|\-|\*|\/)\b',
    'contract_creation': r'new\s+\w+\s*\(.*?\)',
    'event_logging': r'\b(event|emit)\b',
    'state_variables': r'\b(storage|memory|calldata)\b',
    'modifiers': r'modifier\s+\w+\(.*?\)',
    'assertions': r'\b(require|assert|revert)\s*\(.*?\)',
    'external_calls': r'\.(call|delegatecall|transfer|send)\s*\(',
    'ether_transfer': r'(msg\.value|address\(.*?\)\.transfer|address\(.*?\)\.send)',
    'fallback_functions': r'function\s+(fallback|receive)\s*\(\s*\)\s*(external|public)?',

    'low_level_call': r'\.(callcode|delegatecall)\s*\(',
    'dynamic_address_handling': r'address\s*\(\s*\w+\s*\)\.',
    'time_based_logic': r'\b(block\.timestamp|now)\b',
    'complex_state_changes': r'\b(storage|mapping|array)\s+\w+\s*=',
    'unchecked_arithmetic': r'\+\+|--|\+=|-=|\*=|/=',
    'unconventional_control_flows': r'\b(goto|continue)\b',
    'unchecked_return_values': r'(\.send\(|\.call\(|\.delegatecall\()\s*;'
}

# Dictionary to map group names to numerical labels
label_map = {
    "Authentication_and_Authorization_Vulnerabilities": 1,
    "Dependency_vulnerabilities": 2,
    "Interaction_and_constract_state_vulnerabilities": 3,
    "Resource_(Gas)_usage_vulnerabilities": 4,
    "Clean": 5
}

def extract_functions(contract_code):
    function_pattern = re.compile(
        r'\bfunction\s+\w+\s*\([^)]*\)\s*(?:public|private|internal|external|view|pure|payable|constant)?\s*(?:returns\s*\(.*?\))?\s*\{[\s\S]*?\}',
        re.MULTILINE | re.DOTALL
    )
    return [match.group(0) for match in function_pattern.finditer(contract_code)]


def extract_code_block(code, start, end):
    """
    Extract the logical block of code containing the match.
    This function finds the nearest surrounding braces or semicolons.
    """
    brace_stack = []
    code_block = []
    in_block = False
    
    for i, line in enumerate(code.split('\n')):
        if start <= i <= end:
            in_block = True
        if in_block:
            code_block.append(line)
            if '{' in line:
                brace_stack.append('{')
            if '}' in line and brace_stack:
                brace_stack.pop()
            if not brace_stack and ';' in line:
                break
    return '\n'.join(code_block)


def extract_info(function_content):
    results = {}
    lines = function_content.split('\n')
    for key, pattern in patterns.items():
        matches = re.finditer(pattern, function_content, re.MULTILINE | re.DOTALL)
        context_blocks = []
        for match in matches:
            start_line = function_content[:match.start()].count('\n')
            end_line = function_content[:match.end()].count('\n')
            code_block = extract_code_block(function_content, start_line, end_line)
            flattened_block = code_block.replace('\n', ' <newline> ')
            context_blocks.append(flattened_block)
        results[key] = " <newline> ".join(context_blocks) if context_blocks else ""
    return results

def extract_functions_with_issues(contract_code):
    # Initialize the dictionary with all features
    functions_with_issues = {key: [] for key in patterns.keys()}
    
    functions = extract_functions(contract_code)
    
    for function_content in functions:
        issues = extract_info(function_content)
        for key, value in issues.items():
            if value:
                functions_with_issues[key].append({
                    "function_name": function_content.split('(')[0].split()[-1],
                    "function_content": function_content,
                    "matches": value
                })
    return functions_with_issues

def clean_for_csv(text):
    """Clean text for CSV output"""
    return str(text).replace('\n', ' ').replace('\r', '').replace('"', '""')

def process_file(file_path, csv_writer):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    functions_with_issues = extract_functions_with_issues(content)
    
    # Xác định label dựa trên thư mục con ở lớp đầu tiên
    label = "Unknown"
    for folder in label_map.keys():
        if folder in str(Path(file_path)):
            label = folder
            break
    
    # Create document for MongoDB
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
    
    # Create row for CSV
    row = [clean_for_csv(file_path), clean_for_csv(label)]
    for key in patterns.keys():
        if functions_with_issues[key]:
            row.append(clean_for_csv("; ".join([issue["matches"] for issue in functions_with_issues[key]])))
        else:
            row.append("")
    csv_writer.writerow(row)

def process_directory(directory_path, csv_writer):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".sol"):
                file_path = os.path.join(root, file)
                try:
                    process_file(file_path, csv_writer)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

# Main execution
if __name__ == "__main__":
    # Setup MongoDB connection
    client = MongoClient('mongodb://localhost:27017/')
    db = client['Data6']
    collection = db['Data']

    current_dir = Path(__file__).resolve().parent
    solidity_files_directory = r"D:\GitHub\Blockchain-Smart-Contract-Security\Dataset2"
    csv_output_file = r"D:\GitHub\Blockchain-Smart-Contract-Security\Dataset2\extracted2_data.csv"

    # Open the CSV file
    with open(csv_output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        
        # Create header
        headers = ['File Path', 'Label'] + list(patterns.keys())
        csv_writer.writerow(headers)
        
        # Process the directory
        process_directory(solidity_files_directory, csv_writer)

    print(f"Processing complete. Output saved to {csv_output_file}")