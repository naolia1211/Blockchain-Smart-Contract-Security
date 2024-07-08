import os
import sys
import re
import csv
from pymongo import MongoClient
from pathlib import Path
import chardet
from tqdm import tqdm

patterns = {
    # Black List (Known Vulnerabilities)
    # Interaction and Contract State Vulnerabilities.
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

    # Dependency Vulnerabilities
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

    # Authentication and Authorization Vulnerabilities
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

    # Resource (Gas) Usage Vulnerabilities
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
    
    # Arithmetic Vulnerabilities
    'integer_overflow_underflow': r'(uint|int)(\d+)?\s+\w+\s*=\s*[\w\d]+\s*;',
    'unchecked_arithmetic_vuln': r'(?<!SafeMath\.)(?<!require\()(?<!assert\()(\w+\s*[\+\-\*\/\%]\s*[\w\d]+)',
    'division_precision_loss': r'(\w+)\s*\/\s*(\w+)',
    'increment_decrement_overflow': r'(\w+)\s*(\+\+|\-\-)',
    'unsafe_assignment': r'(\w+)\s*(\+=|\-=|\*=|\/=)',
    'unsafe_integer_types': r'\b(u?int(?!256)(?:\d{1,2}|[1-9]\d{2,}))\b',
    'unchecked_bounds': r'(?<!require\()(?<!assert\()(\w+\s*[\+\-\*\/]\s*\w+)',
    'modulo_bias': r'(\w+)\s*\%\s*(\w+)',
    'multiplication_overflow': r'(\w+)\s*\*\s*(\w+)',
    'safemath_missing': r'function\s+(\w+)\s*\([^\)]*\)\s*[^\{]*\{(?!.*SafeMath)',
    'floating_point_approximation': r'(\w+)\s*\/\s*(\w+)\s*\*\s*(\d+)',
    'unsafe_type_casting': r'(uint|int)(\d+)?\((\w+)\)',
    'unchecked_array_length': r'(\w+)\.length\s*[\+\-\*\/]',
    'implicit_type_conversion': r'(uint|int)(\d+)?\s+\w+\s*=\s*\w+',
    'arithmetic_in_loop_condition': r'(for|while)\s*\([^)]*[\+\-\*\/][^)]*\)',
    'division_by_zero': r'(\w+)\s*\/\s*0',
    'precision_loss_in_division': r'(\w+)\s*\/\s*(\w+)\s*\*\s*(\w+)',
    'unsafe_exponentiation': r'(\w+)\s*\*\*\s*(\w+)',
    'unchecked_math_function': r'\b(abs|addmod|mulmod)\s*\(',
    'bitwise_operations': r'(\w+)\s*(\&|\||\^|<<|>>)\s*(\w+)',
    'unsafe_math_functions': r'\b(addmod|mulmod)\s*\(',
    'ternary_arithmetic': r'(\w+)\s*\?\s*[\w\d]+\s*:\s*[\w\d]+',

    # Additional Keywords and Patterns for Comprehensive Analysis
    'arithmetic_operations': r'\b(\+|\-|\*|\/)\b',
    'contract_creation': r'new\s+\w+\s*\(.*?\)',
    'event_logging': r'\b(event|emit)\b',
    'state_variables': r'\b(storage|memory|calldata)\b',
    'modifiers': r'modifier\s+\w+\(.*?\)',
    'assertions': r'\b(require|assert|revert)\s*\(.*?\)',
    'external_calls': r'\.(call|delegatecall|transfer|send)\s*\(',
    'ether_transfer': r'(msg\.value|address\(.*?\)\.transfer|address\(.*?\)\.send)',
    'fallback_functions': r'function\s+(fallback|receive)\s*\(\s*\)\s*(external|public)?',

    # Grey List (Potential Unknown Vulnerabilities)
    'low_level_call_code': r'\.(callcode|delegatecall)\s*\(',
    'dynamic_address_handling': r'address\s*\(\s*\w+\s*\)\.',
    'time_based_logic': r'\b(block\.timestamp|now)\b',
    'complex_state_changes': r'\b(storage|mapping|array)\s+\w+\s*=',
    'unchecked_arithmetic': r'\+\+|--|\+=|-=|\*=|/=',
    'unchecked_return_values': r'(\.send\(|\.call\(|\.delegatecall\()\s*;',
    'unconventional_control_flows': r'\b(goto|continue)\b'
}

def extract_functions(contract_code):
    function_pattern = re.compile(
        r'\bfunction\s+\w+\s*\([^)]*\)\s*(?:public|private|internal|external|view|pure|payable|constant)?\s*(?:returns\s*\(.*?\))?\s*\{[\s\S]*?\}',
        re.MULTILINE | re.DOTALL
    )
    return [match.group(0) for match in function_pattern.finditer(contract_code)]

def extract_code_block(code, start, end):
    brace_stack = []
    code_block = []
    in_block = False
#
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

    code_block_str = '\n'.join(code_block)
    cleaned_code_block = re.sub(r'\n{2,}', '\n', code_block_str)
    return cleaned_code_block

def extract_info(function_content):
    results = {}
    for key, pattern in patterns.items():
        matches = re.finditer(pattern, function_content, re.MULTILINE | re.DOTALL)
        context_blocks = []
        for match in matches:
            start_line = function_content[:match.start()].count('\n')
            end_line = function_content[:match.end()].count('\n')
            code_block = extract_code_block(function_content, start_line, end_line)
            flattened_block = re.sub(r'\n{2,}', '\n', code_block)
            flattened_block = flattened_block.replace('\n', ' <newline> ')
            flattened_block = re.sub(r'( <newline> )+', ' <newline> ', flattened_block)
            flattened_block = re.sub(r'^\s+', '', flattened_block, flags=re.MULTILINE)
            flattened_block = re.sub(r' +', ' ', flattened_block)
            flattened_block = re.sub(r'\s*;\s*', ';', flattened_block)
            flattened_block = re.sub(r';+', ';', flattened_block)
            flattened_block = re.sub(r';\s+', ';', flattened_block)
            context_blocks.append(flattened_block)

        if context_blocks:
            combined_blocks = " <newline> ".join(context_blocks)
            combined_blocks = combined_blocks.strip(' <newline> ')
            combined_blocks = re.sub(r'( <newline> )+', ' <newline> ', combined_blocks)
            combined_blocks = combined_blocks.replace('\n', ' <newline> ')
            results[key] = combined_blocks
        else:
            results[key] = "na"

    return results

def extract_functions_with_issues(contract_code):
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

def process_file(file_path, csv_writer, root_dir, collection):
    try:
        # Attempt to detect the file encoding
        with open(file_path, 'rb') as raw_file:
            raw_content = raw_file.read()
            detected_encoding = chardet.detect(raw_content)['encoding']

        # Try to read the file with the detected encoding
        with open(file_path, 'r', encoding=detected_encoding) as file:
            content = file.read()
    except UnicodeDecodeError:
        # If detection fails, try with a more lenient encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        except Exception as e:
            print(f"Failed to read file {file_path}: {str(e)}")
            return  # Skip this file

    # Rest of the function remains the same
    functions_with_issues = extract_functions_with_issues(content)
    
    relative_path = os.path.relpath(file_path, root_dir)
    vulnerability_label = relative_path.split(os.sep)[0]
    
    document = {
        "File Name": os.path.basename(file_path),
        "content": content,
        "functions_with_issues": functions_with_issues,
        "label": vulnerability_label
    }
    result = collection.insert_one(document)
    if result.acknowledged:
        print(f"Extracting file {vulnerability_label}.")
    else:
        print(f"Failed to store findings for {file_path}.")
    
    row = [os.path.basename(file_path), vulnerability_label]
    for key in patterns.keys():
        if functions_with_issues.get(key):
            matches = "; ".join([re.sub(r'\s+', ' ', issue["matches"]).strip() for issue in functions_with_issues[key]])
            row.append(matches)
        else:
            row.append("")

    cleaned_row = [re.sub(r'\s+', ' ', item).strip() for item in row]
    csv_writer.writerow(cleaned_row)
    
def process_directory(directory_path, csv_output_file, mongo_uri, db_name, collection_name):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    with open(csv_output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        headers = ['File Name', 'Vulnerability Label'] + list(patterns.keys())
        csv_writer.writerow(headers)
        
        # Đếm số lượng tệp .sol để sử dụng cho thanh tiến trình
        total_files = sum([len(files) for r, d, files in os.walk(directory_path) if any(f.endswith('.sol') for f in files)])

        with tqdm(total=total_files, desc="Processing files") as pbar:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if file.endswith(".sol"):
                        process_file(os.path.join(root, file), csv_writer, directory_path, collection)
                        pbar.update(1)  # Cập nhật thanh tiến trình sau mỗi lần xử lý tệp

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process Solidity files and extract vulnerabilities.")
    parser.add_argument('--solidity_files_directory', type=str, required=True, help="Path to the directory containing Solidity files.")
    parser.add_argument('--csv_output_file', type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument('--mongo_uri', type=str, required=True, help="MongoDB connection URI.")
    parser.add_argument('--db_name', type=str, required=True, help="MongoDB database name.")
    parser.add_argument('--collection_name', type=str, required=True, help="MongoDB collection name.")

    args = parser.parse_args()

    process_directory(args.solidity_files_directory, args.csv_output_file, args.mongo_uri, args.db_name, args.collection_name)
