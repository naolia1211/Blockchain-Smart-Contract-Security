import os
import re
import csv
from tqdm import tqdm

def extract_info(contract_code):
    patterns = {
        # Interaction and Contract State Vulnerabilities
        'unsafe_external_call_function': r'\.(call|send|delegatecall|transfer)\s*\(',
        'state_changes_after_external_calls': r'\.(call|send|delegatecall|transfer)\s*\(.*?\)\s*;',
        'error_handling': r'\b(require|revert|assert)\s*\(.*?\)',
        'fallback_function_interaction': r'function\s+fallback\s*\(\s*\)\s*(external|public)?',
        'ignored_return_value': r'\b(call|send|delegatecall|transfer)\b\s*\(.*?\)\s*;',
        'state_variables_manipulation': r'\b\w+\s*=\s*.*?;',
        'recursive_calls': r'\bfunction\s+(\w+)\s*\(.*?\)\s*{(?:(?!\bfunction\b).)*\b\1\s*\(',
        'use_of_msg_value_or_sender': r'\b(msg\.value|msg\.sender)\b',
        'use_of_delegatecall': r'\bdelegatecall\s*\(',
        'library_safe_practices': r'\blibrary\s+\w+',
        'input_and_parameter_validation': r'\brequire\s*\(.*?\)',
        'gas_limitations_in_fallback_functions': r'function\s+fallback\s*\(\s*\)\s*(external|public)?\s*{(?:(?!\bgasleft\s*\(\s*\)).)*(gasleft\s*\(\s*\).*?;)',
        # Dependency Vulnerabilities
        'usage_of_block_timestamp': r'\bblock\.timestamp\b',
        'usage_of_block_number': r'\bblock\.number\b',
        'usage_of_block_blockhash': r'\bblock\.blockhash\s*\(',
        'miner_manipulation': r'\b(block\.timestamp|block\.number)\b',
        'usage_in_comparison_operations': r'\b(block\.number)\s*(==|!=|>|<|>=|<=)',
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
        # Resource (Gas) Usage Vulnerabilities   
        'dependency_on_function_order': r'function\s+\w+\s*\(.*?\)\s*{(?:(?!\bfunction\b).)*\b\w+\s*\(',
        'high_gas_price_transactions': r'\.gas\s*\(\s*\d+\s*\)\.',
        'require_statements': r'\brequire\s*\(.*?\)',
        'state_variables_manipulation': r'\b(storage|mapping|array)\b',
        'concurrent_function_invocations': r'function\s+\w+\s*\(.*?\)\s*{(?:(?!\bfunction\b).)*\b\w+\s*\(',
        'gas_limit_checks': r'\bgasleft\s*\(\s*\)',
        'state_changing_operations': r'\b(storage|mapping|array)\b',
        'recursive_function_calls': r'\bfunction\s+(\w+)\s*\(.*?\)\s*{(?:(?!\bfunction\b).)*\b\1\s*\(',
        'high_complexity_loops': r'\b(for|while)\s*\(.*?\)\s*{',
    }

    results = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, contract_code, re.MULTILINE | re.DOTALL)
        # Join matches properly
        if matches:
            if isinstance(matches[0], tuple):
                matches = [''.join(match) for match in matches]
            results[key] = "\n".join(matches)
        else:
            results[key] = ""

    return results

def process_file(filepath, writer):
    try:
        with open(filepath, 'r') as file:
            contract_code = file.read()
            info = extract_info(contract_code)
            write_results(info, filepath, writer)

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

def write_results(info, filepath, writer):
    filename = os.path.basename(filepath)
    writer.writerow([filename, info.get('unsafe_external_call_function', ''), 
                     info.get('state_changes_after_external_calls', ''), 
                     info.get('error_handling', ''), 
                     info.get('fallback_function_interaction', ''), 
                     info.get('ignored_return_value', ''), 
                     info.get('state_variables_manipulation', ''), 
                     info.get('recursive_calls', ''), 
                     info.get('use_of_msg_value_or_sender', ''), 
                     info.get('use_of_delegatecall', ''), 
                     info.get('library_safe_practices', ''), 
                     info.get('input_and_parameter_validation', ''), 
                     info.get('gas_limitations_in_fallback_functions', ''), 
                     info.get('usage_of_block_timestamp', ''), 
                     info.get('usage_of_block_number', ''), 
                     info.get('usage_of_block_blockhash', ''), 
                     info.get('miner_manipulation', ''), 
                     info.get('usage_in_comparison_operations', ''), 
                     info.get('lack_of_private_seed_storage', ''), 
                     info.get('historical_block_data', ''), 
                     info.get('predictability_of_randomness_sources', ''), 
                     info.get('insufficient_input_and_parameter_validation', ''), 
                     info.get('owner_variable', ''), 
                     info.get('modifier_usage', ''), 
                     info.get('function_visibility', ''), 
                     info.get('authentication_checks', ''), 
                     info.get('state_changing_external_calls', ''), 
                     info.get('selfdestruct_usage', ''), 
                     info.get('authorization_checks_using_tx_origin', ''), 
                     info.get('dependency_on_function_order', ''), 
                     info.get('high_gas_price_transactions', ''), 
                     info.get('require_statements', ''), 
                     info.get('concurrent_function_invocations', ''), 
                     info.get('gas_limit_checks', ''), 
                     info.get('state_changing_operations', ''), 
                     info.get('recursive_function_calls', ''), 
                     info.get('high_complexity_loops', '')])

def main():
    contract_directory = r"D:\GitHub\Blockchain-Smart-Contract-Security\Data\BigData"
    output_file = "vulnerability_group_extraction_results.csv"

    # Recursively find all .sol files
    file_list = []
    for root, dirs, files in os.walk(contract_directory):
        for file in files:
            if file.endswith(".sol"):
                file_list.append(os.path.join(root, file))

    with open(output_file, 'w', newline='') as file:
        fieldnames = ["Contract", "Unsafe External Call Function", "State Changes After External Calls",
                      "Error Handling", "Fallback Function Interaction",
                      "Ignored Return Value", "State Variables Manipulation",
                      "Recursive Calls", "Use of msg.value or msg.sender",
                      "Use of delegatecall", "Library Safe Practices",
                      "Input and Parameter Validation", "Gas Limitations in Fallback Functions",
                      "Usage of block.timestamp", "Usage of block.number",
                      "Usage of block.blockhash", "Miner Manipulation",
                      "Usage in Comparison Operations", "Lack of Private Seed Storage",
                      "Historical Block Data", "Predictability of Randomness Sources",
                      "Insufficient Input and Parameter Validation", "Owner Variable",
                      "Modifier Usage", "Function Visibility",
                      "Authentication Checks", "State-changing External Calls",
                      "Selfdestruct Usage", "Authorization Checks using tx.origin",
                      "Dependency on Function Order", "High Gas Price Transactions",
                      "Require Statements", "Concurrent Function Invocations",
                      "Gas Limit Checks", "State-changing Operations",
                      "Recursive Function Calls", "High Complexity Loops"]
        writer = csv.writer(file)
        writer.writerow(fieldnames)

        for filepath in tqdm(file_list, desc="Processing files"):
            process_file(filepath, writer)

    print("Vulnerability group extraction completed. Results saved to", output_file)

if __name__ == '__main__':
    main()
