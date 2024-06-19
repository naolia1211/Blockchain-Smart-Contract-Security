import os
import re
import csv
from tqdm import tqdm

def extract_info(contract_code):
    patterns = {
        'function_calls': r'function\s+\w+\s*\([^)]*\)\s*(?:internal|external|public|private|payable|pure|view|constant|virtual|override)?\s*(?:internal|external|public|private|payable|pure|view|constant|virtual|override)?\s*(?:returns\s*\((?:[^()]+|\([^()]*\))*\))?\s*\{[\s\S]*?\}',
        'external_calls': r'(?:\w+(?:\[\w+\])?\.)?(?:delegatecall|call|staticcall|send|transfer)\s*(?:\.gas\(\w+\)|\.value\(\w+\))?\(.*?\);',
        'loops': r'(?:for|while)\s*\((?:[^()]+|\([^()]*\))*\)\s*(?:\{[\s\S]*?\}|\s*;)',
        'function_callbacks': r'function\s*\((?:[^()]+|\([^()]*\))*\)\s*(?:external|public)?\s*(?:payable)?\s*(?:\{[\s\S]*?\})?',
        'reentrancy': r'(?:call|delegatecall|callcode)\s*\(\s*.*?(?:value|gas)\s*\(\s*.*?\s*\)',
        'integer_overflows': r'\b(?:\+\+|--|\+=|-=|\*=|/=|%=|=\s*\w+\s*\+\s*\w+|\=\s*\w+\s*\-\s*\w+|\=\s*\w+\s*\*\s*\w+|\=\s*\w+\s*\/\s*\w+|\=\s*\w+\s*\%\s*\w+)\b',
        'unchecked_low_level_calls': r'\.\s*(?:call|delegatecall|callcode)\s*\(\s*(?:\w*\s*\.\s*value\(\w*\)\s*)?\)',
        'uninitialized_storage': r'\b(?:uint|int|string|address|bool|bytes)\s+\w+\s*;\s*(?!\=)',
        'timestamp_dependence': r'\bblock\.timestamp\b|\bnow\b',
        'insecure_imports': r'import\s+"[^"]+";',
        'unsafe_inheritance': r'contract\s+\w+\s+is\s+[^;]+',
        'modifiers': r'modifier\s+\w+\s*\([^)]*\)\s*\{[\s\S]*?\}',
        'state_variables': r'\b(?:uint|int|string|address|bool|bytes)\s+\w+\s*(?:=\s*[^;]+)?;',
        'insecure_interface_implementations': r'contract\s+\w+\s+is\s+[^;]+',
        'fallback_functions': r'fallback\s*\(\s*\)\s*(?:external|public)?\s*(?:payable)?\s*(?:\{[\s\S]*?\})?',
        'selfdestruct_functions': r'\bselfdestruct\s*\(\s*(?:address\s*\(\s*\))?(?:\s*[\.\w]+\s*)*\)',
        'delegatecall_usage': r'\bdelegateCall\s*\(\s*(?:bytes memory|string memory)?\s*.*?\)',
        'default_visibilities': r'\b(?:function|constructor|state variable)\s+\w+\s*\((.*?)\)\s*(?!\bexternal\b|\bpublic\b|\binternal\b|\bprivate\b)',
        'dos_patterns': r'for\s*\(.*?\)\s*\{(?:(?!\})[\s\S])*\}',
        'insecure_randomness_usage': r'\b(?:keccak256|sha256|ripemd160|ecrecover|addmod|mulmod|block\.timestamp|block\.number|block\.difficulty|block\.gaslimit|blockhash|msg\.sender)\s*\(',
        'parameter_ordering': r'function\s+\w+\s*\(.*?\)',
        'tod_patterns': r'(?:call|delegatecall|callcode)\s*\(\s*.*?(?:value|gas)\s*\(\s*.*?\s*\)',
        'tx_origin_usage': r'\btx\.origin\b',
        'block_number_dependence': r'\bblock\.number\b',
        'underflows': r'\b(?:\+\+|--|\+=|-=|\*=|/=|%=|=\s*\w+\s*\+\s*\w+|\=\s*\w+\s*\-\s*\w+|\=\s*\w+\s*\*\s*\w+|\=\s*\w+\s*\/\s*\w+|\=\s*\w+\s*\%\s*\w+)\b',
        # Interaction and Contract State Vulnerabilities
        'unsafe_external_call_function': r'\.(call|send|delegatecall|transfer)',
        'state_changes_after_external_calls': r'\.(call|send|delegatecall|transfer).*?\;',
        'error_handling': r'\b(require|revert|assert)\b',
        'fallback_function_interaction': r'function\s+fallback\s*\(',
        'ignored_return_value': r'\b(call|send|delegatecall|transfer)\b(?!.*?\b(bool|boolean)\b)',
        'state_variables_manipulation': r'\b\w+\s*=\s*.*?\b',
        'recursive_calls': r'\bfunction\b\s+\w+\s*\(.*\)\s*{.*\b\1\s*\(.*\).*}',
        'use_of_msg_value_or_sender': r'\b(msg\.value|msg\.sender)\b',
        'use_of_delegatecall': r'\bdelegatecall\b',
        'library_safe_practices': r'\blibrary\b',
        'input_and_parameter_validation': r'\b(require\(.*?\))\b',
        'gas_limitations_in_fallback_functions': r'function\s+fallback\s*\(.*\)\s*{.*?\bgasleft\(\)\b.*?}',
        # Dependency Vulnerabilities
        'usage_of_block_timestamp': r'\bblock\.timestamp\b',
        'usage_of_block_number': r'\bblock\.number\b',
        'usage_of_block_blockhash': r'\bblock\.blockhash\s*\(',
        'miner_manipulation': r'\b(block\.timestamp|block\.number)\b',
        'usage_in_comparison_operations': r'\b(block\.number)\s*(==|!=|>|<|>=|<=)',
        'lack_of_private_seed_storage': r'\bprivate\b.*\bseed\b',
        'historical_block_data': r'\b(block\.number|block\.timestamp|block\.blockhash)\b',
        'predictability_of_randomness_sources': r'\b(block\.timestamp|block\.blockhash|block\.number)\b',
        # Authentication and Authorization Vulnerabilities
        'insufficient_input_and_parameter_validation': r'\b(require|assert|revert)\s*\(.*\)',
        'owner_variable': r'\baddress\s+public\s+owner\b',
        'modifier_usage': r'\bmodifier\s+(onlyOwner|onlyAuthorized)\b',
        'function_visibility': r'\b(public|external|internal|private)\b',
        'authentication_checks': r'require\s*\(\s*msg\.sender\s*==\s*owner\s*\)|tx\.origin',
        'state_changing_external_calls': r'\b(address|address payable)\.call\.(value|gas)\b',
        'selfdestruct_usage': r'selfdestruct\s*\(\s*msg\.sender\s*\)',
        'authorization_checks_using_tx_origin': r'require\s*\(\s*tx\.origin\s*==\s*\w+\s*\)',
        # Resource (Gas) Usage Vulnerabilities   
        'dependency_on_function_order': r'function\s+\w+\s*\([^)]*\)\s*{[^}]*\b\w+\s*\(',
        'high_gas_price_transactions': r'transaction with high gas price',
        'require_statements': r'require\s*\(\s*.*\s*\)',
        'concurrent_function_invocations': r'function\s+\w+\s*\([^)]*\)\s*{[^}]*\b\w+\s*\(',
        'gas_limit_checks': r'\bgasleft\s*\(\s*\)\b',
        'state_changing_operations': r'\b(storage|mapping|array)\b',
        'recursive_function_calls': r'function\s+\w+\s*\([^)]*\)\s*{[^}]*\b\w+\s*\(',  
        'high_complexity_loops': r'\b(for|while)\s*\(.*\)\s*{',
    }

    results = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, contract_code, re.MULTILINE | re.DOTALL)
        results[key] = "\n".join(matches)

    return results

def process_file(filepath, writer):
    try:
        with open(filepath, 'r') as file:
            contract_code = ''
            for line in file:
                contract_code += line
                if len(contract_code) >= 1024 * 1024:  # Xử lý khi đọc đủ 1MB
                    info = extract_info(contract_code)
                    write_results(info, filepath, writer)
                    contract_code = ''  # Đặt lại contract_code sau khi xử lý

            # Xử lý phần còn lại của contract_code (nếu có)
            if contract_code:
                info = extract_info(contract_code)
                write_results(info, filepath, writer)

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

def write_results(info, filepath, writer):
    filename = os.path.basename(filepath)
    writer.writerow([filename, info['function_calls'], info['external_calls'],
                     info['loops'], info['function_callbacks'], info['reentrancy'],
                     info['integer_overflows'], info['unchecked_low_level_calls'],
                     info['uninitialized_storage'], info['timestamp_dependence'],
                     info['insecure_imports'], info['unsafe_inheritance'],
                     info['modifiers'], info['state_variables'],
                     info['insecure_interface_implementations'], info['fallback_functions'],
                     info['selfdestruct_functions'], info['delegatecall_usage'],
                     info['default_visibilities'], info['dos_patterns'],
                     info['insecure_randomness_usage'], info['parameter_ordering'],
                     info['tod_patterns'], info['tx_origin_usage'],
                     info['block_number_dependence'], info['underflows'],
                     info['unsafe_external_call_function'], info['state_changes_after_external_calls'],
                     info['error_handling'], info['fallback_function_interaction'],
                     info['ignored_return_value'], info['state_variables_manipulation'],
                     info['recursive_calls'], info['use_of_msg_value_or_sender'],
                     info['use_of_delegatecall'], info['library_safe_practices'],
                     info['input_and_parameter_validation'], info['gas_limitations_in_fallback_functions'],
                     info['usage_of_block_timestamp'], info['usage_of_block_number'],
                     info['usage_of_block_blockhash'], info['miner_manipulation'],
                     info['usage_in_comparison_operations'], info['lack_of_private_seed_storage'],
                     info['historical_block_data'], info['predictability_of_randomness_sources'],
                     info['insufficient_input_and_parameter_validation'], info['owner_variable'],
                     info['modifier_usage'], info['function_visibility'],
                     info['authentication_checks'], info['state_changing_external_calls'],
                     info['selfdestruct_usage'], info['authorization_checks_using_tx_origin'],
                     info['dependency_on_function_order'], info['high_gas_price_transactions'],
                     info['require_statements'], info['concurrent_function_invocations'],
                     info['gas_limit_checks'], info['state_changing_operations'],
                     info['recursive_function_calls'], info['high_complexity_loops']])

def main():
    contract_directory = r"D:\GitHub\Blockchain-Smart-Contract-Security\Data\Interaction and Contract State Vulnerabilities\Unchecked_external_call\source"
    output_file = "enhanced_extraction_Unchecked_external_call_results.csv"

    file_list = [os.path.join(contract_directory, f) for f in os.listdir(contract_directory) if f.endswith(".sol")]

    with open(output_file, 'w', newline='') as file:
        fieldnames = ["Contract", "Function Calls", "External Calls", "Loops",
                      "Function Callbacks", "Reentrancy", "Integer Overflows",
                      "Unchecked Low-Level Calls", "Uninitialized Storage",
                      "Timestamp Dependence", "Insecure Imports", "Unsafe Inheritance",
                      "Modifiers", "State Variables", "Insecure Interface Implementations",
                      "Fallback Functions", "Self-destruct Functions", "Delegatecall Usage",
                      "Default Visibilities", "DoS Patterns", "Insecure Randomness Usage",
                      "Parameter Ordering", "Transaction Order Dependence", "Tx.origin Usage",
                      "Block Number Dependence", "Underflows",
                      "Unsafe External Call Function", "State Changes After External Calls",
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

    print("Enhanced extraction completed. Results saved to", output_file)

if __name__ == '__main__':
    main()