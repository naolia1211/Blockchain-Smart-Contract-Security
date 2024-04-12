import re
import os
from pymongo import MongoClient

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['database']
collection = db['block number dependency']

def preprocess_code(code):
    # Standard preprocessing to clean up the source code
    code = re.sub(re.compile(r"/\*.*?\*/", re.DOTALL), "", code)
    code = re.sub(re.compile(r"//.*?$", re.MULTILINE), "", code)
    code = re.sub(r'\d+', '<number>', code)
    code = re.sub(r'".*?"', '<string>', code)
    code = re.sub(r'\t', '    ', code)
    code = re.sub(r' {2,}', ' ', code)
    code = re.sub(r' *\n', '\n', code)
    code = re.sub(r'\n\s*\n', '\n', code)
    return code.strip()

def check_for_reentrancy_issues(function_content):
    # #reentrancy
    # vulnerability_patterns = {
    #     'external_calls': r'\.(call|delegatecall|send|transfer)\b',
    #     'state_changes_after_external_calls': r'\.(call|delegatecall|send|transfer)\(.*\);.*=',
    #     'recursive_calls': r'function\s+(\w+)\s*\(.*?\)\s*{[\s\S]*?\b\1\b\s*\(.*?\);',
    #     'fallback_functions': r'function\s*\(\)\s*external\s*payable\s*\{[^}]*\}'
    # }
    # #uncheck external call
    # vulnerability_patterns = {
    # 'external_calls': r'\b(address\.|msg\.sender\.)?(send|call|delegatecall)\s*\(',
    # 'ignored_return_values': r'\b(address\.|msg\.sender\.)?(send|call|delegatecall)\s*\([^)]*\);',
    # 'state_changes_after_external_calls': r'\b(address\.|msg\.sender\.)?(send|call|delegatecall)\s*\([^)]*\);.*=.*;',
    # 'lack_of_error_handling': r'\b(address\.|msg\.sender\.)?(send|call|delegatecall)\s*\([^)]*\);(?!\s*(require|revert|assert)\s*\()',
    # 'fallback_functions_without_gas_check': r'function\s*\(\)\s*(external\s*)?payable\s*{\s*if\s*\(msg\.gas\s*<\s*2300\)'
    # }
    #delegatecall
    # vulnerability_patterns = {
    # 'usage_of_delegatecall': r'delegatecall\(',
    # 'state_variables_manipulation': r'\.delegatecall\(.*\);.*=',
    # 'input_and_parameter_validation': r'\.delegatecall\((?!.*require\(|.*assert\(|.*if.*\()',
    # 'context_preservation': r'\.delegatecall\([^)]*\);(?!\s*this\.)',
    # 'library_safe_practices': r'library\s+\w+\s*{[^}]*\.delegatecall\('
    # }
    # #unchecked send
    # vulnerability_patterns = {
    # 'usage_of_send': r'\.send\(',
    # 'unchecked_send_return': r'\.send\(.*\);(?!\s*if\s*\(|\s*require\()',
    # 'state_update_without_verification': r'\.send\(.*\);.*=.*;',
    # 'fallback_function_risks': r'function\s*\(\)\s*(external\s*)?payable\s*{[^}]*\.send\('
    # }
    # #timestamp dependency
    # vulnerability_patterns = {
    # 'reliance_on_block_timestamp': r'block\.timestamp',
    # 'miner_manipulation_risks': r'block\.timestamp|block\.number',
    # 'incorrect_time_estimation': r'block\.number\s*(?:\*|\/)\s*\d+',
    # 'timestamp_used_for_randomness': r'block\.timestamp.*%(?:\s*\d+)'
    # }
    #block number denpendency
    vulnerability_patterns = {
    'dependence_on_block_number': r'block\.number',
    'miner_manipulation_risks': r'block\.(timestamp|number)',
    'usage_in_comparison_operations': r'block\.number\s*(==|!=|>|<|>=|<=)\s*<number>',
    'random_number_generation': r'(block\.blockhash\(block\.number\)|block\.number)'
    }



    issues = []
    for issue, pattern in vulnerability_patterns.items():
        if re.search(pattern, function_content, re.MULTILINE | re.DOTALL):
            issues.append(issue.replace('_', ' '))
    return issues

def extract_contracts_and_functions(content):
    contracts_functions = {}
    current_contract = None
    function_block = []
    brace_count = 0
    inside_function = False

    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if 'contract' in line and '{' in line:  # Check if it's a contract definition line
            current_contract = re.search(r'contract\s+(\w+)', line)
            if current_contract:
                current_contract = current_contract.group(1)
                contracts_functions[current_contract] = []

        if current_contract:
            if 'function' in line and '{' in line:  # Function starts
                inside_function = True
                brace_count = 1
                function_block = [line]
            elif inside_function:
                function_block.append(line)
                if '{' in line:
                    brace_count += line.count('{')
                if '}' in line:
                    brace_count -= line.count('}')
                if brace_count == 0:  # Function ends
                    inside_function = False
                    function_text = '\n'.join(function_block)
                    function_name_match = re.search(r'function\s+([\w]+)', function_text)
                    if function_name_match:
                        function_name = function_name_match.group(1)
                        contracts_functions[current_contract].append((function_name, function_text))
                    function_block = []

    return contracts_functions


def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".sol"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            preprocessed_content = preprocess_code(content)
            contracts_functions = extract_contracts_and_functions(preprocessed_content)
            for contract, functions in contracts_functions.items():
                for function_name, function_content in functions:
                    issues = check_for_reentrancy_issues(function_content)
                    if issues:  # Only save if there are issues
                        document = {
                            "filename": filename,
                            "content": preprocessed_content,
                            "extract_feature": f"{contract}.{function_name}\n{function_content}",
                            "issues": issues
                        }
                        collection.insert_one(document)
                        print(f"Processed and stored findings for {contract}.{function_name} in MongoDB.")

# Specify the directory containing Solidity files
solidity_files_directory = r'D:\GitHub\Blockchain-Smart-Contract-Security\Dataset\block number dependency'

# Process the directory
process_directory(solidity_files_directory)
