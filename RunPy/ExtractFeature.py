import re
import os
from pymongo import MongoClient
from nltk.tokenize import RegexpTokenizer

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['Interaction_and_Contract_State_Vulnerabilities']
collection = db['uncheck_send']

def check_for_reentrancy_issues(function_content):
# #reentrancy
    # vulnerability_patterns = {
    #     'external_calls': r'(?<!\w)\.(call|delegatecall|send|transfer|staticcall)\b',
    #     'use_of_msg_value_and_sender': r'\b(msg\.value|msg\.sender)\b',
    #     'recursive_calls': r'function\s+(\w+)\s*\(.*?\)\s*\{[\s\S]*?(?<=\{|;)\s*\b\1\b\s*\(.*?\)\s*;',
    #     'state_changes_after_external_calls': r'\.(call|delegatecall|send|transfer)\(.*?\)\s*;[\s\S]*?\s*\w+\s*=',
    # }
#     #uncheck external call
#     vulnerability_patterns = {
#         'external_calls': r'(?<!\w)(?:address\s*\(\s*\w+\s*\)\s*\.)?(?:send|call|delegatecall)\s*\(',
#         'ignored_return_values': r'(?<!\w)(?:address\s*\(\s*\w+\s*\)\s*\.)?(?:send|call|delegatecall)\s*\(.*?\)\s*;(?!\s*(?:require|assert|revert)\s*\()',
#         'state_changes_after_external_calls': r'(?<!\w)(?:address\s*\(\s*\w+\s*\)\s*\.)?(?:send|call|delegatecall)\s*\(.*?\)\s*;[\s\S]*?\s*\w+\s*=',
#         'lack_of_error_handling': r'(?<!\w)(?:address\s*\(\s*\w+\s*\)\s*\.)?(?:send|call|delegatecall)\s*\(.*?\)\s*;(?!\s*(?:require|assert|revert)\s*\()',
#         #'fallback_functions_without_gas_check': r'function\s+\(\)\s+(?:external\s+)?payable\s*\{(?![\s\S]*?require\s*\(\s*msg\.gas\s*>=\s*2300\s*\))'
# }
    # #delegatecall
    # vulnerability_patterns = {
    #     'usage_of_delegatecall': r'\bdelegatecall\s*\(',
    #     'state_variables_manipulation': r'\bdelegatecall\s*\([^)]*\)[^;]*;[^;]*\b\w+\s*=',
    #     'input_and_parameter_validation': r'\bdelegatecall\s*\([^)]*\)[^;]*;\s*(?!.*(?:require|assert|if)\s*\()',
    #     'context_preservation': r'\bdelegatecall\s*\([^)]*\)\s*;\s*(?!.*(?:require|assert)\s*\((?:this|msg\.sender)\s*==\s*)',
    #     'library_safe_practices': r'\blibrary\s+\w+\s*{[^{}]*?\bdelegatecall\b[^{}]*}'
    # }

    vulnerability_patterns = {
        'unchecked_send': r'\bsend\s*\(.*?\)\s*;',
        'unchecked_transfer': r'\btransfer\s*\(.*?\)\s*;',
        'msg_sender_transfer': r'\bmsg\.sender\.transfer\s*\(.*?\)\s*;'
    }

    # #timestamp dependency
    # vulnerability_patterns = {
    # 'reliance_on_block_timestamp': r'block\.timestamp',
    # 'miner_manipulation_risks': r'block\.timestamp|block\.number',
    # 'incorrect_time_estimation': r'block\.number\s*(?:\*|\/)\s*\d+',
    # 'timestamp_used_for_randomness': r'block\.timestamp.*%(?:\s*\d+)'
    # }
    
    #block number denpendency
    # vulnerability_patterns = {
    # 'dependence_on_block_number': r'block\.number',
    # 'miner_manipulation_risks': r'block\.(timestamp|number)',
    # 'usage_in_comparison_operations': r'block\.number\s*(==|!=|>|<|>=|<=)\s*<number>',
    # 'random_number_generation': r'(block\.blockhash\(block\.number\)|block\.number)'
    # }
    issues = []
    for issue, pattern in vulnerability_patterns.items():
        if re.search(pattern, function_content, re.MULTILINE | re.DOTALL):
            issues.append(issue.replace('_', ' '))
    return issues

def extract_functions_with_issues(content):
    functions_with_issues = []
    function_pattern = re.compile(r'function\s+[\w\s,]*\([^)]*\)\s*(?:returns\s*\(.*?\))?\s*\{[\s\S]*?\}', re.MULTILINE | re.DOTALL)

    for match in function_pattern.finditer(content):
        function_declaration = match.group(0)
        function_content = function_declaration.strip()

        issues = check_for_reentrancy_issues(function_content)
        if issues:
            tokenizer = RegexpTokenizer(r'\w+|\{|\}|\(|\)|\[|\]|\;|\=|\+|\-|\*|\/|\!|\%|<|>|\||\&')
            tokens = tokenizer.tokenize(function_content)

            for issue in issues:
                functions_with_issues.append({
                    "feature_type": issue,
                    "function_name": function_content.split('(')[0].split()[-1],
                    "function_content": function_content,
                    "tokens": tokens
                })

    return functions_with_issues

def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".sol"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            functions_with_issues = extract_functions_with_issues(content)
            if functions_with_issues:
                document = {
                    "filename": filename,
                    "content": content,
                    "extract_feature": functions_with_issues,
                }
                result = collection.insert_one(document)
                if result.acknowledged:
                    print(f"Processed and stored findings for {filename} in MongoDB.")
                else:
                    print(f"Failed to store findings for {filename}.")

# Specify the directory containing Solidity files
solidity_files_directory = r'D:\GitHub\Blockchain-Smart-Contract-Security\Data\Interaction and Contract State Vulnerabilities\unchecked send'

# Process the directory
process_directory(solidity_files_directory)
