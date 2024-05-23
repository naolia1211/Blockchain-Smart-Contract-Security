import re
import os
from pymongo import MongoClient
from nltk.tokenize import RegexpTokenizer

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['Interaction_and_Contract_State_Vulnerabilities']
collection = db['test1']

def preprocess_code(code):
    # Loại bỏ comment nhiều dòng (/* ... */)
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    
    # Loại bỏ comment một dòng (// ...)
    code = re.sub(r'//.*', '', code)
    
    # Loại bỏ các ký tự tab (\t)
    code = code.replace('\t', '')
    
    # Loại bỏ khoảng trắng thừa (2 hoặc nhiều khoảng trắng liên tiếp)
    code = re.sub(r'\s{2,}', ' ', code)
    
    # Loại bỏ khoảng trắng trước dấu * ở đầu dòng
    code = re.sub(r'^\s*\*', '', code, flags=re.MULTILINE)
    
    # Loại bỏ các dòng chỉ chứa khoảng trắng
    code = re.sub(r'^\s*$', '', code, flags=re.MULTILINE)
    
    return code.strip()

def check_for_reentrancy_issues(function_content):
 
   #    #reentrancy
    # vulnerability_patterns = {
    #     'external_calls': r'(?<!\w)\.(call|delegatecall|send|transfer|staticcall)\b',
    #     'use_of_msg_value_and_sender': r'\b(msg\.value|msg\.sender)\b',
    #     'recursive_calls': r'function\s+(\w+)\s*\(.*?\)\s*\{[\s\S]*?(?<=\{|;)\s*\b\1\b\s*\(.*?\)\s*;',
    #     'state_changes_after_external_calls': r'\.(call|delegatecall|send|transfer)\(.*?\)\s*;[\s\S]*?\s*\w+\s*=',
    # }
    #uncheck external call
    vulnerability_patterns = {
        'external_calls': r'(?<!\w)(?:address\s*\(\s*\w+\s*\)\s*\.)?(?:send|call|delegatecall)\s*\(',
        'ignored_return_values': r'(?<!\w)(?:address\s*\(\s*\w+\s*\)\s*\.)?(?:send|call|delegatecall)\s*\(.*?\)\s*;(?!\s*(?:require|assert|revert)\s*\()',
        'state_changes_after_external_calls': r'(?<!\w)(?:address\s*\(\s*\w+\s*\)\s*\.)?(?:send|call|delegatecall)\s*\(.*?\)\s*;[\s\S]*?\s*\w+\s*=',
        'lack_of_error_handling': r'(?<!\w)(?:address\s*\(\s*\w+\s*\)\s*\.)?(?:send|call|delegatecall)\s*\(.*?\)\s*;(?!\s*(?:require|assert|revert)\s*\()',
        #'fallback_functions_without_gas_check': r'function\s+\(\)\s+(?:external\s+)?payable\s*\{(?![\s\S]*?require\s*\(\s*msg\.gas\s*>=\s*2300\s*\))'
}
#     #delegatecall
#     vulnerability_patterns = {
#     'usage_of_delegatecall': r'(?<!\w)delegatecall\s*\(',
#     'state_variables_manipulation': r'\.delegatecall\s*\((?:[^;\n])*\)[^;\n]*;[^;\n]*\s*\w+\s*=',
#     'input_and_parameter_validation': r'(?<!\w)delegatecall\s*\((?!.*(?:require|assert|if)\s*\()',
#     #'context_preservation': r'(?<!\/\/.*)\bdelegatecall\s*\(\s*[^)]*?\s*\)\s*;(?!\s*(?:require|assert)\s*\((?:this|msg\.sender)\s*==\s*)?',
#     #'library_safe_practices': r'library\s+\w+\s*{(?:[^{}]*(?:{(?:[^{}]*(?:{[^{}]*})*[^{}]*)*})*[^{}]*)*?\bdelegateCall\b(?:[^{}]*(?:{(?:[^{}]*(?:{[^{}]*})*[^{}]*)*})*[^{}]*)*}'
# }

#     #unchecked send
#     vulnerability_patterns = {
#     'usage_of_send': r'(?<!\\)\bsend\s*\(',
#     'unchecked_send_return': r'\bsend\s*\(.*?\)\s*;(?!\s*(?:require|assert|revert|if)\s*\(.*?\)\s*;)',
#     'state_update_without_verification': r'\bsend\s*\(.*?\)\s*;[\s\S]*?\s*(?:\w+\s*=|(?:require|assert|revert|if)\s*\()',
#     #'fallback_function_risks': r'function\s+\(\)\s+(?:external\s+)?payable\s*\{(?!.*(?:require|assert|revert)\s*\(msg\.value\s*(?:<=|<)\s*2300\))[\s\S]*?\bsend\b'
# }
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
    function_pattern = re.compile(r'function\s+(\w+)\s*\(.*?\)\s*\{(.*?)\}', re.DOTALL)

    for match in function_pattern.finditer(content):
        function_name = match.group(1)
        function_content = match.group(2).strip()

        issues = check_for_reentrancy_issues(function_content)
        if issues:
            # Tokenize function_content
            tokenizer = RegexpTokenizer(r'\w+|\{|\}|\(|\)|\\[|\\]|\.|\;|\=|\+|\-|\*|\/|\!|\%|<|>|\||\&')
            tokens = tokenizer.tokenize(function_content)

            # Modify here to get the desired output format
            for issue in issues:
                functions_with_issues.append({
                    "feature_type": issue,
                    "function_name": function_name,
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
            preprocessed_content = preprocess_code(content)
            functions_with_issues = extract_functions_with_issues(preprocessed_content)
            if functions_with_issues:
                document = {
                    "filename": filename,
                    "content": preprocessed_content,
                    "extract_feature": functions_with_issues,
                }
                collection.insert_one(document)
                print(f"Processed and stored findings for {filename} in MongoDB.")

# Specify the directory containing Solidity files
solidity_files_directory = r'C:\Users\hao30\Documents\GitHub\Blockchain-Smart-Contract-Security\Dataset\unchecked external call (UC)\source'

# Process the directory
process_directory(solidity_files_directory)