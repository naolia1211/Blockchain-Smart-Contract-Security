import re
import os
from pymongo import MongoClient
from nltk.tokenize import RegexpTokenizer

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['Interaction_and_Contract_State_Vulnerabilities']
collection = db['uncheck_send']

def check_for_reentrancy_issues(function_content):
    vulnerability_patterns = {
        'usage_of_delegatecall': r'\bdelegatecall\s*\(',
        'state_variables_manipulation': r'\bdelegatecall\s*\([^)]*\)[^;]*;[^;]*\b\w+\s*=',
        'input_and_parameter_validation': r'\bdelegatecall\s*\([^)]*\)[^;]*;\s*(?!.*(?:require|assert|if)\s*\()',
        'context_preservation': r'\bdelegatecall\s*\([^)]*\)\s*;\s*(?!.*(?:require|assert)\s*\((?:this|msg\.sender)\s*==\s*)',
        'library_safe_practices': r'\blibrary\s+\w+\s*{[^{}]*?\bdelegatecall\b[^{}]*}'
    }
    issues = []
    for issue, pattern in vulnerability_patterns.items():
        if re.search(pattern, function_content, re.MULTILINE | re.DOTALL):
            issues.append(issue.replace('_', ' '))
    return issues

def extract_functions_with_issues(content):
    functions_with_issues = []
    # Enhanced regex pattern for Solidity functions
    function_pattern = re.compile(
        r'function\s+\w+\s*\([^)]*\)\s*(?:public|private|internal|external)?\s*(?:pure|view|payable)?\s*(?:returns\s*\(.*?\))?\s*\{[\s\S]*?\}',
        re.MULTILINE | re.DOTALL
    )

    for match in function_pattern.finditer(content):
        function_declaration = match.group(0)
        function_content = function_declaration.strip()

        issues = check_for_reentrancy_issues(function_content)
        if issues:
            tokenizer = RegexpTokenizer(r'\w+|\{|\}|\(|\)|\[|\]|\;|\=|\+|\-|\*|\/|\!|\%|<|>|\||\&|\.')
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
