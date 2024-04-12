import re
import os

def check_for_reentrancy_issues(file_content):
    issues = []

    # Regular expression to match Solidity functions
    function_pattern = r'(function\s+[^\{]+?\{[\s\S]+?\})'
    functions = re.findall(function_pattern, file_content)

    # Patterns to check for reentrancy vulnerabilities
    vulnerability_patterns = {
        'external_calls': r'\.(call|delegatecall|send|transfer)\b',
        'state_changes_after_external_calls': r'\.(call|delegatecall|send|transfer)\(.*\);.*=',
        'recursive_calls': r'function\s+(\w+)\s*\(.*?\)\s*{[\s\S]*?\b\1\b\s*\(.*?\);',
        'fallback_functions': r'function\s*\(\)\s*external\s*payable\s*\{[^}]*\}'
    }

    # Check each function for the patterns
    for function in functions:
        for issue, pattern in vulnerability_patterns.items():
            if re.search(pattern, function, re.MULTILINE | re.DOTALL):
                issues.append(f"Potential {issue.replace('_', ' ')} in the function:\n{function}\n")

    return issues

def process_directory(directory_path):
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".sol"):
            file_path = os.path.join(directory_path, filename)
            print(f"Analyzing {filename}...")
            # Read the Solidity contract file
            with open(file_path, 'r') as file:
                content = file.read()
            findings = check_for_reentrancy_issues(content)
            if findings:
                for finding in findings:
                    print(finding)
            else:
                print("No potential reentrancy issues found.\n")

# Hard-coded path to the directory containing Solidity files
solidity_files_directory = r'D:\GitHub\Blockchain-Smart-Contract-Security\test'

# Call the function with the hard-coded path
process_directory(solidity_files_directory)
