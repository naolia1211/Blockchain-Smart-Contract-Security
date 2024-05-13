import os
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
num_vulnerabilities = 3  # Number of vulnerability categories
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_vulnerabilities)

# Define vulnerability patterns
reentrancy_patterns = {
    'external_calls': r'\\.(call|delegatecall|send|transfer)\\b',
    'state_changes_after_external_calls': r'\\.(call|delegatecall|send|transfer)\\(.\*\\);.\*=',
    'recursive_calls': r'function\\s+(\\w+)\\s\*\\(.\*?\\)\\s\*{\[\\s\\S\]\*?\\b\\1\\b\\s\*\\(.\*?\\);',
    'fallback_functions': r'function\\s\*\\(\\)\\s\*external\\s\*payable\\s\*\\{\[^}\]\*\\}'
}

unchecked_external_call_patterns = {
    'external_calls': r'\\b(address\\.|msg\\.sender\\.)?(send|call|delegatecall)\\s\*\\(',
    'ignored_return_values': r'\\b(address\\.|msg\\.sender\\.)?(send|call|delegatecall)\\s\*\\(\[^)\]\*\\);',
    'state_changes_after_external_calls': r'\\b(address\\.|msg\\.sender\\.)?(send|call|delegatecall)\\s\*\\(\[^)\]\*\\);.\*=.\*;',
    'lack_of_error_handling': r'\\b(address\\.|msg\\.sender\\.)?(send|call|delegatecall)\\s\*\\(\[^)\]\*\\);(?!\\s\*(require|revert|assert)\\s\*\\()',
    'fallback_functions_without_gas_check': r'function\\s\*\\(\\)\\s\*(external\\s\*)?payable\\s\*{\\s\*if\\s\*\\(msg\\.gas\\s\*<\\s\*2300\\)'
}

delegatecall_patterns = {
    'usage_of_delegatecall': r'delegatecall\\(',
    'state_variables_manipulation': r'\\.delegatecall\\(.\*\\);.\*=',
    'input_and_parameter_validation': r'\\.delegatecall\\((?!.\*require\\(|.\*assert\\(|.\*if.\*\\()',
    'context_preservation': r'\\.delegatecall\\(\[^)\]\*\\);(?!\\s\*this\\.)',
    'library_safe_practices': r'library\\s+\\w+\\s\*{\[^}\]\*\\.delegatecall\\('
}

# Group patterns into categories
vulnerability_patterns = [
    reentrancy_patterns,
    unchecked_external_call_patterns,
    delegatecall_patterns
]

# Path to the directory containing Solidity files
solidity_dir = r'C:\Users\hao30\Documents\GitHub\Blockchain-Smart-Contract-Security\testdata'

for filename in os.listdir(solidity_dir):
    if filename.endswith('.sol'):
        file_path = os.path.join(solidity_dir, filename)
        with open(file_path, 'r',encoding='utf-8',errors='ignore') as f:
            code_snippet = f.read()

        # Tokenize input Solidity code
        inputs = tokenizer.batch_encode_plus([code_snippet], return_tensors='pt', padding=True, truncation=True)

        # Get output logits from BERT model
        outputs = model(**inputs)[0]

        # Check for vulnerabilities
        for i, patterns in enumerate(vulnerability_patterns):
            for vulnerability, pattern in patterns.items():
                if re.search(pattern, code_snippet):
                    print(f'File: {filename}, Potential {vulnerability} vulnerability detected.')

        # Get predicted vulnerability category
        _, predicted = torch.max(outputs, dim=1)
        vulnerability_category = predicted.item()

        print(f'File: {filename}, Predicted vulnerability category: {vulnerability_category}')