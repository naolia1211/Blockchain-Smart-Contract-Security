import os
import re
import csv
import json

def extract_function_calls(contract_code):
    pattern = r'function\s+\w+\s*\([^)]*\)\s*(?:internal|external|public|private|payable)?\s*(?:pure|view|payable|constant)?\s*(?:returns\s*\((?:[^()]+|\([^()]*\))*\))?\s*\{[\s\S]*?\}'
    matches = re.findall(pattern, contract_code)
    return matches

def extract_external_calls(contract_code):
    pattern = r'(?:\w+(?:\[\w+\])?\.)?(?:delegatecall|call|staticcall|send|transfer)\s*(?:\.gas\(\w+\)|\.value\(\w+\))?\(.*?\);'
    matches = re.findall(pattern, contract_code)
    return matches

def extract_loops(contract_code):
    pattern = r'(?:for|while)\s*\((?:[^()]+|\([^()]*\))*\)\s*(?:\{[\s\S]*?\}|\s*;)'
    matches = re.findall(pattern, contract_code)
    return matches

def extract_function_callbacks(contract_code):
    pattern = r'function\s*\((?:[^()]+|\([^()]*\))*\)\s*(?:external|public)?\s*(?:payable)?\s*(?:\{[\s\S]*?\})?'
    matches = re.findall(pattern, contract_code)
    return matches

def count_words(function_code):
    words = re.findall(r'\b\w+\b|\{|\}|\(|\)|;|\.|\,', function_code)
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

def extract_reentrancy(contract_code):
    pattern = r'(?:call|delegatecall|callcode)\s*\(\s*.*?(?:value|gas)\s*\(\s*.*?\s*\)'
    matches = re.findall(pattern, contract_code)
    return matches

def extract_tx_origin(contract_code):
    pattern = r'tx\.origin'
    matches = re.findall(pattern, contract_code)
    return matches

def extract_inline_assembly(contract_code):
    pattern = r'assembly\s*\{'
    matches = re.findall(pattern, contract_code)
    return matches

def extract_low_level_calls(contract_code):
    pattern = r'(?:call|delegatecall|callcode)\s*\('
    matches = re.findall(pattern, contract_code)
    return matches

def extract_block_timestamp(contract_code):
    pattern = r'block\.timestamp'
    matches = re.findall(pattern, contract_code)
    return matches

# Directory containing smart contract files
contract_directory = r"D:\GitHub\Blockchain-Smart-Contract-Security\Data\Interaction and Contract State Vulnerabilities\delegatecall"

# Output CSV file path
output_file = "extraction_results.csv"

# Create a CSV file and write the header
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Contract", "Function Calls", "External Calls", "Loops", "Function Callbacks", "Number of Calls",
                     "Reentrancy", "Tx.Origin", "Inline Assembly", "Low-Level Calls", "Block.timestamp"])

    # Iterate over the files in the contract directory
    for filename in os.listdir(contract_directory):
        if filename.endswith(".sol"):
            contract_file = os.path.join(contract_directory, filename)

            # Read the contract code
            with open(contract_file, 'r') as f:
                contract_code = f.read()

            # Extract information using regex
            function_calls = extract_function_calls(contract_code)
            external_calls = extract_external_calls(contract_code)
            loops = extract_loops(contract_code)
            function_callbacks = extract_function_callbacks(contract_code)
            function_word_counts = [count_words(func) for func in function_calls]
            reentrancy = extract_reentrancy(contract_code)
            tx_origin = extract_tx_origin(contract_code)
            inline_assembly = extract_inline_assembly(contract_code)
            low_level_calls = extract_low_level_calls(contract_code)
            block_timestamp = extract_block_timestamp(contract_code)

            # Format the function word counts as JSON
            formatted_function_word_counts = [json.dumps(counts) for counts in function_word_counts]

            # Write the extracted information to the CSV file
            writer.writerow([filename, "\n".join(function_calls), "\n".join(external_calls),
                             "\n".join(loops), "\n".join(function_callbacks), "\n".join(formatted_function_word_counts),
                             "\n".join(reentrancy), "\n".join(tx_origin), "\n".join(inline_assembly),
                             "\n".join(low_level_calls), "\n".join(block_timestamp)])

print("Extraction completed. Results saved to", output_file)
