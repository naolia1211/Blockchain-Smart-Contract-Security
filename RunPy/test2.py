import os
import re
import csv
from tqdm import tqdm

def extract_function_calls(contract_code):
    pattern = re.compile(r'function\s+\w+\s*\([^)]*\)\s*(?:internal|external|public|private|payable|pure|view|constant|virtual|override)?\s*(?:internal|external|public|private|payable|pure|view|constant|virtual|override)?\s*(?:returns\s*\((?:[^()]+|\([^()]*\))*\))?\s*\{[\s\S]*?\}')
    return pattern.findall(contract_code)

def extract_external_calls(contract_code):
    pattern = re.compile(r'(?:\w+(?:\[\w+\])?\.)?(?:delegatecall|call|staticcall|send|transfer)\s*(?:\.gas\(\w+\)|\.value\(\w+\))?\(.*?\);')
    return pattern.findall(contract_code)

def extract_loops(contract_code):
    pattern = re.compile(r'(?:for|while)\s*\((?:[^()]+|\([^()]*\))*\)\s*(?:\{[\s\S]*?\}|\s*;)')
    return pattern.findall(contract_code)

def extract_function_callbacks(contract_code):
    pattern = re.compile(r'function\s*\((?:[^()]+|\([^()]*\))*\)\s*(?:external|public)?\s*(?:payable)?\s*(?:\{[\s\S]*?\})?')
    return pattern.findall(contract_code)

def extract_reentrancy(contract_code):
    pattern = re.compile(r'(?:call|delegatecall|callcode)\s*\(\s*.*?(?:value|gas)\s*\(\s*.*?\s*\)')
    return pattern.findall(contract_code)

def extract_integer_overflows(contract_code):
    pattern = re.compile(r'\b(?:\+\+|--|\+=|-=|\*=|/=|%=|=\s*\w+\s*\+\s*\w+|\=\s*\w+\s*\-\s*\w+|\=\s*\w+\s*\*\s*\w+|\=\s*\w+\s*\/\s*\w+|\=\s*\w+\s*\%\s*\w+)\b')
    return pattern.findall(contract_code)

def extract_unchecked_low_level_calls(contract_code):
    pattern = re.compile(r'\.\s*(?:call|delegatecall|callcode)\s*\(\s*(?:\w*\s*\.\s*value\(\w*\)\s*)?\)')
    return pattern.findall(contract_code)

def extract_uninitialized_storage(contract_code):
    pattern = re.compile(r'\b(?:uint|int|string|address|bool|bytes)\s+\w+\s*;\s*(?!\=)')
    return pattern.findall(contract_code)

def extract_timestamp_dependence(contract_code):
    pattern = re.compile(r'\bblock\.timestamp\b|\bnow\b')
    return pattern.findall(contract_code)

def extract_insecure_imports(contract_code):
    pattern = re.compile(r'import\s+"[^"]+";')
    return pattern.findall(contract_code)

def extract_unsafe_inheritance(contract_code):
    pattern = re.compile(r'contract\s+\w+\s+is\s+[^;]+')
    return pattern.findall(contract_code)

def extract_modifiers(contract_code):
    pattern = re.compile(r'modifier\s+\w+\s*\([^)]*\)\s*\{[\s\S]*?\}')
    return pattern.findall(contract_code)

def extract_state_variables(contract_code):
    pattern = re.compile(r'\b(?:uint|int|string|address|bool|bytes)\s+\w+\s*(?:=\s*[^;]+)?;')
    return pattern.findall(contract_code)

def extract_insecure_interface_implementations(contract_code):
    pattern = re.compile(r'contract\s+\w+\s+is\s+[^;]+')
    return pattern.findall(contract_code)

def extract_fallback_functions(contract_code):
    pattern = re.compile(r'fallback\s*\(\s*\)\s*(?:external|public)?\s*(?:payable)?\s*(?:\{[\s\S]*?\})?')
    return pattern.findall(contract_code)

def extract_selfdestruct_functions(contract_code):
    pattern = re.compile(r'\bselfdestruct\s*\(\s*(?:address\s*\(\s*\))?(?:\s*[\.\w]+\s*)*\)')
    return pattern.findall(contract_code)

def extract_delegatecall_usage(contract_code):
    pattern = re.compile(r'\bdelegateCall\s*\(\s*(?:bytes memory|string memory)?\s*.*?\)')
    return pattern.findall(contract_code)

def extract_default_visibilities(contract_code):
    pattern = re.compile(r'\b(?:function|constructor|state variable)\s+\w+\s*\((.*?)\)\s*(?!\bexternal\b|\bpublic\b|\binternal\b|\bprivate\b)')
    return pattern.findall(contract_code)

def extract_dos_patterns(contract_code):
    pattern = re.compile(r'for\s*\(.*?\)\s*\{(?:(?!\})[\s\S])*\}')
    return pattern.findall(contract_code)

def extract_insecure_randomness_usage(contract_code):
    pattern = re.compile(r'\b(?:keccak256|sha256|ripemd160|ecrecover|addmod|mulmod|block\.timestamp|block\.number|block\.difficulty|block\.gaslimit|blockhash|msg\.sender)\s*\(')
    return pattern.findall(contract_code)

def extract_parameter_ordering(contract_code):
    pattern = re.compile(r'function\s+\w+\s*\(.*?\)')
    return pattern.findall(contract_code)

def extract_tod_patterns(contract_code):
    pattern = re.compile(r'(?:call|delegatecall|callcode)\s*\(\s*.*?(?:value|gas)\s*\(\s*.*?\s*\)')
    return pattern.findall(contract_code)

def extract_tx_origin_usage(contract_code):
    pattern = re.compile(r'\btx\.origin\b')
    return pattern.findall(contract_code)

def extract_block_number_dependence(contract_code):
    pattern = re.compile(r'\bblock\.number\b')
    return pattern.findall(contract_code)

def extract_underflows(contract_code):
    pattern = re.compile(r'\b(?:\+\+|--|\+=|-=|\*=|/=|%=|=\s*\w+\s*\+\s*\w+|\=\s*\w+\s*\-\s*\w+|\=\s*\w+\s*\*\s*\w+|\=\s*\w+\s*\/\s*\w+|\=\s*\w+\s*\%\s*\w+)\b')
    return pattern.findall(contract_code)

def process_file(filename, contract_directory):
    try:
        contract_file = os.path.join(contract_directory, filename)

        # Read the contract code
        with open(contract_file, 'r') as f:
            contract_code = f.read()

        # Extract information using regex
        function_calls = extract_function_calls(contract_code)
        external_calls = extract_external_calls(contract_code)
        loops = extract_loops(contract_code)
        function_callbacks = extract_function_callbacks(contract_code)
        reentrancy = extract_reentrancy(contract_code)
        integer_overflows = extract_integer_overflows(contract_code)
        unchecked_low_level_calls = extract_unchecked_low_level_calls(contract_code)
        uninitialized_storage = extract_uninitialized_storage(contract_code)
        timestamp_dependence = extract_timestamp_dependence(contract_code)
        insecure_imports = extract_insecure_imports(contract_code)
        unsafe_inheritance = extract_unsafe_inheritance(contract_code)
        modifiers = extract_modifiers(contract_code)
        state_variables = extract_state_variables(contract_code)
        insecure_interface_implementations = extract_insecure_interface_implementations(contract_code)
        fallback_functions = extract_fallback_functions(contract_code)
        selfdestruct_functions = extract_selfdestruct_functions(contract_code)
        delegatecall_usage = extract_delegatecall_usage(contract_code)
        default_visibilities = extract_default_visibilities(contract_code)
        dos_patterns = extract_dos_patterns(contract_code)
        insecure_randomness_usage = extract_insecure_randomness_usage(contract_code)
        parameter_ordering = extract_parameter_ordering(contract_code)
        tod_patterns = extract_tod_patterns(contract_code)
        tx_origin_usage = extract_tx_origin_usage(contract_code)
        block_number_dependence = extract_block_number_dependence(contract_code)
        underflows = extract_underflows(contract_code)

        return [filename, "\n".join(function_calls), "\n".join(external_calls),
                "\n".join(loops), "\n".join(function_callbacks), "\n".join(reentrancy),
                "\n".join(integer_overflows), "\n".join(unchecked_low_level_calls),
                "\n".join(uninitialized_storage), "\n".join(timestamp_dependence),
                "\n".join(insecure_imports), "\n".join(unsafe_inheritance),
                "\n".join(modifiers), "\n".join(state_variables),
                "\n".join(insecure_interface_implementations), "\n".join(fallback_functions),
                "\n".join(selfdestruct_functions), "\n".join(delegatecall_usage),
                "\n".join(default_visibilities), "\n".join(dos_patterns),
                "\n".join(insecure_randomness_usage), "\n".join(parameter_ordering),
                "\n".join(tod_patterns), "\n".join(tx_origin_usage),
                "\n".join(block_number_dependence), "\n".join(underflows)]
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None

# Directory containing smart contract files
contract_directory = r"D:\GitHub\Blockchain-Smart-Contract-Security\test\delegatecall"

# Output CSV file path
output_file = "enhanced_extraction_results.csv"

# Get a list of files in the directory
file_list = [f for f in os.listdir(contract_directory) if f.endswith(".sol")]

# Create a CSV file and write the header
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Contract", "Function Calls", "External Calls", "Loops",
                     "Function Callbacks", "Reentrancy", "Integer Overflows",
                     "Unchecked Low-Level Calls", "Uninitialized Storage",
                     "Timestamp Dependence", "Insecure Imports", "Unsafe Inheritance",
                     "Modifiers", "State Variables", "Insecure Interface Implementations",
                     "Fallback Functions", "Self-destruct Functions", "Delegatecall Usage",
                     "Default Visibilities", "DoS Patterns", "Insecure Randomness Usage",
                     "Parameter Ordering", "Transaction Order Dependence", "Tx.origin Usage",
                     "Block Number Dependence", "Underflows"])

    # Process files sequentially
    for filename in tqdm(file_list, desc="Processing files"):
        result = process_file(filename, contract_directory)
        if result:
            writer.writerow(result)

print("Enhanced extraction completed. Results saved to", output_file)