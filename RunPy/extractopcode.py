import os
import csv

def bytecode_to_opcode(bytecode):
    opcodes = {
        '00': 'STOP', '01': 'ADD', '02': 'MUL', '03': 'SUB', '04': 'DIV', '05': 'SDIV', '06': 'MOD', '07': 'SMOD',
        '08': 'ADDMOD', '09': 'MULMOD', '0a': 'EXP', '0b': 'SIGNEXTEND', '10': 'LT', '11': 'GT', '12': 'SLT',
        '13': 'SGT', '14': 'EQ', '15': 'ISZERO', '16': 'AND', '17': 'OR', '18': 'XOR', '19': 'NOT', '1a': 'BYTE',
        '1b': 'SHL', '1c': 'SHR', '1d': 'SAR', '20': 'SHA3', '30': 'ADDRESS', '31': 'BALANCE', '32': 'ORIGIN',
        '33': 'CALLER', '34': 'CALLVALUE', '35': 'CALLDATALOAD', '36': 'CALLDATASIZE', '37': 'CALLDATACOPY',
        '38': 'CODESIZE', '39': 'CODECOPY', '3a': 'GASPRICE', '3b': 'EXTCODESIZE', '3c': 'EXTCODECOPY', '3d': 'RETURNDATASIZE',
        '3e': 'RETURNDATACOPY', '3f': 'EXTCODEHASH', '40': 'BLOCKHASH', '41': 'COINBASE', '42': 'TIMESTAMP', '43': 'NUMBER',
        '44': 'DIFFICULTY', '45': 'GASLIMIT', '46': 'CHAINID', '47': 'SELFBALANCE', '48': 'BASEFEE', '50': 'POP',
        '51': 'MLOAD', '52': 'MSTORE', '53': 'MSTORE8', '54': 'SLOAD', '55': 'SSTORE', '56': 'JUMP', '57': 'JUMPI',
        '58': 'PC', '59': 'MSIZE', '5a': 'GAS', '5b': 'JUMPDEST', '60': 'PUSH1', '61': 'PUSH2', '62': 'PUSH3',
        '63': 'PUSH4', '64': 'PUSH5', '65': 'PUSH6', '66': 'PUSH7', '67': 'PUSH8', '68': 'PUSH9', '69': 'PUSH10',
        '6a': 'PUSH11', '6b': 'PUSH12', '6c': 'PUSH13', '6d': 'PUSH14', '6e': 'PUSH15', '6f': 'PUSH16', '70': 'PUSH17',
        '71': 'PUSH18', '72': 'PUSH19', '73': 'PUSH20', '74': 'PUSH21', '75': 'PUSH22', '76': 'PUSH23', '77': 'PUSH24',
        '78': 'PUSH25', '79': 'PUSH26', '7a': 'PUSH27', '7b': 'PUSH28', '7c': 'PUSH29', '7d': 'PUSH30', '7e': 'PUSH31',
        '7f': 'PUSH32', '80': 'DUP1', '81': 'DUP2', '82': 'DUP3', '83': 'DUP4', '84': 'DUP5', '85': 'DUP6', '86': 'DUP7',
        '87': 'DUP8', '88': 'DUP9', '89': 'DUP10', '8a': 'DUP11', '8b': 'DUP12', '8c': 'DUP13', '8d': 'DUP14', '8e': 'DUP15',
        '8f': 'DUP16', '90': 'SWAP1', '91': 'SWAP2', '92': 'SWAP3', '93': 'SWAP4', '94': 'SWAP5', '95': 'SWAP6', '96': 'SWAP7',
        '97': 'SWAP8', '98': 'SWAP9', '99': 'SWAP10', '9a': 'SWAP11', '9b': 'SWAP12', '9c': 'SWAP13', '9d': 'SWAP14',
        '9e': 'SWAP15', '9f': 'SWAP16', 'a0': 'LOG0', 'a1': 'LOG1', 'a2': 'LOG2', 'a3': 'LOG3', 'a4': 'LOG4', 'f0': 'CREATE',
        'f1': 'CALL', 'f2': 'CALLCODE', 'f3': 'RETURN', 'f4': 'DELEGATECALL', 'f5': 'CREATE2', 'fa': 'STATICCALL', 'fd': 'REVERT',
        'fe': 'INVALID', 'ff': 'SELFDESTRUCT'
    }
    
    i = 0
    length = len(bytecode)
    opcode_sequence = []
    
    while i < length:
        opcode = bytecode[i:i+2]
        if opcode in opcodes:
            instruction = opcodes[opcode]
            if instruction.startswith('PUSH'):
                num_bytes = int(instruction[4:])
                data = bytecode[i+2:i+2+num_bytes*2]
                opcode_sequence.append(f"[{i}] {instruction} 0x{data}")
                i += num_bytes*2
            else:
                opcode_sequence.append(f"[{i}] {instruction}")
        else:
            opcode_sequence.append(f"[{i}] UNKNOWN({opcode})")
        i += 2

    return opcode_sequence

def extract_features(opcode_sequence):
    features = {
        "External Calls": [],
        "Ether Transfers": [],
        "Storage Interactions": [],
        "Contract Creations": [],
        "Timestamp Dependency": [],
        "Signature Verification": [],
        "Dangerous Opcodes": [],
        "Delegate Calls": [],
        "External Calls with GAS": [],
        "Fallback Functions": [],
        "Dynamic Length Operations": [],
        "Loops": [],
        "Self-destructs": [],
        "Block Number Dependency": [],
        "Arithmetic Operations": [],
        "Unchecked Return Values": [],
        "Modifiers": [],
        "Low-level Calls": []
    }
    
    for opcode in opcode_sequence:
        if "CALL" in opcode and "DELEGATECALL" not in opcode and "STATICCALL" not in opcode:
            features["External Calls"].append(opcode)
        if "CALLVALUE" in opcode:
            features["Ether Transfers"].append(opcode)
        if "SLOAD" in opcode or "SSTORE" in opcode:
            features["Storage Interactions"].append(opcode)
        if "CREATE" in opcode:
            features["Contract Creations"].append(opcode)
        if "TIMESTAMP" in opcode:
            features["Timestamp Dependency"].append(opcode)
        if "CALLER" in opcode or "ORIGIN" in opcode:
            features["Signature Verification"].append(opcode)
        if "DELEGATECALL" in opcode:
            features["Delegate Calls"].append(opcode)
        if "CALL" in opcode and "GAS" in opcode:
            features["External Calls with GAS"].append(opcode)
        if any(func in opcode for func in ["CALLDATASIZE", "CALLDATALOAD", "CALLDATACOPY"]):
            features["Fallback Functions"].append(opcode)
        if "RETURNDATACOPY" in opcode or "CODECOPY" in opcode:
            features["Dynamic Length Operations"].append(opcode)
        if "CALL" in opcode or "DELEGATECALL" in opcode or "STATICCALL" in opcode:
            features["Dangerous Opcodes"].append(opcode)
        if "JUMP" in opcode or "JUMPI" in opcode:
            features["Loops"].append(opcode)
        if "SELFDESTRUCT" in opcode:
            features["Self-destructs"].append(opcode)
        if "NUMBER" in opcode:
            features["Block Number Dependency"].append(opcode)
        if any(op in opcode for op in ["ADD", "MUL", "SUB", "DIV", "SDIV", "MOD", "SMOD", "ADDMOD", "MULMOD", "EXP"]):
            features["Arithmetic Operations"].append(opcode)
        if "CALL" in opcode and "ISZERO" not in opcode:
            features["Unchecked Return Values"].append(opcode)
        if "JUMPDEST" in opcode:
            features["Modifiers"].append(opcode)
        if any(op in opcode for op in ["MLOAD", "MSTORE", "MSTORE8", "SLOAD", "SSTORE", "CALL", "CALLCODE", "DELEGATECALL", "STATICCALL", "LOG0", "LOG1", "LOG2", "LOG3", "LOG4", "CREATE", "CREATE2", "RETURN", "REVERT", "SELFDESTRUCT"]):
            features["Low-level Calls"].append(opcode)
    
    return features

# Đường dẫn đến thư mục chứa các tệp .bin
input_folder = r"D:\GitHub\Blockchain-Smart-Contract-Security\test\bin"

# Đường dẫn đến tệp CSV đầu ra
output_csv = r"D:\GitHub\Blockchain-Smart-Contract-Security\test\features.csv"

# Tạo hoặc mở tệp CSV để ghi kết quả
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ["File"] + [
        "External Calls", "Ether Transfers", "Storage Interactions", "Contract Creations",
        "Timestamp Dependency", "Signature Verification", "Dangerous Opcodes", "Delegate Calls",
        "External Calls with GAS", "Fallback Functions", "Dynamic Length Operations",
        "Loops", "Self-destructs", "Block Number Dependency", "Arithmetic Operations",
        "Unchecked Return Values", "Modifiers", "Low-level Calls"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # Duyệt qua tất cả các tệp trong thư mục đầu vào và các thư mục con
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".sol"):
                # Đọc nội dung tệp .bin
                with open(os.path.join(root, file), 'r') as f:
                    bytecode = f.read().strip()

                # Nếu tệp .bin không có nội dung thì bỏ qua
                if not bytecode:
                    continue

                # Chuyển đổi bytecode thành opcode
                opcode_sequence = bytecode_to_opcode(bytecode)

                # Trích xuất các đặc trưng từ opcode
                features = extract_features(opcode_sequence)
                features["File"] = os.path.join(root, file)

                # Chuyển đổi danh sách các opcode thành chuỗi
                for key in features.keys():
                    if isinstance(features[key], list):
                        features[key] = " | ".join(features[key])

                # Ghi các đặc trưng vào tệp CSV
                writer.writerow(features)

print(f"Features have been extracted and saved to {output_csv} successfully!")
