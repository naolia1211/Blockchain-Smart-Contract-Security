import os
import re
import glob
import csv

def extract_ether_transfers(bytecode):
    pattern = re.compile(r'F0(?!.*?(15|F0))')
    return pattern.findall(bytecode)

def extract_storage_interactions(bytecode):
    pattern = re.compile(r'54|55')
    return pattern.findall(bytecode)

def extract_selfdestruct(bytecode):
    pattern = re.compile(r'FF')
    return pattern.findall(bytecode)

def extract_timestamp_dependency(bytecode):
    pattern = re.compile(r'42|43')
    return pattern.findall(bytecode)

def extract_dangerous_opcodes(bytecode):
    pattern = re.compile(r'F0|F1|F2|F4|F5|FA|FC|FD')
    return pattern.findall(bytecode)

def extract_arithmetic_operations(bytecode):
    pattern = re.compile(r'01|02|03|04|05|06|07|08|09|0A')
    return pattern.findall(bytecode)

def extract_fallback_functions(bytecode):
    pattern = re.compile(r'35|36|37')
    return pattern.findall(bytecode)

def extract_loops(bytecode):
    pattern = re.compile(r'56|57')
    return pattern.findall(bytecode)

def process_bytecode(bytecode):
    try:
        ether_transfers = extract_ether_transfers(bytecode)
        storage_interactions = extract_storage_interactions(bytecode)
        selfdestruct_ops = extract_selfdestruct(bytecode)
        timestamp_dependency = extract_timestamp_dependency(bytecode)
        dangerous_opcodes = extract_dangerous_opcodes(bytecode)
        arithmetic_operations = extract_arithmetic_operations(bytecode)
        fallback_functions = extract_fallback_functions(bytecode)
        loops = extract_loops(bytecode)

        return [bytecode, "\n".join(ether_transfers),
                "\n".join(storage_interactions), "\n".join(selfdestruct_ops),
                "\n".join(timestamp_dependency), "\n".join(dangerous_opcodes),
                "\n".join(arithmetic_operations), "\n".join(fallback_functions),
                "\n".join(loops)]
    except Exception as e:
        print(f"Error processing bytecode: {e}")
        return None

def read_and_process_files(directory):
    bin_files = glob.glob(os.path.join(directory, '**/*.bin'), recursive=True)
    results = []

    for bin_file in bin_files:
        with open(bin_file, 'r') as file:
            bytecode = file.read().strip()
            result = process_bytecode(bytecode)
            if result:
                results.append({
                    'file': os.path.basename(bin_file),
                    'bytecode': result[0],
                    'ether_transfers': result[1],
                    'storage_interactions': result[2],
                    'selfdestruct_ops': result[3],
                    'timestamp_dependency': result[4],
                    'dangerous_opcodes': result[5],
                    'arithmetic_operations': result[6],
                    'fallback_functions': result[7],
                    'loops': result[8]
                })

    return results

def save_to_csv(results, output_file):
    with open(output_file, mode='w', newline='') as csv_file:
        fieldnames = ['file', 'bytecode', 'ether_transfers', 'storage_interactions',
                      'selfdestruct_ops', 'timestamp_dependency', 'dangerous_opcodes',
                      'arithmetic_operations', 'fallback_functions', 'loops']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

# Đường dẫn tới thư mục chứa các file .bin
directory_path = r'D:\GitHub\Blockchain-Smart-Contract-Security\Data\Interaction and Contract State Vulnerabilities\unchecked_send\bytecode'  # Thay đổi đường dẫn tới thư mục bin của bạn
output_csv_file = 'output.csv'  # Đặt tên cho file CSV đầu ra

results = read_and_process_files(directory_path)
save_to_csv(results, output_csv_file)

print(f"Results saved to {output_csv_file}")
