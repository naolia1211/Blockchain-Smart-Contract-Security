import json
import os
import shutil
import subprocess

# Define paths
json_file_path = r'D:\GitHub\Blockchain-Smart-Contract-Security\slither_analyze\reentrancy\33221.json'  # Replace with the correct path
sol_file_directory = r'D:\GitHub\Blockchain-Smart-Contract-Security\Dataset\reentrancy\source'  # Replace with the directory that contains the .sol file
destination_directory = r'D:\GitHub\Blockchain-Smart-Contract-Security\cfg\reentrancy'  # Replace with the destination directory for .dot files

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Step 1: Read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Step 2: Collect all functions with reentrancy-eth error
reentrancy_dot_files = set()
for detector in data['results']['detectors']:
    if detector['check'] == 'reentrancy-eth':
        for element in detector['elements']:
            if element['type'] == 'function':
                function_name = element['name']
                contract_name = element['type_specific_fields']['parent']['name']
                # Function names may include parameters, which we need to convert to the .dot file name format
                # Slither converts function signatures into file names by removing spaces and commas,
                # and replacing parentheses with underscores
                function_signature = function_name.replace(' ', '').replace(',', '').replace('()', '()').replace(')', '_').replace('(', '_')
                reentrancy_dot_files.add(f"{contract_name}-{function_signature}.dot")
                
# Step 3: Run slither command to generate .dot files
subprocess.run(['slither', './33221.sol', '--print', 'cfg'], cwd=sol_file_directory)
print(reentrancy_dot_files)

# Step 4: Move the specific .dot files and delete others
dot_files = [file for file in os.listdir(sol_file_directory) if file.endswith('.dot')]
print(dot_files)
# First, move the relevant .dot files
for file in dot_files:
    if file in reentrancy_dot_files:
      
        shutil.move(os.path.join(sol_file_directory, file), os.path.join(destination_directory, file))

# Then, remove any remaining .dot files in the source directory
for file in os.listdir(sol_file_directory):
     if file.endswith('.dot'):
        os.remove(os.path.join(sol_file_directory, file))
