import os
import json

def find_specific_check_in_json(directory, check_value):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Assuming the structure contains a 'results' key followed by 'detectors'
                        for result in data.get('results', {}).get('detectors', []):
                            if result.get('check') == check_value:
                                print(f'Found in {os.path.splitext(file)[0]}')
                except Exception as e:
                    print(f'Error reading {file_path}: {e}')

# Replace '/path/to/directory' with the path to the directory containing your JSON files
directory_path = r'D:\GitHub\Blockchain-Smart-Contract-Security\slither_analyze\reentrancy'
check_value = 'reentrancy-eth'

find_specific_check_in_json(directory_path, check_value)
