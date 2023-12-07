import os
import json

def print_external_call_counts(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    function_counts = {}  # Dictionary to store function counts

    if "detectors" in data["results"]:
        for detector in data["results"]["detectors"]:
            for element in detector["elements"]:
                if element["type"] == "node" and element.get("additional_fields", {}).get("underlying_type") == "external_calls":
                    # Extracting only the function name without parameters
                    function_name = element["name"].split('(')[0]
                    if function_name in function_counts:
                        function_counts[function_name] += 1
                    else:
                        function_counts[function_name] = 1

    # Print the function counts
    for function_name, count in function_counts.items():
        print(f"{function_name}: {count}")

def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")
            print_external_call_counts(file_path)
            print("\n")  # Add a newline for better readability between files

# Usage
directory_path = 'D:\\Github\\Blockchain-Smart-Contract-Security\\extract_feature_from_slither'
process_directory(directory_path)
