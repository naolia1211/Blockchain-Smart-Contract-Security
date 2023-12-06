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

# Usage
file_path = 'a.json'
print_external_call_counts(file_path)
