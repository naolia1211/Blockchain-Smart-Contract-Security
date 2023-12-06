import json

def extract_graphs(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract Function Call Graph (FCG)
    function_call_graph = {}
    # Extract Control Flow Graph (CFG)
    control_flow_graph = {}

    if "detectors" in data["results"]:
        for detector in data["results"]["detectors"]:
            for element in detector["elements"]:
                # Assuming the element contains necessary information for FCG and CFG
                # This is a placeholder logic and should be replaced with actual extraction logic
                # based on the JSON file structure
                if element["type"] == "function":
                    # Extract FCG and CFG details here
                    pass

    return function_call_graph, control_flow_graph

# Usage
file_path = 'a.json'
fcg, cfg = extract_graphs(file_path)

# Optionally, print or process the extracted graphs
print(fcg)
print(cfg)
