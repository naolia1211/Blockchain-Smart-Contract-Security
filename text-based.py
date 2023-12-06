import json

def extract_text_features(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    text_features = {
        "function_names": [],
        "comments": []  # Assuming there are comment fields
    }

    if "detectors" in data["results"]:
        for detector in data["results"]["detectors"]:
            for element in detector["elements"]:
                # Extracting function names
                if element["type"] == "function":
                    text_features["function_names"].append(element["name"])
                # Extracting comments (if available)
                if "comment" in element:  # Replace 'comment' with the actual key for comments in your JSON
                    text_features["comments"].append(element["comment"])

    return text_features

# Usage
file_path = 'a.json'
features = extract_text_features(file_path)
print(features)
