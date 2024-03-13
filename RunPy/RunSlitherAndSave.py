import os
import subprocess
import re
from packaging import version

def run_slither_and_save(source_directory, destination_directory):
    # Get all .sol files in source_directory
    filenames = [f for f in os.listdir(source_directory) if f.endswith('.sol')]

    for filename in filenames:
        file_path = os.path.join(source_directory, filename)
        
        # Determine the appropriate Solidity version
        with open(file_path, 'r', encoding='latin1') as f:
            lines = f.readlines()
            pragma_lines = [line for line in lines if line.startswith('pragma solidity')]
            versions = []
            try:
                for pragma_line in pragma_lines:
                    versions.extend(re.findall(r'\d+\.\d+\.\d+', pragma_line))
                if len(versions) == 2:
                    min_version = min(version.parse(v) for v in versions)
                    max_version = max(version.parse(v) for v in versions)
                    mid_version = str(min_version + (max_version - min_version) / 2)
                else:
                    mid_version = str(max(version.parse(v) for v in versions if version.parse(v) < version.parse('0.9.0') and version.parse(v) >= version.parse('0.4.1'))) if versions else 'latest'
                if mid_version == '0.4.4' or '0.4.15':
                    mid_version = '0.4.26'
            except Exception as e:
                print(f"Error parsing Solidity version for {filename}: {str(e)}. Skipping...")
                continue

        # Use the appropriate Solidity version
        subprocess.run(['solc-select', 'use', mid_version], cwd=source_directory)

        # Create the path for the JSON file (relative to source directory)
        json_file_path = os.path.join(destination_directory, f"{os.path.splitext(filename)[0]}.json")

        # Skip if the JSON file already exists
        if os.path.exists(json_file_path):
            print(f"JSON file for {filename} already exists. Skipping...")
            continue

        # Run Slither and capture the output
        try:
            subprocess.run(["slither", filename, "--json", json_file_path], cwd=source_directory, check=True)
            print(f"Saved analysis results for {filename} to JSON file")

        except subprocess.CalledProcessError as e:
            print(f"Slither could not run on {filename}: {e}. Skipping...")

run_slither_and_save(r"D:\GitHub\Blockchain-Smart-Contract-Security\Dataset\timestamp dependency (TP)", r"D:\GitHub\Blockchain-Smart-Contract-Security\slither_analyze\timestamp dependency (TP)")
