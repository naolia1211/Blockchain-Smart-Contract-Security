import os
import re
import csv
from typing import List, Dict, Set

def extract_function_implementations(source_code: str) -> List[str]:
    implementation_pattern = re.compile(
        r'(?:^|(?<=\}))\s*function\s+\w+\s*\([^)]*\)\s*(?:internal|private|public|external|view|pure|payable|constant)?\s*(?:returns\s*\(.*?\))?\s*\{[^}]*\}', 
        re.MULTILINE | re.DOTALL
    )
    return implementation_pattern.findall(source_code)

def extract_from_source_code(implementations: List[str]) -> Dict[str, List[str]]:
    vulnerabilities = {
        "Unsafe External Call": [],
        "State changes after External Calls": [],
        "Error Handling": [],
        "Fallback Function Interaction": [],
        "Ignored Return Value": [],
        "State variables manipulation": [],
        "Recursive calls": [],
        "Use of 'msg.value' or 'msg.sender'": [],
        "Use of delegatecall()": [],
        "Library safe practices": [],
        "Input and parameter validation": [],
        "Gas limitations in fallback functions": [],
    }

    for impl in implementations:
        vulnerabilities["Unsafe External Call"].extend(re.findall(r'\.(call|send|delegatecall|transfer)\s*\(\s*[\w\.\(\)]*\s*\)', impl))
        vulnerabilities["State changes after External Calls"].extend(re.findall(r'\.(call|send|delegatecall|transfer)\s*\(.*\).*=', impl))
        vulnerabilities["Error Handling"].extend(re.findall(r'\b(require|revert|assert)\s*\(.*\)', impl))
        vulnerabilities["Fallback Function Interaction"].extend(re.findall(r'\bfallback\s*\(\)\s*(?:external|public|internal|private)?(?:\s+payable)?\s*\{', impl))
        vulnerabilities["Ignored Return Value"].extend(re.findall(r'\.(call|send|delegatecall|transfer)\s*\(.*\)(?!\s*=)', impl))
        vulnerabilities["State variables manipulation"].extend(re.findall(r'(?:(?:(?:private|internal|public)\s+)?(?:uint\d*|int\d*|bool|string|address|bytes\d*)\s+[a-zA-Z_$][a-zA-Z_$\d]*|mapping\s*\(.*\)\s*[a-zA-Z_$][a-zA-Z_$\d]*|struct\s+[a-zA-Z_$][a-zA-Z_$\d]*\s*\{(?:(?!\})[\s\S])*\})\s*(?:=|;)', impl))
        vulnerabilities["Recursive calls"].extend(re.findall(r'function\s+(\w+)\s*\(.*\).*\{(?:(?!\})[\s\S])*\s*\1\s*\(.*\)\s*;(?:(?!\})[\s\S])*\}', impl))
        vulnerabilities["Use of 'msg.value' or 'msg.sender'"].extend(re.findall(r'\b(msg\.value|msg\.sender)\b', impl))
        vulnerabilities["Use of delegatecall()"].extend(re.findall(r'\.delegatecall\s*\(', impl))
        vulnerabilities["Library safe practices"].extend(re.findall(r'\blibrary\s+\w+\s*\{[\s\S]*(require|assert|revert)\s*\((?:(?!\))[\s\S])*\)[\s\S]*\}', impl))
        vulnerabilities["Input and parameter validation"].extend(re.findall(r'\brequire\s*\(\s*[^)]*\s*\)', impl))
        vulnerabilities["Gas limitations in fallback functions"].extend(re.findall(r'\bfallback\s*\(\)\s*(?:external|public|internal|private)?(?:\s+payable)?\s*\{(?:(?!\})(?!return)[\s\S])+\}', impl))

    return vulnerabilities

def get_function_containing_pattern(implementations: List[str], pattern: str) -> List[str]:
    relevant_functions = []
    for function in implementations:
        if re.search(pattern, function):
            relevant_functions.append(function)

    return relevant_functions

def scan_directory(directory: str) -> List[Dict[str, List[str]]]:
    results = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".sol"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                    implementations = extract_function_implementations(source_code)
                    vulnerabilities = extract_from_source_code(implementations)
                    
                    result = {"File": file}
                    for vuln_type, patterns in vulnerabilities.items():
                        functions_with_vuln: Set[str] = set()
                        for pattern in patterns:
                            functions = get_function_containing_pattern(implementations, re.escape(pattern))
                            for function in functions:
                                functions_with_vuln.add(function)
                        if not functions_with_vuln:
                            functions_with_vuln.add("Not found")
                        result[vuln_type] = list(functions_with_vuln)
                    results.append(result)
    return results

def save_to_csv(results: List[Dict[str, List[str]]], output_file: str) -> None:
    all_vuln_types = set()
    for result in results:
        all_vuln_types.update(result.keys())
    all_vuln_types.discard("File")
    all_vuln_types = list(all_vuln_types)

    with open(output_file, 'w', newline='', encoding='utf-8') as output_csv:
        fieldnames = ["File"] + all_vuln_types
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {vuln_type: "\n".join(result.get(vuln_type, ["Not found"])) for vuln_type in all_vuln_types}
            row["File"] = result["File"]
            writer.writerow(row)

def main():
    directory_to_scan = r"D:\GitHub\Blockchain-Smart-Contract-Security\Data\Interaction and Contract State Vulnerabilities\reentrancy"
    output_csv_file = "./result.csv"

    results = scan_directory(directory_to_scan)
    save_to_csv(results, output_csv_file)

    print(f"Vulnerability scan completed. Results saved to {output_csv_file}")

if __name__ == "__main__":
    main()
