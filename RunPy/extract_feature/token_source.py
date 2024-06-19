import csv
import os
from nltk.tokenize import RegexpTokenizer

# Define the tokenizer with the provided pattern
tokenizer = RegexpTokenizer(r'\w+|\{|\}|\(|\)|\[|\]|\;|\=|\+|\-|\*|\/|\!|\%|<|>|\||\&|\.|\,')

# Function to tokenize each cell in each feature
def tokenize_features(row, fieldnames):
    tokenized_row = {}
    for field in fieldnames:
        if field == "Contract":
            tokenized_row[field] = row[field]  # Keep the Contract column unchanged
        else:
            if row[field]:
                tokens = tokenizer.tokenize(row[field])
                tokenized_row[field] = tokens  # Store the tokens as a list
            else:
                tokenized_row[field] = []
    return tokenized_row

# Path to the existing CSV file
input_csv_path = r'D:\GitHub\Blockchain-Smart-Contract-Security\combined_results.csv'  # Update with your actual path

# Path to the new CSV file for tokenized features
output_csv_path = './tokenized_extraction_results.csv'

# Read the existing CSV file and process it
with open(input_csv_path, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    
    # Create a new CSV file and write the header
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            tokenized_row = tokenize_features(row, fieldnames)
            # Convert lists to strings for CSV writing
            stringified_row = {key: str(value) for key, value in tokenized_row.items()}
            writer.writerow(stringified_row)

print("Tokenization completed. Results saved to", output_csv_path)
