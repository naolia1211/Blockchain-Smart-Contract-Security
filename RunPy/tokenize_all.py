import os
from pymongo import MongoClient
from nltk.tokenize import RegexpTokenizer

try:
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017')
    db = client['vul']
    collection = db['a']

    # Initialize the tokenizer with the desired regular expression
    tokenizer = RegexpTokenizer(r'\w+|\{|\}|\(|\)|\[|\]|\.|\;|\=|\+|\-|\*|\/|\!|\%|<|>|\||&')

    # Query the documents from the collection
    documents = collection.find()

    # Specify the output directory
    output_directory = r'C:\Users\Admin\Documents\GitHub\Blockchain-Smart-Contract-Security\RunPy'

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Open the output file in write mode
    output_file_path = os.path.join(output_directory, 'filtered_tokens.txt')
    with open(output_file_path, 'w') as output_file:
        # Iterate over each document
        for document in documents:
            # Check if the 'extract_feature' field exists in the document
            if 'extract_feature' in document:
                # Get the list of extract_features
                extract_features = document['extract_feature']
                
                # Iterate over each extract_feature
                for extract_feature in extract_features:
                    # Tokenize the code
                    tokens = tokenizer.tokenize(extract_feature)
                    
                    # Filter out unwanted tokens like '(', ')', etc.
                    filtered_tokens = [token for token in tokens if token not in ['(', ')']]
                    
                    # Write the filtered tokens to the output file
                    output_file.write(str(filtered_tokens) + '\n')
            else:
                print("The 'extract_feature' field does not exist in the document.")

    print(f"Filtered tokens have been written to '{output_file_path}'.")

except Exception as e:
    print(f"An error occurred: {str(e)}")