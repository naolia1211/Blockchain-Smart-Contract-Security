import os
from pymongo import MongoClient
from nltk.tokenize import RegexpTokenizer

try:
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['Interaction_and_Contract_State_Vulnerabilities']

    # Initialize the tokenizer with the desired regular expression
    tokenizer = RegexpTokenizer(r'\w+|\{|\}|\(|\)|\\[|\\]|\.|\\;|\\=|\\+|\\-|\\\*|\\/|\\!|\\%|<|>|\\||&')

    # Specify the output directory
    output_directory = r'D:\GitHub\Blockchain-Smart-Contract-Security\RunPy\output_tokenize\unchecked send'

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Query the documents from the collection
    collection = db['unchecked_send']
    documents = collection.find()

    # Iterate over each document
    for document in documents:
        # Check if the 'extract_feature' field exists in the document
        if 'extract_feature' in document:
            # Get the list of extract_features
            extract_features = document['extract_feature']

            # Iterate over each extract_feature
            for extract_feature in extract_features:
                # Check if the 'feature_type' and 'function_content' fields exist in the extract_feature
                if 'feature_type' in extract_feature and 'function_content' in extract_feature:
                    feature_type = extract_feature['feature_type']
                    function_content = extract_feature['function_content']

                    # Tokenize the code
                    tokens = tokenizer.tokenize(function_content)

                    # Filter out unwanted tokens like '(', ')', etc.
                    filtered_tokens = [token for token in tokens if token not in ['(', ')']]

                    # Open the output file in append mode
                    output_file_path = os.path.join(output_directory, f'{feature_type}_filtered_tokens.txt')
                    with open(output_file_path, 'a') as output_file:
                        # Write the filtered tokens to the output file
                        output_file.write(str(filtered_tokens) + '\n')
                else:
                    print(f"The 'feature_type' or 'function_content' field is missing in the extract_feature.")
        else:
            print("The 'extract_feature' field does not exist in the document.")

    print("Tokenization completed successfully.")

except Exception as e:
    print(f"An error occurred: {str(e)}")