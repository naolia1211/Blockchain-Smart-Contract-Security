import pandas as pd
import numpy as np
import ast  # For safely evaluating string representations of lists

# Read the tokenized data from Excel file
df = pd.read_excel(r'D:\GitHub\Blockchain-Smart-Contract-Security\result.xlsx')

# Print the first few rows of the DataFrame to understand its structure
print("DataFrame head:")
print(df.head())

# Print the column names to identify the correct names
print("Column names:")
print(df.columns)

# Assuming the tokenized data is stored in the second column
tokenized_column_name = df.columns[1]

# Convert the tokenized data from string representation to lists
df[tokenized_column_name] = df[tokenized_column_name].apply(ast.literal_eval)

# Calculate statistics
token_lengths = df[tokenized_column_name].apply(len)

# Basic statistics
mean_length = token_lengths.mean()
max_length = token_lengths.max()
min_length = token_lengths.min()
std_length = token_lengths.std()

# Percentiles
percentile_90 = np.percentile(token_lengths, 90)
percentile_95 = np.percentile(token_lengths, 95)
percentile_99 = np.percentile(token_lengths, 99)

# Output statistics
print(f'Total number of samples: {len(df)}')
print(f'Mean token length: {mean_length}')
print(f'Max token length: {max_length}')
print(f'Min token length: {min_length}')
print(f'Standard deviation of token length: {std_length}')
print(f'90th percentile token length: {percentile_90}')
print(f'95th percentile token length: {percentile_95}')
print(f'99th percentile token length: {percentile_99}')

# Additional visualization (optional)
import matplotlib.pyplot as plt
import seaborn as sns

# Plot token length distribution
plt.figure(figsize=(10, 6))
sns.histplot(token_lengths, bins=50, kde=True)
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.title('Token Length Distribution')
plt.show()
