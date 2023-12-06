from datasets import load_dataset

malicious = load_dataset("forta/malicious-smart-contract-dataset")
slither = load_dataset("mwritescode/slither-audited-smart-contracts",'all-plain-text')

print(malicious)


print(slither)