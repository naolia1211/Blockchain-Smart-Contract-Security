from nltk.tokenize import RegexpTokenizer
 
# Solidity code as a string
code = """
VulnerableBank.withdrawBalance function withdrawBalance() public {
uint amountToWithdraw = userBalances[msg.sender];
if (!(msg.sender.call.value(amountToWithdraw)())) { throw; }
userBalances[msg.sender] = <number>;
}
"""
 
# Initialize the tokenizer with the desired regular expression
tokenizer = RegexpTokenizer(r'\w+|\{|\}|\(|\)|\[|\]|\.|\;|\=|\+|\-|\*|\/|\!|\%|<|>|\||&')
 
# Tokenize the document
tokens = tokenizer.tokenize(code)
 
# Filter out unwanted tokens like '(', ')', etc.
filtered_tokens = [token for token in tokens if token not in ['(', ')']]
 
print(tokens)