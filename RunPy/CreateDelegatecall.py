import os
import random

# Define different Solidity code snippets with various vulnerabilities

# Delegatecall vulnerabilities
def delegatecall_vulnerability_1():
    return """
    function vulnerableDelegatecall1(address callee, bytes memory data) public {
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
    }
"""

def delegatecall_vulnerability_2():
    return """
    function vulnerableDelegatecall2(address callee, bytes memory data) public {
        require(msg.sender == tx.origin, "No contract calls");
        (bool success,) = callee.delegatecall(data);
        if (!success) {
            revert("Delegatecall failed");
        }
    }
"""

def delegatecall_vulnerability_3():
    return """
    function vulnerableDelegatecall3(address callee, bytes memory data) public {
        address target = callee;
        (bool success, bytes memory returnData) = target.delegatecall(data);
        require(success, string(returnData));
    }
"""

def delegatecall_vulnerability_4():
    return """
    function vulnerableDelegatecall4(address callee, bytes memory data) public {
        (bool success,) = callee.delegatecall(abi.encodeWithSignature("someFunction(uint256)", 123));
        require(success, "Delegatecall failed");
    }
"""

# State Variables Manipulation vulnerabilities
def state_variables_manipulation_1():
    return """
    uint256 public count;

    function incrementCount() public {
        count++;
    }
"""

def state_variables_manipulation_2():
    return """
    uint256 public balance;

    function setBalance(uint256 _balance) public {
        balance = _balance;
    }
"""

def state_variables_manipulation_3():
    return """
    uint256 public value;

    function increaseValue(uint256 _value) public {
        value += _value;
    }
"""

def state_variables_manipulation_4():
    return """
    uint256 public total;

    function decrementTotal(uint256 _value) public {
        total -= _value;
    }
"""

# Input and Parameter Validation vulnerabilities
def input_validation_1():
    return """
    function unsafeFunction1(uint256 _value) public pure returns (uint256) {
        require(_value > 0, "Value must be greater than zero");
        return _value / 0; // Division by zero
    }
"""

def input_validation_2():
    return """
    function uncheckedSubtraction1(uint256 a, uint256 b) public pure returns (uint256) {
        return a - b; // No underflow check
    }
"""

def input_validation_3():
    return """
    function uncheckedAddition(uint256 a, uint256 b) public pure returns (uint256) {
        return a + b; // No overflow check
    }
"""

def input_validation_4():
    return """
    function unsafeFunction2(uint256 _value) public pure returns (uint256) {
        require(_value > 0, "Value must be greater than zero");
        return 100 / _value; // Possible division by zero
    }
"""

# Context Preservation vulnerabilities
def context_preservation_1():
    return """
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function changeOwner1(address newOwner) public {
        require(tx.origin == owner, "Only owner can change the owner");
        owner = newOwner;
    }
"""

def context_preservation_2():
    return """
    address public admin;

    function setAdmin1(address _admin) public {
        require(msg.sender == tx.origin, "No contract calls");
        admin = _admin;
    }
"""

def context_preservation_3():
    return """
    address public owner;
    uint256 public lastUpdate;

    function updateOwner(address newOwner) public {
        require(tx.origin == owner, "Only owner can change the owner");
        owner = newOwner;
        lastUpdate = block.timestamp;
    }
"""

def context_preservation_4():
    return """
    address public admin;
    uint256 public changeCount;

    function setAdmin2(address _admin) public {
        require(msg.sender == tx.origin, "No contract calls");
        admin = _admin;
        changeCount++;
    }
"""

# Library Safe Practices vulnerabilities
def library_safe_practices_1():
    return """
    library UnsafeLibrary1 {
        function unsafeFunction1(uint256 _value) public pure returns (uint256) {
            return _value / 0; // Division by zero
        }
    }

    contract UsingUnsafeLibrary1 {
        using UnsafeLibrary1 for uint256;

        function doSomething1(uint256 value) public pure returns (uint256) {
            return value.unsafeFunction1();
        }
    }
"""

def library_safe_practices_2():
    return """
    library InsecureLibrary1 {
        function unsafeAdd1(uint256 a, uint256 b) public pure returns (uint256) {
            return a + b; // No overflow check
        }
    }

    contract UsingInsecureLibrary1 {
        using InsecureLibrary1 for uint256;

        function addValues1(uint256 x, uint256 y) public pure returns (uint256) {
            return x.unsafeAdd1(y);
        }
    }
"""

def library_safe_practices_3():
    return """
    library UnsafeLibrary2 {
        function unsafeFunction2(uint256 _value) public pure returns (uint256) {
            return _value / 0; // Division by zero
        }
    }

    contract UsingUnsafeLibrary2 {
        using UnsafeLibrary2 for uint256;

        function doSomething2(uint256 value) public pure returns (uint256) {
            return value.unsafeFunction2();
        }
    }
"""

def library_safe_practices_4():
    return """
    library InsecureLibrary2 {
        function unsafeAdd2(uint256 a, uint256 b) public pure returns (uint256) {
            return a + b; // No overflow check
        }
    }

    contract UsingInsecureLibrary2 {
        using InsecureLibrary2 for uint256;

        function addValues2(uint256 x, uint256 y) public pure returns (uint256) {
            return x.unsafeAdd2(y);
        }
    }
"""

# Directory paths
clean_dir = r'C:\Users\hao30\Documents\GitHub\Blockchain-Smart-Contract-Security\Smart_Contracts\0_clean'
output_dir = r'C:\Users\hao30\Documents\GitHub\Blockchain-Smart-Contract-Security\test'
os.makedirs(output_dir, exist_ok=True)

# List of vulnerability functions
vulnerabilities = [
    delegatecall_vulnerability_1,
    delegatecall_vulnerability_2,
    delegatecall_vulnerability_3,
    delegatecall_vulnerability_4,
    state_variables_manipulation_1,
    state_variables_manipulation_2,
    state_variables_manipulation_3,
    state_variables_manipulation_4,
    input_validation_1,
    input_validation_2,
    input_validation_3,
    input_validation_4,
    context_preservation_1,
    context_preservation_2,
    context_preservation_3,
    context_preservation_4,
    library_safe_practices_1,
    library_safe_practices_2,
    library_safe_practices_3,
    library_safe_practices_4
]

# Process each clean Solidity file
for file_name in os.listdir(clean_dir):
    if file_name.endswith('.sol'):
        with open(os.path.join(clean_dir, file_name), 'r') as file:
            clean_code = file.read()

        # Inject a random number of vulnerabilities (at least one)
        num_vulnerabilities = random.randint(1, 4)
        selected_vulnerabilities = random.sample(vulnerabilities, num_vulnerabilities)
        vulnerability_code = "\n".join(vuln() for vuln in selected_vulnerabilities)
        modified_code = clean_code + vulnerability_code

        # Save the modified file
        with open(os.path.join(output_dir, file_name), 'w') as file:
            file.write(modified_code)

print(f"Processed all files and injected vulnerabilities into them.")
