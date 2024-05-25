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

    function vulnerableDelegatecallWithStateManipulation1(address callee, bytes memory data) public {
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
        count++;
    }
"""

def state_variables_manipulation_2():
    return """
    uint256 public balance;

    function vulnerableDelegatecallWithStateManipulation2(address callee, bytes memory data) public {
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
        balance += 100;
    }
"""

def state_variables_manipulation_3():
    return """
    uint256 public value;

    function vulnerableDelegatecallWithStateManipulation3(address callee, bytes memory data) public {
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
        value = 0;
    }
"""

def state_variables_manipulation_4():
    return """
    uint256 public total;

    function vulnerableDelegatecallWithStateManipulation4(address callee, bytes memory data) public {
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
        total -= 10;
    }
"""

# Input and Parameter Validation vulnerabilities
def input_validation_1():
    return """
    function unsafeFunction1(address callee, bytes memory data, uint256 _value) public pure returns (uint256) {
        require(_value > 0, "Value must be greater than zero");
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
        return _value / 0; // Division by zero
    }
"""

def input_validation_2():
    return """
    function uncheckedSubtraction1(address callee, bytes memory data, uint256 a, uint256 b) public pure returns (uint256) {
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
        return a - b; // No underflow check
    }
"""

def input_validation_3():
    return """
    function uncheckedAddition(address callee, bytes memory data, uint256 a, uint256 b) public pure returns (uint256) {
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
        return a + b; // No overflow check
    }
"""

def input_validation_4():
    return """
    function unsafeFunction2(address callee, bytes memory data, uint256 _value) public pure returns (uint256) {
        require(_value > 0, "Value must be greater than zero");
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
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

    function changeOwner1(address callee, bytes memory data, address newOwner) public {
        require(tx.origin == owner, "Only owner can change the owner");
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
        owner = newOwner;
    }
"""

def context_preservation_2():
    return """
    address public admin;

    function setAdmin1(address callee, bytes memory data, address _admin) public {
        require(msg.sender == tx.origin, "No contract calls");
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
        admin = _admin;
    }
"""

def context_preservation_3():
    return """
    address public owner;
    uint256 public lastUpdate;

    function updateOwner(address callee, bytes memory data, address newOwner) public {
        require(tx.origin == owner, "Only owner can change the owner");
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
        owner = newOwner;
        lastUpdate = block.timestamp;
    }
"""

def context_preservation_4():
    return """
    address public admin;
    uint256 public changeCount;

    function setAdmin2(address callee, bytes memory data, address _admin) public {
        require(msg.sender == tx.origin, "No contract calls");
        (bool success,) = callee.delegatecall(data);
        require(success, "Delegatecall failed");
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

        function doSomething1(address callee, bytes memory data, uint256 value) public pure returns (uint256) {
            (bool success,) = callee.delegatecall(data);
            require(success, "Delegatecall failed");
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

        function addValues1(address callee, bytes memory data, uint256 x, uint256 y) public pure returns (uint256) {
            (bool success,) = callee.delegatecall(data);
            require(success, "Delegatecall failed");
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

        function doSomething2(address callee, bytes memory data, uint256 value) public pure returns (uint256) {
            (bool success,) = callee.delegatecall(data);
            require(success, "Delegatecall failed");
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

        function addValues2(address callee, bytes memory data, uint256 x, uint256 y) public pure returns (uint256) {
            (bool success,) = callee.delegatecall(data);
            require(success, "Delegatecall failed");
            return x.unsafeAdd2(y);
        }
    }
"""
def unchecked_send_vulnerability_1():
    return """
    function sendEther1(address payable recipient) public {
        recipient.send(1 ether); // Unchecked send
    }
"""

def unchecked_send_vulnerability_2():
    return """
    function sendEther2(address payable recipient) public {
        recipient.send(address(this).balance); // Unchecked send
    }
"""

def unchecked_send_vulnerability_3():
    return """
    function sendEther3(address payable recipient) public {
        bool success = recipient.send(1 ether); // Unchecked send
    }
"""

def unchecked_send_vulnerability_4():
    return """
    function sendEther4(address payable recipient) public {
        recipient.send(0.1 ether); // Unchecked send
    }
"""

# Unchecked Transfer vulnerabilities
def unchecked_transfer_vulnerability_1():
    return """
    function transferEther1(address payable recipient) public {
        recipient.transfer(1 ether); // Unchecked transfer
    }
"""

def unchecked_transfer_vulnerability_2():
    return """
    function transferEther2(address payable recipient) public {
        recipient.transfer(address(this).balance); // Unchecked transfer
    }
"""

def unchecked_transfer_vulnerability_3():
    return """
    function transferEther3(address payable recipient) public {
        bool success = recipient.transfer(1 ether); // Unchecked transfer
    }
"""

def unchecked_transfer_vulnerability_4():
    return """
    function transferEther4(address payable recipient) public {
        recipient.transfer(0.1 ether); // Unchecked transfer
    }
"""

# Msg.sender Transfer vulnerabilities
def msg_sender_transfer_vulnerability_1():
    return """
    function transferToSender1() public {
        msg.sender.transfer(1 ether); // Unchecked transfer to msg.sender
    }
"""

def msg_sender_transfer_vulnerability_2():
    return """
    function transferToSender2() public {
        msg.sender.transfer(address(this).balance); // Unchecked transfer to msg.sender
    }
"""

def msg_sender_transfer_vulnerability_3():
    return """
    function transferToSender3() public {
        bool success = msg.sender.transfer(1 ether); // Unchecked transfer to msg.sender
    }
"""

def msg_sender_transfer_vulnerability_4():
    return """
    function transferToSender4() public {
        msg.sender.transfer(0.1 ether); // Unchecked transfer to msg.sender
    }
"""

# Define vulnerability patterns
vulnerability_patterns = {
    'unchecked_send': r'\bsend\s*\(.*?\)\s*;',
    'unchecked_transfer': r'\btransfer\s*\(.*?\)\s*;',
    'msg_sender_transfer': r'\bmsg\.sender\.transfer\s*\(.*?\)\s*;'
}
# Directory paths
clean_dir = r'D:\GitHub\Blockchain-Smart-Contract-Security\Smart_Contracts\0_clean'
output_dir = r'D:\GitHub\Blockchain-Smart-Contract-Security\test1'
os.makedirs(output_dir, exist_ok=True)

# List of vulnerability functions
vulnerabilities = [
    # delegatecall_vulnerability_1,
    # delegatecall_vulnerability_2,
    # delegatecall_vulnerability_3,
    # delegatecall_vulnerability_4,
    # state_variables_manipulation_1,
    # state_variables_manipulation_2,
    # state_variables_manipulation_3,
    # state_variables_manipulation_4,
    # input_validation_1,
    # input_validation_2,
    # input_validation_3,
    # input_validation_4,
    # context_preservation_1,
    # context_preservation_2,
    # context_preservation_3,
    # context_preservation_4,
    # library_safe_practices_1,
    # library_safe_practices_2,
    # library_safe_practices_3,
    # library_safe_practices_4
    unchecked_send_vulnerability_1,
    unchecked_send_vulnerability_2,
    unchecked_send_vulnerability_3,
    unchecked_send_vulnerability_4,
    unchecked_transfer_vulnerability_1,
    unchecked_transfer_vulnerability_2,
    unchecked_transfer_vulnerability_3,
    unchecked_transfer_vulnerability_4,
    msg_sender_transfer_vulnerability_1,
    msg_sender_transfer_vulnerability_2,
    msg_sender_transfer_vulnerability_3,
    msg_sender_transfer_vulnerability_4
]

# Get all .sol files from the clean directory
all_files = [f for f in os.listdir(clean_dir) if f.endswith('.sol')]

# Ensure there are enough files to reach 700
if len(all_files) < 700:
    all_files = all_files * (700 // len(all_files)) + all_files[:700 % len(all_files)]

# Randomly select up to 700 files without removing them from the directory
selected_files = random.sample(all_files, 700)

# Process each selected file and rename them sequentially
for index, file_name in enumerate(selected_files, start=1):
    with open(os.path.join(clean_dir, file_name), 'r') as file:
        clean_code = file.read()

    # Inject a random number of vulnerabilities (at least one)
    num_vulnerabilities = random.randint(1, 4)
    selected_vulnerabilities = random.sample(vulnerabilities, num_vulnerabilities)
    vulnerability_code = "\n".join(vuln() for vuln in selected_vulnerabilities)
    modified_code = clean_code + vulnerability_code

    # Save the modified file with a new sequential name
    output_file_name = f"preprocessed_{index}.sol"
    with open(os.path.join(output_dir, output_file_name), 'w') as file:
        file.write(modified_code)

print(f"Processed and injected vulnerabilities into 700 files.")
