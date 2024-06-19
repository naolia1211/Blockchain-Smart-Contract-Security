import os
import random

# Directory paths
input_dir = r"D:\GitHub\Blockchain-Smart-Contract-Security\RunPy\clean"
output_dir = r"D:\GitHub\Blockchain-Smart-Contract-Security\Data\BigData\Authentication_and_Authorization_Vulnerabilities\Transaction_Ordering_Dependence"
os.makedirs(output_dir, exist_ok=True)

# Define multiple Transaction Ordering Dependence (TOD) vulnerability variants
tod_variants = [
    """
    contract TODVulnerability1 {
        address public highestBidder;
        uint public highestBid;

        function bid() public payable {
            require(msg.value > highestBid, "Bid not high enough");
            highestBidder = msg.sender;
            highestBid = msg.value;
        }

        function withdraw() public {
            require(msg.sender == highestBidder, "Only highest bidder can withdraw");
            highestBidder = address(0);
            highestBid = 0;
            msg.sender.transfer(highestBid);
        }
    }
    """,
    """
    contract TODVulnerability2 {
        uint public highestBid;

        function bid() public payable {
            require(msg.value > highestBid, "Bid not high enough");
            highestBid = msg.value;
        }

        function finalizeAuction() public {
            if (highestBid > 0) {
                msg.sender.transfer(highestBid);
                highestBid = 0;
            }
        }
    }
    """,
    """
    contract TODVulnerability3 {
        mapping(address => uint) public balances;

        function deposit() public payable {
            balances[msg.sender] += msg.value;
        }

        function withdraw() public {
            require(balances[msg.sender] > 0, "Insufficient balance");
            uint amount = balances[msg.sender];
            balances[msg.sender] = 0;
            msg.sender.transfer(amount);
        }
    }
    """,
    """
    contract TODVulnerability4 {
        struct Bid {
            address bidder;
            uint amount;
        }
        Bid public highestBid;

        function bid() public payable {
            require(msg.value > highestBid.amount, "Bid not high enough");
            highestBid = Bid(msg.sender, msg.value);
        }

        function withdraw() public {
            require(msg.sender == highestBid.bidder, "Only highest bidder can withdraw");
            uint amount = highestBid.amount;
            highestBid = Bid(address(0), 0);
            msg.sender.transfer(amount);
        }
    }
    """
]

# Function to inject the TOD vulnerability function into the contract at random positions
def inject_vulnerability_function(contract_code, vulnerability_function, max_insertions):
    lines = contract_code.split('\n')
    insertion_points = random.sample(range(1, len(lines) - 1), k=max_insertions)
    for point in sorted(insertion_points, reverse=True):
        lines.insert(point, vulnerability_function)
    return '\n'.join(lines)

# Read input contracts
input_files = [f for f in os.listdir(input_dir) if f.endswith(".sol")]

variant_index = 0
max_variants_per_contract = 5  # To reach around 400 total variants with 4 contracts

for filename in input_files:
    filepath = os.path.join(input_dir, filename)
    with open(filepath, "r") as file:
        contract_code = file.read()
    
    for _ in range(max_variants_per_contract):
        # Choose a random TOD vulnerability variant
        variant = random.choice(tod_variants)
        # Determine the number of insertions
        max_insertions = random.randint(1, 3)
        # Inject the vulnerability function into the contract
        vulnerable_contract_code = inject_vulnerability_function(contract_code, variant, max_insertions)
        # Save the vulnerable contract
        variant_filename = f"{filename.rstrip('.sol')}_tod_vuln_{variant_index+1}.sol"
        variant_filepath = os.path.join(output_dir, variant_filename)
        with open(variant_filepath, "w") as variant_file:
            variant_file.write(vulnerable_contract_code)
        print(f"Vulnerable contract saved to {variant_filepath}")
        variant_index += 1

print("All vulnerable contracts with TOD vulnerabilities have been generated.")
