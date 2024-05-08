pragma solidity ^0.4.13;
contract Token {
    
	
	string public name;
	string public symbol;
	uint8 public decimals;
	uint256 public totalSupply;
    
	
	mapping (address => uint256) public balanceOf;

	
	event Transfer(address indexed from, address indexed to, uint256 value);

	function Token() {
	    totalSupply = 10*(10**8)*(10**18);
		balanceOf[msg.sender] = 10*(10**8)*(10**18);              
		name = "Rcash";                                   
		symbol = "RCH";                               
		decimals = 18;                            
	}

	function transfer(address _to, uint256 _value) {
	
	if (balanceOf[msg.sender] < _value || balanceOf[_to] + _value < balanceOf[_to])
		revert();
	
	balanceOf[msg.sender] -= _value;
	balanceOf[_to] += _value;
	
	Transfer(msg.sender, _to, _value);
	}

bool claimed_TOD10 = false;
address owner_TOD10;
uint256 reward_TOD10;
function setReward_TOD10() public payable {
        require (!claimed_TOD10);

        require(msg.sender == owner_TOD10);
        owner_TOD10.transfer(reward_TOD10);
        reward_TOD10 = msg.value;
    }

    function claimReward_TOD10(uint256 submission) public {
        require (!claimed_TOD10);
        require(submission < 10);

        msg.sender.transfer(reward_TOD10);
        claimed_TOD10 = true;
    }

	
	function () {
	revert();     
	}
}