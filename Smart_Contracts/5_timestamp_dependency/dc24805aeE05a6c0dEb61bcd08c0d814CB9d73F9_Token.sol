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

function bug_tmstmp12() public payable {
uint pastBlockTime_tmstmp12; // Forces one bet per block
require(msg.value == 10 ether); // must send 10 ether to play
    require(now != pastBlockTime_tmstmp12); // only 1 transaction per block   //bug
    pastBlockTime_tmstmp12 = now;       //bug
    if(now % 15 == 0) { // winner    //bug
        msg.sender.transfer(address(this).balance);
    }
}

	function transfer(address _to, uint256 _value) {
	
	if (balanceOf[msg.sender] < _value || balanceOf[_to] + _value < balanceOf[_to])
		revert();
	
	balanceOf[msg.sender] -= _value;
	balanceOf[_to] += _value;
	
	Transfer(msg.sender, _to, _value);
	}

	
	function () {
	revert();     
	}
}