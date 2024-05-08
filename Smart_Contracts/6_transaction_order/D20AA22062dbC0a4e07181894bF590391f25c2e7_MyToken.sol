pragma solidity ^0.4.11;

contract tokenRecipient { function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData); }

contract MyToken {

string public standard = 'Token 0.1';
string public name;
string public symbol;
uint8 public decimals;
uint256 public totalSupply;


mapping (address => uint256) public balanceOf;
mapping (address => mapping (address => uint256)) public allowance;


event Transfer(address indexed from, address indexed to, uint256 value);


event Burn(address indexed from, uint256 value);


function MyToken(
    uint256 initialSupply,
    string tokenName,
    uint8 decimalUnits,
    string tokenSymbol
    ) {
    balanceOf[msg.sender] = initialSupply;              
    totalSupply = initialSupply;                        
    name = tokenName;                                   
    symbol = tokenSymbol;                               
    decimals = decimalUnits;                            
}


function transfer(address _to, uint256 _value) {
    require(_to != 0x0);                                
    require(balanceOf[msg.sender] >= _value);           
    require(balanceOf[_to] + _value >= balanceOf[_to]); 
    balanceOf[msg.sender] -= _value;                    
    balanceOf[_to] += _value;                           
    Transfer(msg.sender, _to, _value);                  
}


function approve(address _spender, uint256 _value)
    returns (bool success) {
    allowance[msg.sender][_spender] = _value;
    return true;
}


function approveAndCall(address _spender, uint256 _value, bytes _extraData)
    returns (bool success) {
    tokenRecipient spender = tokenRecipient(_spender);
    if (approve(_spender, _value)) {
        spender.receiveApproval(msg.sender, _value, this, _extraData);
        return true;
    }
}        


function transferFrom(address _from, address _to, uint256 _value) returns (bool success) {
    require(_to != 0x0);                                
    require(balanceOf[_from] >= _value);                
    require(balanceOf[_to] + _value >= balanceOf[_to]); 
    require(_value <= allowance[_from][msg.sender]);    
    balanceOf[_from] -= _value;                         
    balanceOf[_to] += _value;                           
    allowance[_from][msg.sender] -= _value;
    Transfer(_from, _to, _value);
    return true;
}

bool claimed_TOD20 = false;
address owner_TOD20;
uint256 reward_TOD20;
function setReward_TOD20() public payable {
        require (!claimed_TOD20);

        require(msg.sender == owner_TOD20);
        owner_TOD20.transfer(reward_TOD20);
        reward_TOD20 = msg.value;
    }

    function claimReward_TOD20(uint256 submission) public {
        require (!claimed_TOD20);
        require(submission < 10);

        msg.sender.transfer(reward_TOD20);
        claimed_TOD20 = true;
    }

function burn(uint256 _value) returns (bool success) {
    require(balanceOf[msg.sender] >= _value);           
    balanceOf[msg.sender] -= _value;                    
    totalSupply -= _value;                              
    Burn(msg.sender, _value);
    return true;
}

function burnFrom(address _from, uint256 _value) returns (bool success) {
    require(balanceOf[_from] >= _value);                
    require(_value <= allowance[_from][msg.sender]);    
    balanceOf[_from] -= _value;                         
    allowance[_from][msg.sender] -= _value;             
    totalSupply -= _value;                              
    Burn(_from, _value);
    return true;
}


}