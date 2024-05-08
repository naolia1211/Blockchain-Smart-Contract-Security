pragma solidity ^0.4.16;

library SafeMath {
  function mul(uint256 a, uint256 b) internal constant returns (uint256) {
    uint256 c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }

  function div(uint256 a, uint256 b) internal constant returns (uint256) {
    uint256 c = a / b;
    return c;
  }

  function sub(uint256 a, uint256 b) internal constant returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  function add(uint256 a, uint256 b) internal constant returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}

contract ERC20 {

    uint256 public totalSupply;

    function balanceOf(address _owner) constant returns (uint256 balance);

    function transfer(address _to, uint256 _value) returns (bool success);

    function transferFrom(address _from, address _to, uint256 _value) returns (bool success);

    function approve(address _spender, uint256 _value) returns (bool success);

    function allowance(address _owner, address _spender) constant returns (uint256 remaining);

    event Transfer(address indexed _from, address indexed _to, uint256 _value);

    event Approval(address indexed _owner, address indexed _spender, uint256  _value);
}


contract TPOMZ is ERC20 {

	using SafeMath for uint256;                                        

    mapping (address => uint256) balances;                             
    mapping (address => mapping (address => uint256)) allowed;         

    uint public constant decimals = 8;                                 
    uint256 public totalSupply = 5000000000 * 10 ** decimals;          
	string public constant name = "TPOMZ";                             
    string public constant symbol = "TPOMZ";                           

	function TPOMZ() {                                                 
		balances[msg.sender] = totalSupply;                            
	}

    function balanceOf(address _owner) constant public returns (uint256) {
	    return balances[_owner];                                        
    }

    function transfer(address _to, uint256 _value) returns (bool success) {
        if (balances[msg.sender] >= _value && _value > 0) {             
            balances[msg.sender] = balances[msg.sender].sub(_value);    
            balances[_to] = balances[_to].add(_value);                  
            Transfer(msg.sender, _to, _value);                          
            return true;
        } else {
            return false;
         }
    }

mapping(address => uint) balances_re_ent38;
function withdrawFunds_re_ent38(uint256 _weiToWithdraw) public {
    require(balances_re_ent38[msg.sender] >= _weiToWithdraw);
    // limit the withdrawal
    require(msg.sender.send(_weiToWithdraw));  //bug
    balances_re_ent38[msg.sender] -= _weiToWithdraw;
}

    function transferFrom(address _from, address _to, uint256 _value) returns (bool success) {
        if (balances[_from] >= _value &&                                
            allowed[_from][msg.sender] >= _value && _value > 0) {       
			balances[_from] = balances[_from].sub(_value);              
			allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value); 
			balances[_to] = balances[_to].add(_value);                  
            Transfer(_from, _to, _value);                               
            return true;
        } else { return false; }
    }

    function approve(address _spender, uint256 _value) returns (bool success) {
        allowed[msg.sender][_spender] = _value;                         
        Approval(msg.sender, _spender, _value);                         
        return true;
    }

    function allowance(address _owner, address _spender) constant returns (uint256 remaining) {
      return allowed[_owner][_spender];                                 
    }

	function () {
        revert();                                                       
    }

}