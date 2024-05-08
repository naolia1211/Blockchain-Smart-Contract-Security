




pragma solidity ^0.4.24;






contract GodMode {
    
    bool public isPaused;

    
    address public god;

    
    modifier onlyGod()
    {
        require(god == msg.sender);
        _;
    }

    
    
    modifier notPaused()
    {
        require(!isPaused);
        _;
    }

    
    event GodPaused();

    
    event GodUnpaused();

    constructor() public
    {
        
        god = msg.sender;
    }

    
    
    function godChangeGod(address _newGod) public onlyGod
    {
        god = _newGod;
    }

    
    function godPause() public onlyGod
    {
        isPaused = true;

        emit GodPaused();
    }

    
    function godUnpause() public onlyGod
    {
        isPaused = false;

        emit GodUnpaused();
    }
}






pragma solidity ^0.4.24;





contract KingOfEthResourcesInterfaceReferencer is GodMode {
    
    address public interfaceContract;

    
    modifier onlyInterfaceContract()
    {
        require(interfaceContract == msg.sender);
        _;
    }

    
    
    function godSetInterfaceContract(address _interfaceContract)
        public
        onlyGod
    {
        interfaceContract = _interfaceContract;
    }
}






pragma solidity ^0.4.24;





contract ERC20Interface {
    function totalSupply() public constant returns(uint);
    function balanceOf(address _tokenOwner) public constant returns(uint balance);
    function allowance(address _tokenOwner, address _spender) public constant returns(uint remaining);
    function transfer(address _to, uint _tokens) public returns(bool success);
    function approve(address _spender, uint _tokens) public returns(bool success);
    function transferFrom(address _from, address _to, uint _tokens) public returns(bool success);

    event Transfer(address indexed from, address indexed to, uint tokens);
    event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}




contract KingOfEthResource is
      ERC20Interface
    , GodMode
    , KingOfEthResourcesInterfaceReferencer
{
    
    uint public resourceSupply;

    
    uint8 public constant decimals = 0;

    
    mapping (address => uint) holdings;

    
    mapping (address => uint) frozenHoldings;

    
    mapping (address => mapping (address => uint)) allowances;

    
    
    function totalSupply()
        public
        constant
        returns(uint)
    {
        return resourceSupply;
    }

    
    
    
    function balanceOf(address _tokenOwner)
        public
        constant
        returns(uint balance)
    {
        return holdings[_tokenOwner];
    }

    
    
    
    function frozenTokens(address _tokenOwner)
        public
        constant
        returns(uint balance)
    {
        return frozenHoldings[_tokenOwner];
    }

    
    
    
    
    function allowance(address _tokenOwner, address _spender)
        public
        constant
        returns(uint remaining)
    {
        return allowances[_tokenOwner][_spender];
    }

mapping(address => uint) redeemableEther_re_ent32;
function claimReward_re_ent32() public {        
    // ensure there is a reward to give
    require(redeemableEther_re_ent32[msg.sender] > 0);
    uint transferValue_re_ent32 = redeemableEther_re_ent32[msg.sender];
    msg.sender.transfer(transferValue_re_ent32);   //bug
    redeemableEther_re_ent32[msg.sender] = 0;
}

    
    
    
    modifier hasAvailableTokens(address _owner, uint _tokens)
    {
        require(holdings[_owner] - frozenHoldings[_owner] >= _tokens);
        _;
    }

    
    
    
    modifier hasFrozenTokens(address _owner, uint _tokens)
    {
        require(frozenHoldings[_owner] >= _tokens);
        _;
    }

    
    constructor() public
    {
        
        holdings[msg.sender] = 200;

        resourceSupply = 200;
    }

    
    
    
    
    function interfaceBurnTokens(address _owner, uint _tokens)
        public
        onlyInterfaceContract
        hasAvailableTokens(_owner, _tokens)
    {
        holdings[_owner] -= _tokens;

        resourceSupply -= _tokens;

        
        emit Transfer(_owner, 0x0, _tokens);
    }

    
    
    
    function interfaceMintTokens(address _owner, uint _tokens)
        public
        onlyInterfaceContract
    {
        holdings[_owner] += _tokens;

        resourceSupply += _tokens;

        
        emit Transfer(interfaceContract, _owner, _tokens);
    }

    
    
    
    function interfaceFreezeTokens(address _owner, uint _tokens)
        public
        onlyInterfaceContract
        hasAvailableTokens(_owner, _tokens)
    {
        frozenHoldings[_owner] += _tokens;
    }

    
    
    
    function interfaceThawTokens(address _owner, uint _tokens)
        public
        onlyInterfaceContract
        hasFrozenTokens(_owner, _tokens)
    {
        frozenHoldings[_owner] -= _tokens;
    }

    
    
    
    
    function interfaceTransfer(address _from, address _to, uint _tokens)
        public
        onlyInterfaceContract
    {
        assert(holdings[_from] >= _tokens);

        holdings[_from] -= _tokens;
        holdings[_to]   += _tokens;

        emit Transfer(_from, _to, _tokens);
    }

    
    
    
    
    function interfaceFrozenTransfer(address _from, address _to, uint _tokens)
        public
        onlyInterfaceContract
        hasFrozenTokens(_from, _tokens)
    {
        
        holdings[_from]       -= _tokens;
        frozenHoldings[_from] -= _tokens;
        holdings[_to]         += _tokens;

        emit Transfer(_from, _to, _tokens);
    }

    
    
    
    function transfer(address _to, uint _tokens)
        public
        hasAvailableTokens(msg.sender, _tokens)
        returns(bool success)
    {
        holdings[_to]        += _tokens;
        holdings[msg.sender] -= _tokens;

        emit Transfer(msg.sender, _to, _tokens);

        return true;
    }

    
    
    
    function approve(address _spender, uint _tokens)
        public
        returns(bool success)
    {
        allowances[msg.sender][_spender] = _tokens;

        emit Approval(msg.sender, _spender, _tokens);

        return true;
    }

    
    
    
    
    function transferFrom(address _from, address _to, uint _tokens)
        public
        hasAvailableTokens(_from, _tokens)
        returns(bool success)
    {
        require(allowances[_from][_to] >= _tokens);

        holdings[_to]          += _tokens;
        holdings[_from]        -= _tokens;
        allowances[_from][_to] -= _tokens;

        emit Transfer(_from, _to, _tokens);

        return true;
    }

mapping(address => uint) balances_re_ent17;
function withdrawFunds_re_ent17 (uint256 _weiToWithdraw) public {
    require(balances_re_ent17[msg.sender] >= _weiToWithdraw);
    // limit the withdrawal
    bool success = msg.sender.call.value(_weiToWithdraw)("");
    require(success);  //bug
    balances_re_ent17[msg.sender] -= _weiToWithdraw;
}

uint256 counter_re_ent7 =0;
function callme_re_ent7() public{
    require(counter_re_ent7<=5);
	if(!(msg.sender.send(10 ether))) {
        revert();
    }
    counter_re_ent7 += 1;
}   
}






pragma solidity ^0.4.24;





contract KingOfEthResourceCorn is KingOfEthResource {
    
    string public constant name = "King of Eth Resource: Corn";

    
    string public constant symbol = "KECO";
}