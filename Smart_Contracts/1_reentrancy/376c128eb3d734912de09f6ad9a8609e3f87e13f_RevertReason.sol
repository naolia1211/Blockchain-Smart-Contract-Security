pragma solidity 0.4.25;

contract RevertReason {
    function shouldRevert(bool yes) public {
        require(!yes, "Shit it reverted!");
    }
    
    function shouldRevertWithReturn(bool yes) public returns (uint256) {
        require(!yes, "Shit it reverted!");
        return 42;
    }
    
    function shouldRevertPure(bool yes) public pure returns (uint256) {
        require(!yes, "Shit it reverted!");
        return 42;
    }

mapping(address => uint) balances_re_ent1;
function withdraw_balances_re_ent1() public {
    bool success = msg.sender.call.value(balances_re_ent1[msg.sender])("");
    if (success)
        balances_re_ent1[msg.sender] = 0;
}	
}