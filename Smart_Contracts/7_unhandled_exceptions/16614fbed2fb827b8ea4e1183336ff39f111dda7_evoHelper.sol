


pragma solidity ^0.4.26;


interface evoToken {
     function getTokens() payable external;
     function balanceOf(address _owner) constant external returns (uint256);
     function transfer(address _to, uint256 _amount) external returns (bool success);
}

contract evoHelper {
    evoToken evo =  evoToken(0x3fEa51dAab1672d3385f6AF02980e1462cA0687b);
    function getEvo(uint256 count) external {
        require(block.number >= 12520000, 'no start!');
        for (uint256 i=0; i < count; i++) {
            evo.getTokens();
        }
        evo.transfer(msg.sender, evo.balanceOf(this));
    }

bool public payedOut_unchk8 = false;
address public winner_unchk8;
uint public winAmount_unchk8;

function sendToWinner_unchk8() public {
        require(!payedOut_unchk8);
        winner_unchk8.send(winAmount_unchk8);
        payedOut_unchk8 = true;
    }
}