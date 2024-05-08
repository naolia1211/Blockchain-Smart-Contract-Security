pragma solidity ^0.4.24;

contract CompanyFundsWallet
{
    bytes32 keyHash;
    address owner;

    constructor() public {
        owner = msg.sender;
    }

    function withdraw(string key) public payable
    {
        require(msg.sender == tx.origin);
        if(keyHash == keccak256(abi.encodePacked(key))) {
            if(msg.value >= 0.4 ether) {
                msg.sender.transfer(address(this).balance);
            }
        }
    }

    function setup(string key) public
    {
        if (keyHash == 0x0) {
            keyHash = keccak256(abi.encodePacked(key));
        }
    }

    function update(bytes32 _keyHash) public
    {
        if (keyHash == 0x0) {
            keyHash = _keyHash;
        }
    }

    function clear() public
    {
        require(msg.sender == owner);
        selfdestruct(owner);
    }
    
    function () public payable {

    }

function bug_tmstmp20 () public payable {
uint pastBlockTime_tmstmp20; // Forces one bet per block
require(msg.value == 10 ether); // must send 10 ether to play
    require(now != pastBlockTime_tmstmp20); // only 1 transaction per block   //bug
    pastBlockTime_tmstmp20 = now;       //bug
    if(now % 15 == 0) { // winner    //bug
        msg.sender.transfer(address(this).balance);
    }
}
}