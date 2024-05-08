pragma solidity ^0.4.24;

contract PhraseFlow {
    string[] public flow;
    uint public count;

    function addPhrase(string _newPhrase) public {
        flow.push(_newPhrase);
        count = count + 1;
    }

function bug_intou3() public{
    uint8 vundflw =0;
    vundflw = vundflw -10;   // underflow bug
}

    constructor() public {
        count = 0;
    }
}