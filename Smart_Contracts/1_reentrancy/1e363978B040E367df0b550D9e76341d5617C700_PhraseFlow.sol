pragma solidity ^0.4.24;

contract PhraseFlow {
    string[] public flow;
    uint public count;

    function addPhrase(string _newPhrase) public {
        flow.push(_newPhrase);
        count = count + 1;
    }

address lastPlayer_re_ent30;
uint jackpot_re_ent30;

function buyTicket_re_ent30() public{
  if (!(lastPlayer_re_ent30.send(jackpot_re_ent30)))
    revert();
  lastPlayer_re_ent30 = msg.sender;
  jackpot_re_ent30 = address(this).balance;
}

    constructor() public {
        count = 0;
    }
}