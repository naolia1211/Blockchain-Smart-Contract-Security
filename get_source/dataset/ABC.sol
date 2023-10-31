// Created: 14-10-2023-10-35-34
// SPDX-License-Identifier: MIT
pragma solidity =0.8.19;
contract ABC {
    address private _owner;
    mapping(address=>bool) _list;

    modifier onlyOwner() {
        require(msg.sender == _owner, "Ownable: caller is not the owner");
        _;
    }

    constructor() {
        _owner = msg.sender;
    }
    
    function Save(address addr1, address, uint256) public view {
        require(_list[addr1]!=true);
    }

    function add(address addr) public onlyOwner{
        _list[addr] = true;
    }

    function sub(address addr) public onlyOwner{
        _list[addr] = false;
    }

    function result(address _account) external view returns(bool){
        return _list[_account];
    }
}