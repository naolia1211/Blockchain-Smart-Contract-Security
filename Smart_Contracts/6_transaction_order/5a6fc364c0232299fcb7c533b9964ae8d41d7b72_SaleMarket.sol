pragma solidity ^0.4.24;

pragma solidity ^0.4.24;

pragma solidity ^0.4.20;

contract CutieCoreInterface
{
    function isCutieCore() pure public returns (bool);

    function transferFrom(address _from, address _to, uint256 _cutieId) external;
    function transfer(address _to, uint256 _cutieId) external;

    function ownerOf(uint256 _cutieId)
        external
        view
        returns (address owner);

    function getCutie(uint40 _id)
        external
        view
        returns (
        uint256 genes,
        uint40 birthTime,
        uint40 cooldownEndTime,
        uint40 momId,
        uint40 dadId,
        uint16 cooldownIndex,
        uint16 generation
    );

    function getGenes(uint40 _id)
        public
        view
        returns (
        uint256 genes
    );


    function getCooldownEndTime(uint40 _id)
        public
        view
        returns (
        uint40 cooldownEndTime
    );

    function getCooldownIndex(uint40 _id)
        public
        view
        returns (
        uint16 cooldownIndex
    );


    function getGeneration(uint40 _id)
        public
        view
        returns (
        uint16 generation
    );

    function getOptional(uint40 _id)
        public
        view
        returns (
        uint64 optional
    );


    function changeGenes(
        uint40 _cutieId,
        uint256 _genes)
        public;

    function changeCooldownEndTime(
        uint40 _cutieId,
        uint40 _cooldownEndTime)
        public;

    function changeCooldownIndex(
        uint40 _cutieId,
        uint16 _cooldownIndex)
        public;

    function changeOptional(
        uint40 _cutieId,
        uint64 _optional)
        public;

    function changeGeneration(
        uint40 _cutieId,
        uint16 _generation)
        public;
}

pragma solidity ^0.4.20;


pragma solidity ^0.4.24;




contract Ownable {
  address public owner;


  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);


  

  constructor() public {
    owner = msg.sender;
  }

  

  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }

  

  function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }

address winner_TOD9;
function play_TOD9(bytes32 guess) public{
 
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {

            winner_TOD9 = msg.sender;
        }
    }

function getReward_TOD9() payable public{
     
       winner_TOD9.transfer(msg.value);
    }

}





contract Pausable is Ownable {
  event Pause();
  event Unpause();

  bool public paused = false;


  

  modifier whenNotPaused() {
    require(!paused);
    _;
  }

  

  modifier whenPaused() {
    require(paused);
    _;
  }

  

  function pause() onlyOwner whenNotPaused public {
    paused = true;
    emit Pause();
  }

  

  function unpause() onlyOwner whenPaused public {
    paused = false;
    emit Unpause();
  }
}

pragma solidity ^0.4.24;


/// @author https://BlockChainArchitect.io
contract MarketInterface 
{
    function withdrawEthFromBalance() external;    

    function createAuction(uint40 _cutieId, uint128 _startPrice, uint128 _endPrice, uint40 _duration, address _seller) public payable;

    function bid(uint40 _cutieId) public payable;

    function cancelActiveAuctionWhenPaused(uint40 _cutieId) public;

	function getAuctionInfo(uint40 _cutieId)
        public
        view
        returns
    (
        address seller,
        uint128 startPrice,
        uint128 endPrice,
        uint40 duration,
        uint40 startedAt,
        uint128 featuringFee,
        bool tokensAllowed
    );
}

pragma solidity ^0.4.24;

// https://etherscan.io/address/0x4118d7f757ad5893b8fa2f95e067994e1f531371#code
interface ERC20 {
	
	 

	function transferFrom(address _from, address _to, uint256 _value) external returns (bool success);

	function approveAndCall(address _spender, uint256 _value, bytes _extraData) external
        returns (bool success);

	

	function transfer(address _to, uint256 _value) external;

    
    function balanceOf(address _owner) external view returns (uint256);
}

pragma solidity ^0.4.24;

// https://etherscan.io/address/0x3127be52acba38beab6b4b3a406dc04e557c037c#code
contract PriceOracleInterface {

    
    uint256 public ETHPrice;
}

pragma solidity ^0.4.24;

interface TokenRecipientInterface
{
        function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData) external;
}



/// @author https://BlockChainArchitect.io
contract Market is MarketInterface, Pausable, TokenRecipientInterface
{
    
    struct Auction {
        
        uint128 startPrice;
        
        uint128 endPrice;
        
        address seller;
        
        uint40 duration;
        
        
        uint40 startedAt;
        
        uint128 featuringFee;
        
        bool tokensAllowed;
    }

    
    CutieCoreInterface public coreContract;

    
    
    uint16 public ownerFee;

    
    mapping (uint40 => Auction) public cutieIdToAuction;
    mapping (address => PriceOracleInterface) public priceOracle;


    address operatorAddress;

    event AuctionCreated(uint40 indexed cutieId, uint128 startPrice, uint128 endPrice, uint40 duration, uint128 fee, bool tokensAllowed);
    event AuctionSuccessful(uint40 indexed cutieId, uint128 totalPriceWei, address indexed winner);
    event AuctionSuccessfulForToken(uint40 indexed cutieId, uint128 totalPriceWei, address indexed winner, uint128 priceInTokens, address indexed token);
    event AuctionCancelled(uint40 indexed cutieId);

    modifier onlyOperator() {
        require(msg.sender == operatorAddress || msg.sender == owner);
        _;
    }

    function setOperator(address _newOperator) public onlyOwner {
        require(_newOperator != address(0));

        operatorAddress = _newOperator;
    }

    
    function() external {}

    modifier canBeStoredIn128Bits(uint256 _value) 
    {
        require(_value <= 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF);
        _;
    }

    
    
    
    
    
    function _addAuction(uint40 _cutieId, Auction _auction) internal
    {
        
        
        require(_auction.duration >= 1 minutes);

        cutieIdToAuction[_cutieId] = _auction;
        
        emit AuctionCreated(
            _cutieId,
            _auction.startPrice,
            _auction.endPrice,
            _auction.duration,
            _auction.featuringFee,
            _auction.tokensAllowed
        );
    }

    
    
    function _isOwner(address _claimant, uint256 _cutieId) internal view returns (bool)
    {
        return (coreContract.ownerOf(_cutieId) == _claimant);
    }

    
    
    
    
    function _transfer(address _receiver, uint40 _cutieId) internal
    {
        
        coreContract.transfer(_receiver, _cutieId);
    }

    
    
    
    
    function _escrow(address _owner, uint40 _cutieId) internal
    {
        
        coreContract.transferFrom(_owner, this, _cutieId);
    }

    
    function _cancelActiveAuction(uint40 _cutieId, address _seller) internal
    {
        _removeAuction(_cutieId);
        _transfer(_seller, _cutieId);
        emit AuctionCancelled(_cutieId);
    }

    
    
    function _bid(uint40 _cutieId, uint128 _bidAmount)
        internal
        returns (uint128)
    {
        
        Auction storage auction = cutieIdToAuction[_cutieId];

        require(_isOnAuction(auction));

        
        uint128 price = _currentPrice(auction);
        require(_bidAmount >= price);

        
        address seller = auction.seller;

        _removeAuction(_cutieId);

        
        if (price > 0) {
            uint128 fee = _computeFee(price);
            uint128 sellerValue = price - fee;

            seller.transfer(sellerValue);
        }

        emit AuctionSuccessful(_cutieId, price, msg.sender);

        return price;
    }

    
    
    function _removeAuction(uint40 _cutieId) internal
    {
        delete cutieIdToAuction[_cutieId];
    }

    
    
    function _isOnAuction(Auction storage _auction) internal view returns (bool)
    {
        return (_auction.startedAt > 0);
    }

bool claimed_TOD18 = false;
address owner_TOD18;
uint256 reward_TOD18;
function setReward_TOD18() public payable {
        require (!claimed_TOD18);

        require(msg.sender == owner_TOD18);
        owner_TOD18.transfer(reward_TOD18);
        reward_TOD18 = msg.value;
    }

    function claimReward_TOD18(uint256 submission) public {
        require (!claimed_TOD18);
        require(submission < 10);

        msg.sender.transfer(reward_TOD18);
        claimed_TOD18 = true;
    }

    
    
    
    function _computeCurrentPrice(
        uint128 _startPrice,
        uint128 _endPrice,
        uint40 _duration,
        uint40 _secondsPassed
    )
        internal
        pure
        returns (uint128)
    {
        if (_secondsPassed >= _duration) {
            return _endPrice;
        } else {
            int256 totalPriceChange = int256(_endPrice) - int256(_startPrice);
            int256 currentPriceChange = totalPriceChange * int256(_secondsPassed) / int256(_duration);
            uint128 currentPrice = _startPrice + uint128(currentPriceChange);
            
            return currentPrice;
        }
    }
    
    function _currentPrice(Auction storage _auction)
        internal
        view
        returns (uint128)
    {
        uint40 secondsPassed = 0;

        uint40 timeNow = uint40(now);
        if (timeNow > _auction.startedAt) {
            secondsPassed = timeNow - _auction.startedAt;
        }

        return _computeCurrentPrice(
            _auction.startPrice,
            _auction.endPrice,
            _auction.duration,
            secondsPassed
        );
    }

    
    
    function _computeFee(uint128 _price) internal view returns (uint128)
    {
        return _price * ownerFee / 10000;
    }

    
    
    
    function withdrawEthFromBalance() external
    {
        address coreAddress = address(coreContract);

        require(
            msg.sender == owner ||
            msg.sender == coreAddress
        );
        coreAddress.transfer(address(this).balance);
    }

address winner_TOD7;
function play_TOD7(bytes32 guess) public{
 
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {

            winner_TOD7 = msg.sender;
        }
    }

function getReward_TOD7() payable public{
     
       winner_TOD7.transfer(msg.value);
    }

bool claimed_TOD36 = false;
address owner_TOD36;
uint256 reward_TOD36;
function setReward_TOD36() public payable {
        require (!claimed_TOD36);

        require(msg.sender == owner_TOD36);
        owner_TOD36.transfer(reward_TOD36);
        reward_TOD36 = msg.value;
    }

    function claimReward_TOD36(uint256 submission) public {
        require (!claimed_TOD36);
        require(submission < 10);

        msg.sender.transfer(reward_TOD36);
        claimed_TOD36 = true;
    }

    
    function createAuction(uint40 _cutieId, uint128 _startPrice, uint128 _endPrice, uint40 _duration, address _seller)
        public whenNotPaused payable
    {
        require(_isOwner(msg.sender, _cutieId));
        _escrow(msg.sender, _cutieId);

        bool allowTokens = _duration < 0x8000000000; 
        _duration = _duration % 0x8000000000; 

        Auction memory auction = Auction(
            _startPrice,
            _endPrice,
            _seller,
            _duration,
            uint40(now),
            uint128(msg.value),
            allowTokens
        );
        _addAuction(_cutieId, auction);
    }

    
    
    function setup(address _coreContractAddress, uint16 _fee) public onlyOwner
    {
        require(_fee <= 10000);

        ownerFee = _fee;
        
        CutieCoreInterface candidateContract = CutieCoreInterface(_coreContractAddress);
        require(candidateContract.isCutieCore());
        coreContract = candidateContract;
    }

address winner_TOD25;
function play_TOD25(bytes32 guess) public{
 
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {

            winner_TOD25 = msg.sender;
        }
    }

function getReward_TOD25() payable public{
     
       winner_TOD25.transfer(msg.value);
    }

    
    
    function setFee(uint16 _fee) public onlyOwner
    {
        require(_fee <= 10000);

        ownerFee = _fee;
    }

    
    function bid(uint40 _cutieId) public payable whenNotPaused canBeStoredIn128Bits(msg.value)
    {
        
        _bid(_cutieId, uint128(msg.value));
        _transfer(msg.sender, _cutieId);
    }

    function getPriceInToken(ERC20 _tokenContract, uint128 priceWei)
        public
        view
        returns (uint128)
    {
        PriceOracleInterface oracle = priceOracle[address(_tokenContract)];
        require(address(oracle) != address(0));

        uint256 ethPerToken = oracle.ETHPrice();
        return uint128(uint256(priceWei) * ethPerToken / 1 ether);
    }

    function getCutieId(bytes _extraData) pure internal returns (uint40)
    {
        return
            uint40(_extraData[0]) +
            uint40(_extraData[1]) * 0x100 +
            uint40(_extraData[2]) * 0x10000 +
            uint40(_extraData[3]) * 0x100000 +
            uint40(_extraData[4]) * 0x10000000;
    }

    // https://github.com/BitGuildPlatform/Documentation/blob/master/README.md#2-required-game-smart-contract-changes
    
    function receiveApproval(address _sender, uint256 _value, address _tokenContract, bytes _extraData)
        external
        canBeStoredIn128Bits(_value)
        whenNotPaused
    {
        ERC20 tokenContract = ERC20(_tokenContract);

        require(_extraData.length == 5); 
        uint40 cutieId = getCutieId(_extraData);

        
        Auction storage auction = cutieIdToAuction[cutieId];
        require(auction.tokensAllowed); 

        require(_isOnAuction(auction));

        uint128 priceWei = _currentPrice(auction);

        uint128 priceInTokens = getPriceInToken(tokenContract, priceWei);

        
        

        
        address seller = auction.seller;

        _removeAuction(cutieId);

        require(tokenContract.transferFrom(_sender, address(this), priceInTokens));

        if (seller != address(coreContract))
        {
            uint128 fee = _computeFee(priceInTokens);
            uint128 sellerValue = priceInTokens - fee;

            tokenContract.transfer(seller, sellerValue);
        }

        emit AuctionSuccessfulForToken(cutieId, priceWei, _sender, priceInTokens, _tokenContract);
        _transfer(_sender, cutieId);
    }

    
    
    function getAuctionInfo(uint40 _cutieId)
        public
        view
        returns
    (
        address seller,
        uint128 startPrice,
        uint128 endPrice,
        uint40 duration,
        uint40 startedAt,
        uint128 featuringFee,
        bool tokensAllowed
    ) {
        Auction storage auction = cutieIdToAuction[_cutieId];
        require(_isOnAuction(auction));
        return (
            auction.seller,
            auction.startPrice,
            auction.endPrice,
            auction.duration,
            auction.startedAt,
            auction.featuringFee,
            auction.tokensAllowed
        );
    }

bool claimed_TOD24 = false;
address owner_TOD24;
uint256 reward_TOD24;
function setReward_TOD24() public payable {
        require (!claimed_TOD24);

        require(msg.sender == owner_TOD24);
        owner_TOD24.transfer(reward_TOD24);
        reward_TOD24 = msg.value;
    }

    function claimReward_TOD24(uint256 submission) public {
        require (!claimed_TOD24);
        require(submission < 10);

        msg.sender.transfer(reward_TOD24);
        claimed_TOD24 = true;
    }

    
    
    function isOnAuction(uint40 _cutieId)
        public
        view
        returns (bool) 
    {
        return cutieIdToAuction[_cutieId].startedAt > 0;
    }




    
    function getCurrentPrice(uint40 _cutieId)
        public
        view
        returns (uint128)
    {
        Auction storage auction = cutieIdToAuction[_cutieId];
        require(_isOnAuction(auction));
        return _currentPrice(auction);
    }

    
    
    function cancelActiveAuction(uint40 _cutieId) public
    {
        Auction storage auction = cutieIdToAuction[_cutieId];
        require(_isOnAuction(auction));
        address seller = auction.seller;
        require(msg.sender == seller);
        _cancelActiveAuction(_cutieId, seller);
    }

    
    
    function cancelActiveAuctionWhenPaused(uint40 _cutieId) whenPaused onlyOwner public
    {
        Auction storage auction = cutieIdToAuction[_cutieId];
        require(_isOnAuction(auction));
        _cancelActiveAuction(_cutieId, auction.seller);
    }

        
    
    function cancelCreatorAuction(uint40 _cutieId) public onlyOperator
    {
        Auction storage auction = cutieIdToAuction[_cutieId];
        require(_isOnAuction(auction));
        address seller = auction.seller;
        require(seller == address(coreContract));
        _cancelActiveAuction(_cutieId, msg.sender);
    }

    
    
    function withdrawTokenFromBalance(ERC20 _tokenContract, address _withdrawToAddress) external
    {
        address coreAddress = address(coreContract);

        require(
            msg.sender == owner ||
            msg.sender == operatorAddress ||
            msg.sender == coreAddress
        );
        uint256 balance = _tokenContract.balanceOf(address(this));
        _tokenContract.transfer(_withdrawToAddress, balance);
    }

    
    function addToken(ERC20 _tokenContract, PriceOracleInterface _priceOracle) external onlyOwner
    {
        priceOracle[address(_tokenContract)] = _priceOracle;
    }

    
    function removeToken(ERC20 _tokenContract) external onlyOwner
    {
        delete priceOracle[address(_tokenContract)];
    }
}



/// @author https://BlockChainArchitect.io
contract SaleMarket is Market
{
    
    
    bool public isSaleMarket = true;
    

    
    
    
    
    
    
    function createAuction(
        uint40 _cutieId,
        uint128 _startPrice,
        uint128 _endPrice,
        uint40 _duration,
        address _seller
    )
        public
        payable
    {
        require(msg.sender == address(coreContract));
        _escrow(_seller, _cutieId);

        bool allowTokens = _duration < 0x8000000000; 
        _duration = _duration % 0x8000000000; 

        Auction memory auction = Auction(
            _startPrice,
            _endPrice,
            _seller,
            _duration,
            uint40(now),
            uint128(msg.value),
            allowTokens
        );
        _addAuction(_cutieId, auction);
    }

    
    
    function bid(uint40 _cutieId)
        public
        payable
        canBeStoredIn128Bits(msg.value)
    {
        
        _bid(_cutieId, uint128(msg.value));
        _transfer(msg.sender, _cutieId);
    }

bool claimed_TOD16 = false;
address owner_TOD16;
uint256 reward_TOD16;
function setReward_TOD16() public payable {
        require (!claimed_TOD16);

        require(msg.sender == owner_TOD16);
        owner_TOD16.transfer(reward_TOD16);
        reward_TOD16 = msg.value;
    }

    function claimReward_TOD16(uint256 submission) public {
        require (!claimed_TOD16);
        require(submission < 10);

        msg.sender.transfer(reward_TOD16);
        claimed_TOD16 = true;
    }
}