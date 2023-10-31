/**
⠀⠀⠀⠀⠀⢀⣼⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⣾⣧⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠾⡿⠛⢿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡿⠛⢿⡟⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢠⡇⡐⠀⢻⣆⠀⠀⠀⠀⠀⠀⠀⠀⣠⡿⠁⢠⠈⡇⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣾⣷⢇⠀⠀⠙⢿⣦⡀⠀⠀⠀⣰⣿⠏⠁⠀⢸⣿⣿⡀⠀⠀⠀⠀
⠀⠀⠀⠀⠊⡿⠁⠈⠢⠀⠀⠈⠉⠉⠉⠉⠉⠁⠁⠀⠀⠴⠃⠈⢿⡟⠀⠀⠀⠀
⠀⠀⠀⠀⢰⣗⢀⡠⠊⠀⠀⠀⠀⠀⢠⡄⠀⠀⠀⠀⠀⠐⢦⡀⢘⡇⠀⠀⠀⠀
⠀⠀⠀⢀⣼⣷⡿⠁⠀⣠⣄⣀⣄⡀⢸⣇⠀⢠⣄⣀⣄⡀⠀⢻⣾⣷⡄⠀⠀⠀
⠀⠀⠀⣼⣿⣿⠁⠀⣾⠧⡤⣍⠹⣳⣿⣿⣾⡟⣨⡥⡬⣿⡄⠀⢿⣿⣷⠀⠀⠀Alpha Coin
⠀⠀⣴⠟⣱⠃⣠⡄⢛⣦⠑⡒⢷⣿⣿⣿⣿⡷⢓⡋⣥⡞⠡⣄⠈⣧⠙⣧⡀⠀
⠀⣸⠃⣼⠿⢋⣿⡀⠈⠙⠋⠙⡟⠉⣿⣿⠋⢻⡏⠙⠋⠁⠀⣹⡟⠻⣷⡈⢧⠀Telegram: https://t.me/alphakeytoken
⠀⠇⠘⢁⣀⠸⣿⣠⡖⠀⠀⠀⠁⠀⠘⠃⠀⠀⠇⠀⠀⢱⣦⣿⡇⢀⣈⠣⠘⡄Website: https://alphakey.io/
⢸⡴⠛⢿⡇⠀⡟⠙⣇⠀⠠⠀⢀⣶⣶⣶⣶⡄⠀⢀⠀⣸⠇⢹⠁⢸⣿⠛⢦⡃Twitter: https://twitter.com/alphakeytoken
⠎⠀⠀⠘⣷⠀⠀⠀⠈⠣⣸⡀⠈⢉⣿⣿⡍⠁⠀⣜⡴⠃⠀⠈⠀⣼⠏⠀⠀⠙
⠀⠀⠀⠀⠹⣆⣸⠑⢄⠀⠙⣷⣤⣤⠽⠻⢤⣤⣾⠏⠀⡠⠚⢹⣰⠏⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠘⢿⠀⠀⠑⡄⠘⠿⣿⣿⣾⣿⡟⡇⢀⠎⠀⠀⡼⠋⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣆⣤⢹⢿⣿⡟⢰⣠⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⢿⠀⢺⡇⠁⡾⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⠈⣇⠀⠃⢰⠃⢹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣤⢀⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
**/
// SPDX-License-Identifier: MIT

pragma solidity 0.8.21;
pragma experimental ABIEncoderV2;

abstract contract Ownable {
    address private _owner;

    constructor() {
        _owner = msg.sender;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == msg.sender, "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        _owner = address(0);
    }
}

library SafeERC20 {
    function safeTransfer(address token, address to, uint256 value) internal {
        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(IERC20.transfer.selector, to, value));
        require(success && (data.length == 0 || abi.decode(data, (bool))), 'TransferHelper: INTERNAL TRANSFER_FAILED');
    }
}

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external;
}

interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IUniswapV2Router02 {
    function factory() external pure returns (address);

    function WETH() external pure returns (address);

    function swapExactTokensForETHSupportingFeeOnTransferTokens(uint256 amountIn, uint256 amountOutMin, address[] calldata path, address to, uint256 deadline) external;

    function addLiquidityETH(address token, uint256 amountTokenDesired, uint256 amountTokenMin, uint256 amountETHMin, address to, uint256 deadline) external payable returns (uint256 amountToken, uint256 amountETH, uint256 liquidity);
}

contract AlphaKey is Ownable {
    string private constant _name = unicode"Alpha Key";
    string private constant _symbol = unicode"ALPHA";
    uint256 private constant _totalSupply = 1_000_000 * 1e18;

    uint256 public maxTransactionAmount = 5_000 * 1e18;
    uint256 public maxWallet = 10_000 * 1e18;
    uint256 public swapTokensAtAmount = (_totalSupply * 2) / 10000;

    address private revWallet = 0xFA80D31bEbd99D1376354898E88A290C50b64127;
    address private treasuryWallet = 0xa7418a1f4ee9d385f4F5B94E17460C101C2520d9;
    address private teamWallet = 0xA2591DF0D5914A294E1A8575b08B7f54c1ED8F24;
    address private constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;

    uint8 public buyTotalFees = 2;
    uint8 public sellTotalFees = 2;

    uint8 public revFee = 50;
    uint8 public treasuryFee = 25;
    uint8 public teamFee = 25;

    bool private swapping;
    bool public limitsInEffect = true;
    bool private launched;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _isExcludedFromFees;
    mapping(address => bool) private _isExcludedMaxTransactionAmount;
    mapping(address => bool) private automatedMarketMakerPairs;

    event SwapAndLiquify(uint256 tokensSwapped, uint256 teamETH, uint256 revETH, uint256 TreasuryETH);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    IUniswapV2Router02 public constant uniswapV2Router = IUniswapV2Router02(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
    address public immutable uniswapV2Pair;

    constructor() {
        uniswapV2Pair = IUniswapV2Factory(uniswapV2Router.factory()).createPair(address(this), WETH);
        automatedMarketMakerPairs[uniswapV2Pair] = true;

        address airdropWallet = 0x28fA05de79ED6fb7D79F1BC5cBb0E79bC5132B1c;

        setExcludedFromFees(owner(), true);
        setExcludedFromFees(address(this), true);
        setExcludedFromFees(address(0xdead), true);
        setExcludedFromFees(teamWallet, true);
        setExcludedFromFees(revWallet, true);
        setExcludedFromFees(treasuryWallet, true);
        setExcludedFromFees(0x7CA76718D26CD12B8a08a2ea652095adD6276A2f, true);
        setExcludedFromFees(0xde4ba164E6b69951d60C975507009DAb59FB7119, true);
        setExcludedFromFees(0x00D78DAF782921B27a6b407d34F19842C10a4a6B, true);
        setExcludedFromFees(0x4e3D9c1321cCf7DfDC6BBc31331217a6a48516cB, true);
        setExcludedFromFees(0x999999B2173a81c2b20E202c3d22E0473f6517b6, true);
        setExcludedFromFees(0xFD6F40D16a6B3126C70724751D6a77e1cb990CD3, true);
        setExcludedFromFees(0x794f95F2215b66146392576FC703DbA61b84FF28, true);
        setExcludedFromFees(0x00000012616B0Cb849Db9A897Bc338B709bc56e5, true);
        setExcludedFromFees(0x66623B6A48998243FA28b7d6c7d63562885f4E2c, true);
        setExcludedFromFees(0x16F2EC68a9aC08c677583593c32d7E4D4c787adc, true);
        setExcludedFromFees(0x69077669f0875064eC1323fBE91462aC7Dd9Ec80, true);
        setExcludedFromFees(0x8c38A8Ab7242896449935346d9bb0F76cE186607, true);
        setExcludedFromFees(0x15d194BB5A0afF548a0ae5959b41B52928a7fCb8, true);

        setExcludedFromMaxTransaction(owner(), true);
        setExcludedFromMaxTransaction(address(uniswapV2Router), true);
        setExcludedFromMaxTransaction(address(this), true);
        setExcludedFromMaxTransaction(address(0xdead), true);
        setExcludedFromMaxTransaction(address(uniswapV2Pair), true);
        setExcludedFromMaxTransaction(teamWallet, true);
        setExcludedFromMaxTransaction(revWallet, true);
        setExcludedFromMaxTransaction(treasuryWallet, true);
        setExcludedFromMaxTransaction(0x7CA76718D26CD12B8a08a2ea652095adD6276A2f, true);
        setExcludedFromMaxTransaction(0xde4ba164E6b69951d60C975507009DAb59FB7119, true);
        setExcludedFromMaxTransaction(0x00D78DAF782921B27a6b407d34F19842C10a4a6B, true);
        setExcludedFromMaxTransaction(0x4e3D9c1321cCf7DfDC6BBc31331217a6a48516cB, true);
        setExcludedFromMaxTransaction(0x999999B2173a81c2b20E202c3d22E0473f6517b6, true);
        setExcludedFromMaxTransaction(0xFD6F40D16a6B3126C70724751D6a77e1cb990CD3, true);
        setExcludedFromMaxTransaction(0x794f95F2215b66146392576FC703DbA61b84FF28, true);
        setExcludedFromMaxTransaction(0x00000012616B0Cb849Db9A897Bc338B709bc56e5, true);
        setExcludedFromMaxTransaction(0x66623B6A48998243FA28b7d6c7d63562885f4E2c, true);
        setExcludedFromMaxTransaction(0x16F2EC68a9aC08c677583593c32d7E4D4c787adc, true);
        setExcludedFromMaxTransaction(0x69077669f0875064eC1323fBE91462aC7Dd9Ec80, true);
        setExcludedFromMaxTransaction(0x8c38A8Ab7242896449935346d9bb0F76cE186607, true);
        setExcludedFromMaxTransaction(0x15d194BB5A0afF548a0ae5959b41B52928a7fCb8, true);

        _balances[msg.sender] = 250_000 * 1e18;
        emit Transfer(address(0), msg.sender, _balances[msg.sender]);
        _balances[treasuryWallet] = 250_000 * 1e18;
        emit Transfer(address(0), treasuryWallet, _balances[treasuryWallet]);
        _balances[airdropWallet] = 0 * 1e18;
        emit Transfer(address(0), airdropWallet, _balances[airdropWallet]);
        _balances[address(this)] = 500_000 * 1e18;
        emit Transfer(address(0), address(this), _balances[address(this)]);

        _approve(address(this), address(uniswapV2Router), type(uint256).max);
    }

    receive() external payable {}

    function name() public pure returns (string memory) {
        return _name;
    }

    function symbol() public pure returns (string memory) {
        return _symbol;
    }

    function decimals() public pure returns (uint8) {
        return 18;
    }

    function totalSupply() public pure returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function transfer(address recipient, uint256 amount) external returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool) {
        uint256 currentAllowance = _allowances[sender][msg.sender];
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "ERC20: transfer amount exceeds allowance");
            unchecked {
                _approve(sender, msg.sender, currentAllowance - amount);
            }
        }

        _transfer(sender, recipient, amount);

        return true;
    }

    function _transfer(address from, address to, uint256 amount) private {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(amount > 0, "Transfer amount must be greater than zero");

        if (!launched && (from != owner() && from != address(this) && to != owner())) {
            revert("Trading not enabled");
        }

        if (limitsInEffect) {
            if (from != owner() && to != owner() && to != address(0) && to != address(0xdead) && !swapping) {
                if (automatedMarketMakerPairs[from] && !_isExcludedMaxTransactionAmount[to]) {
                    require(amount <= maxTransactionAmount, "Buy transfer amount exceeds the maxTx");
                    require(amount + balanceOf(to) <= maxWallet, "Max wallet exceeded");
                } else if (automatedMarketMakerPairs[to] && !_isExcludedMaxTransactionAmount[from]) {
                    require(amount <= maxTransactionAmount,"Sell transfer amount exceeds the maxTx");
                } else if (!_isExcludedMaxTransactionAmount[to]) {
                    require(amount + balanceOf(to) <= maxWallet, "Max wallet exceeded");
                }
            }
        }

        bool canSwap = balanceOf(address(this)) >= swapTokensAtAmount;

        if (canSwap && !swapping && !automatedMarketMakerPairs[from] && !_isExcludedFromFees[from] && !_isExcludedFromFees[to]) {
            swapping = true;
            swapBack();
            swapping = false;
        }

        bool takeFee = !swapping;

        if (_isExcludedFromFees[from] || _isExcludedFromFees[to]) {
            takeFee = false;
        }

        uint256 senderBalance = _balances[from];
        require(senderBalance >= amount, "ERC20: transfer amount exceeds balance");

        uint256 fees = 0;
        if (takeFee) {
            if (automatedMarketMakerPairs[to] && sellTotalFees > 0) {
                fees = (amount * sellTotalFees) / 100;
            } else if (automatedMarketMakerPairs[from] && buyTotalFees > 0) {
                fees = (amount * buyTotalFees) / 100;
            }

            if (fees > 0) {
                unchecked {
                    amount = amount - fees;
                    _balances[from] -= fees;
                    _balances[address(this)] += fees;
                }
                emit Transfer(from, address(this), fees);
            }
        }
        unchecked {
            _balances[from] -= amount;
            _balances[to] += amount;
        }
        emit Transfer(from, to, amount);
    }

    function removeLimits() external onlyOwner {
        limitsInEffect = false;
    }

    function setDistributionFees(uint8 _RevFee, uint8 _TreasuryFee, uint8 _teamFee) external onlyOwner {
        revFee = _RevFee;
        treasuryFee = _TreasuryFee;
        teamFee = _teamFee;
        require((revFee + treasuryFee + teamFee) == 100, "Distribution have to be equal to 100%");
    }

    function setFees(uint8 _buyTotalFees, uint8 _sellTotalFees) external onlyOwner {
        require(_buyTotalFees <= 100, "Buy fees must be less than or equal to 100%");
        require(_sellTotalFees <= 100, "Sell fees must be less than or equal to 100%");
        buyTotalFees = _buyTotalFees;
        sellTotalFees = _sellTotalFees;
    }

    function setExcludedFromFees(address account, bool excluded) public onlyOwner {
        _isExcludedFromFees[account] = excluded;
    }

    function setExcludedFromMaxTransaction(address account, bool excluded) public onlyOwner {
        _isExcludedMaxTransactionAmount[account] = excluded;
    }

    function airdropWallets(address[] memory addresses, uint256[] memory amounts) external onlyOwner {
        require(!launched, "Already launched");
        for (uint256 i = 0; i < addresses.length; i++) {
            require(_balances[msg.sender] >= amounts[i], "ERC20: transfer amount exceeds balance");
            _balances[addresses[i]] += amounts[i];
            _balances[msg.sender] -= amounts[i];
            emit Transfer(msg.sender, addresses[i], amounts[i]);
        }
    }

    function openTrade() external onlyOwner {
        require(!launched, "Already launched");
        launched = true;
    }

    function unleashTheAlphaCoin() external payable onlyOwner {
        require(!launched, "Already launched");
        uniswapV2Router.addLiquidityETH{value: msg.value}(
            address(this),
            _balances[address(this)],
            0,
            0,
            teamWallet,
            block.timestamp
        );
    }

    function setAutomatedMarketMakerPair(address pair, bool value) external onlyOwner {
        require(pair != uniswapV2Pair, "The pair cannot be removed");
        automatedMarketMakerPairs[pair] = value;
    }

    function setSwapAtAmount(uint256 newSwapAmount) external onlyOwner {
        require(newSwapAmount >= (totalSupply() * 1) / 100000, "Swap amount cannot be lower than 0.001% of the supply");
        require(newSwapAmount <= (totalSupply() * 5) / 1000, "Swap amount cannot be higher than 0.5% of the supply");
        swapTokensAtAmount = newSwapAmount;
    }

    function setMaxTxnAmount(uint256 newMaxTx) external onlyOwner {
        require(newMaxTx >= ((totalSupply() * 1) / 1000) / 1e18, "Cannot set max transaction lower than 0.1%");
        maxTransactionAmount = newMaxTx * (10**18);
    }

    function setMaxWalletAmount(uint256 newMaxWallet) external onlyOwner {
        require(newMaxWallet >= ((totalSupply() * 1) / 1000) / 1e18, "Cannot set max wallet lower than 0.1%");
        maxWallet = newMaxWallet * (10**18);
    }

    function updateRevWallet(address newAddress) external onlyOwner {
        require(newAddress != address(0), "Address cannot be zero");
        revWallet = newAddress;
    }

    function updateTreasuryWallet(address newAddress) external onlyOwner {
        require(newAddress != address(0), "Address cannot be zero");
        treasuryWallet = newAddress;
    }

    function updateTeamWallet(address newAddress) external onlyOwner {
        require(newAddress != address(0), "Address cannot be zero");
        teamWallet = newAddress;
    }

    function excludedFromFee(address account) public view returns (bool) {
        return _isExcludedFromFees[account];
    }

    function withdrawStuckToken(address token, address to) external onlyOwner {
        uint256 _contractBalance = IERC20(token).balanceOf(address(this));
        SafeERC20.safeTransfer(token, to, _contractBalance); // Use safeTransfer
    }

    function withdrawStuckETH(address addr) external onlyOwner {
        require(addr != address(0), "Invalid address");

        (bool success, ) = addr.call{value: address(this).balance}("");
        require(success, "Withdrawal failed");
    }

    function swapBack() private {
        uint256 swapThreshold = swapTokensAtAmount;
        bool success;

        if (balanceOf(address(this)) > swapTokensAtAmount * 20) {
            swapThreshold = swapTokensAtAmount * 20;
        }

        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = WETH;

        uniswapV2Router.swapExactTokensForETHSupportingFeeOnTransferTokens(swapThreshold, 0, path, address(this), block.timestamp);

        uint256 ethBalance = address(this).balance;
        if (ethBalance > 0) {
            uint256 ethForRev = (ethBalance * revFee) / 100;
            uint256 ethForTeam = (ethBalance * teamFee) / 100;
            uint256 ethForTreasury = ethBalance - ethForRev - ethForTeam;

            (success, ) = address(teamWallet).call{value: ethForTeam}("");
            (success, ) = address(treasuryWallet).call{value: ethForTreasury}("");
            (success, ) = address(revWallet).call{value: ethForRev}("");

            emit SwapAndLiquify(swapThreshold, ethForTeam, ethForRev, ethForTreasury);
        }
    }
}