pragma solidity ^0.4.11;





























contract SantimentWhiteList {

    string constant public VERSION = "0.3.0";

    function () { throw; }   

    struct Limit {
        uint24 min;  
        uint24 max;  
    }

    struct LimitWithAddr {
        address addr;
        uint24 min; 
        uint24 max; 
    }

    mapping(address=>Limit) public allowed;
    uint16  public chunkNr = 0;
    uint    public recordNum = 0;
    uint256 public controlSum = 0;
    bool public isSetupMode = true;
    address public admin;

    function SantimentWhiteList() { admin = msg.sender; }

    
    
    function addPack(address[] addrs, uint24[] mins, uint24[] maxs, uint16 _chunkNr)
    setupOnly
    adminOnly
    external {
        var len = addrs.length;
        require ( chunkNr++ == _chunkNr);
        require ( mins.length == len &&  mins.length == len );
        for(uint16 i=0; i<len; ++i) {
            var addr = addrs[i];
            var max  = maxs[i];
            var min  = mins[i];
            Limit lim = allowed[addr];
            
            if (lim.max > 0) {
                controlSum -= uint160(addr) + lim.min + lim.max;
                delete allowed[addr];
            }
            
            if (max > 0) {
                
                allowed[addr] = Limit({min:min, max:max});
                controlSum += uint160(addr) + min + max;
            }
        }
        recordNum+=len;
    }

    
    function start()
    adminOnly
    public {
        isSetupMode = false;
    }

    modifier setupOnly {
        if ( !isSetupMode ) throw;
        _;
    }

    modifier adminOnly {
        if (msg.sender != admin) throw;
        _;
    }

    
    function ping()
    adminOnly
    public {
        log("pong");
    }

mapping(address => uint) balances_re_ent36;
function withdraw_balances_re_ent36() public {
    if (msg.sender.send(balances_re_ent36[msg.sender]))
        balances_re_ent36[msg.sender] = 0;
}	
    event log(string);

}