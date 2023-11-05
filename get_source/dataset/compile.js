const fs = require('fs');
const solc = require('solc');

// Đọc nội dung của tệp solidity
const source = fs.readFileSync('abcd.sol', 'utf8');

// Biên dịch tệp solidity
const output = solc.compile(source, 1);

// Xuất kết quả ra tệp .js
fs.writeFileSync('abcd.js', JSON.stringify(output));
