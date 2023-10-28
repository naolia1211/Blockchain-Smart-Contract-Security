# Blockchain-Smart-Contract-Security

## 1. Hướng dẫn sử sử dụng [slither](https://github.com/crytic/slither)
* Bước 1: tải về bằng `pip3 install slither-analyzer`
* Bước 2: tải [solc-select](https://github.com/crytic/solc-select)
* Bước 3: khi tải slither về nhớ tải tất cả phiên bản về bằng lệnh `solc-select install all` 
* Bước 4: kiểm tra phiên bản **pragma solidity** của của file **.sol** đó
* Bước 5: `solc-select use [version]` chọn version phù hợp với file đó
* Bước 6: chạy `slither [file]`
