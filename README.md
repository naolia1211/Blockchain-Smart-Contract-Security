# Blockchain-Smart-Contract-Security

## 1. Hướng dẫn sử sử dụng [slither](https://github.com/crytic/slither)
* Bước 1: tải về bằng `pip3 install slither-analyzer`
* Bước 2: tải [solc-select](https://github.com/crytic/solc-select)
* Bước 3: khi tải slither về nhớ tải tất cả phiên bản về bằng lệnh `solc-select install all` 
* Bước 4: kiểm tra phiên bản **pragma solidity** của của file **.sol** đó
* Bước 5: `solc-select use [version]` chọn version phù hợp với file đó
* Bước 6: chạy `slither [file]`
## 2. Hướng dẫn sử dụng [securify2](https://github.com/eth-sri/securify2)
khi chạy lệnh `sudo add-apt-repository ppa:ethereum/ethereum` nó sẽ hiện ra lệnh lỗi 
> Traceback (most recent call last):<br>
> &nbsp File "/usr/bin/add-apt-repository", line 3, in <module> <br>
> <hr>import apt_pkg<br>
> ModuleNotFoundError: No module named 'apt_pkg'
và sau khi tìm hiểu thì nó sử dụng python3.6 để chạy cho nên là mọi người chạy thì nhớ cài **python3.6** nha
### cách để cài python3.7 cũng như 3.6
* Bước 1: đầu tiên mọi người lên https://www.python.org/downloads/ để tải phiên bản đó về (file tar) sau đó giải nén và ghi các lệnh dưới đây
  `./configure
    make
    make test
    sudo make install`
* Bước 2: sau khi chạy xong thì ta sẽ cấp độ ưu tiên cho bó bằng lệnh
`sudo update-alternatives --install /usr/bin/python3 python3 /home/kali/Downloads/Python-3.7.16/python 1`
* Bước 3 ở đường dãn đầu là python3 có trong hệ thông và đường đãn thứ 2 là nơi mà mọi người đã giải nén và chạy
* Bước 4:sau khi chạy xong mọi người kiểm tra phiên bản python hiện có là python3 --version
