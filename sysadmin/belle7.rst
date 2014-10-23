Belle7
========

Block from simon
-----------------

::

    simon:~ blyth$ ssh belle7.nuu.edu.tw
    ssh: connect to host belle7.nuu.edu.tw port 22: Operation timed out
    simon:~ blyth$ 
    simon:~ blyth$ date
    Tue Oct 21 12:09:21 HKT 2014
    simon:~ blyth$ 

::

    simon:~ blyth$ ping belle7.nuu.edu.tw
    PING belle7.nuu.edu.tw (203.64.184.126): 56 data bytes


traceroute not helping
------------------------

traceroute for blocked hop and succeeding one
both look the same. Presumably icmp packets used by 
traceroute and ping are being filtered differently to tcp
packets used by ssh. 

succeeding connection::

    [blyth@cms01 ~]$ ping belle7.nuu.edu.tw 
    PING belle7.nuu.edu.tw (203.64.184.126) 56(84) bytes of data.
    64 bytes from belle7.nuu.edu.tw (203.64.184.126): icmp_seq=20 ttl=52 time=11.9 ms
    64 bytes from belle7.nuu.edu.tw (203.64.184.126): icmp_seq=21 ttl=52 time=7.78 ms
    64 bytes from belle7.nuu.edu.tw (203.64.184.126): icmp_seq=22 ttl=52 time=5.68 ms
    64 bytes from belle7.nuu.edu.tw (203.64.184.126): icmp_seq=23 ttl=52 time=7.17 ms

    [blyth@cms01 ~]$ sudo traceroute belle7.nuu.edu.tw 
    Password:
    traceroute to belle7.nuu.edu.tw (203.64.184.126), 30 hops max, 38 byte packets
     1  140.112.101.254 (140.112.101.254)  0.631 ms  0.720 ms  0.551 ms
     2  140.112.149.117 (140.112.149.117)  0.316 ms  0.284 ms  0.316 ms
     3  140.112.0.214 (140.112.0.214)  0.616 ms  0.544 ms  0.716 ms
     4  140.112.0.186 (140.112.0.186)  0.856 ms  0.825 ms  0.659 ms
     5  140.112.0.198 (140.112.0.198)  1.179 ms  1.202 ms  1.137 ms
     6  140.112.0.70 (140.112.0.70)  1.226 ms  1.224 ms  1.156 ms
     7  bb-NCTU-TWAREN.TANet.edu.tw (192.83.196.113)  13.994 ms  5.936 ms  4.197 ms
     8  * * 140.113.0.113 (140.113.0.113)  81.585 ms
     9  not-a-legal-address (203.68.12.82)  72.515 ms  79.694 ms  75.637 ms
    10  120.105.145.1 (120.105.145.1)  73.677 ms  5.407 ms  8.089 ms
    11  * * *
    12  * * *
    13  * * *

blocked connection::

    delta:~ blyth$ ping belle7.nuu.edu.tw 
    PING belle7.nuu.edu.tw (203.64.184.126): 56 data bytes
    Request timeout for icmp_seq 0
    Request timeout for icmp_seq 1
    Request timeout for icmp_seq 2

    delta:~ blyth$ sudo traceroute belle7.nuu.edu.tw 
    Password:
    traceroute to belle7.nuu.edu.tw (203.64.184.126), 64 hops max, 52 byte packets
     1  10.0.2.1 (10.0.2.1)  0.853 ms  1.075 ms  0.419 ms
     2  140.112.101.254 (140.112.101.254)  1.944 ms  1.898 ms  2.462 ms
     3  140.112.149.117 (140.112.149.117)  1.426 ms  1.234 ms  0.957 ms
     4  140.112.0.214 (140.112.0.214)  1.891 ms  2.995 ms  1.973 ms
     5  140.112.0.186 (140.112.0.186)  1.633 ms  2.437 ms  3.854 ms
     6  140.112.0.198 (140.112.0.198)  3.564 ms  1.949 ms  1.834 ms
     7  140.112.0.70 (140.112.0.70)  2.452 ms  2.010 ms  1.895 ms
     8  bb-nctu-twaren.tanet.edu.tw (192.83.196.113)  9.529 ms  4.713 ms  7.045 ms
     9  140.113.0.113 (140.113.0.113)  6.538 ms  10.861 ms  11.176 ms
    10  not-a-legal-address (203.68.12.82)  8.997 ms  10.803 ms  8.114 ms
    11  120.105.145.1 (120.105.145.1)  22.929 ms  15.566 ms  15.245 ms
    12  * * *

    delta:~ blyth$ sudo tcptraceroute belle7.nuu.edu.tw
    Selected device en0, address 10.0.2.5, port 50553 for outgoing packets
    Tracing the path to belle7.nuu.edu.tw (203.64.184.126) on TCP port 80 (http), 30 hops max
     1  10.0.2.1  1.210 ms  0.358 ms  0.371 ms
     2  140.112.101.254  3.097 ms  1.299 ms  1.249 ms
     3  140.112.149.117  1.016 ms  0.993 ms  0.926 ms
     4  140.112.0.214  1.230 ms  1.111 ms  1.249 ms
     5  140.112.0.186  1.600 ms  1.642 ms  1.451 ms
     6  140.112.0.198  1.467 ms  1.612 ms  1.770 ms
     7  140.112.0.70  1.499 ms  1.549 ms  1.376 ms
     8  bb-nctu-twaren.tanet.edu.tw (192.83.196.113)  16.061 ms  5.115 ms  6.632 ms
     9  140.113.0.113  6.612 ms  5.342 ms  4.858 ms
    10  not-a-legal-address (203.68.12.82)  6.075 ms  9.361 ms  9.119 ms
    11  120.105.145.1  14.079 ms  14.130 ms  11.226 ms
    12  * * *




