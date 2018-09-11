#!/usr/bin/env python
"""

1. Generate two sets of 16M random numbers, with a common seed.
2. Verify that a and b are the same
3. Persist to files

::

    In [15]: np.save( "a.npy", a )

    In [16]: import commands

    In [18]: print commands.getoutput( "xxd -l 256 -c 32 a.npy")
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363 7227 3a20 273c 6638 272c 2027 666f 7274  .NUMPY..v.{'descr': '<f8', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73 652c 2027 7368 6170 6527 3a20 2831 3030  ran_order': False, 'shape': (100
    00000040: 3030 3030 2c20 342c 2034 292c 207d 2020 2020 2020 2020 2020 2020 2020 2020 2020  0000, 4, 4), }                  
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020 2020 2020 2020 2020 2020 2020 2020 200a                                 .
    00000080: b049 ca8e 5389 da3f 7f82 a8d1 c6c5 e53f ec76 e111 ce7e e43f 1ad5 9fe1 b450 d23f  .I..S..?.......?.v...~.?.....P.?
    000000a0: 2387 95f3 1844 e43f eea9 36b4 b3f5 e13f c634 3b97 e621 ee3f c0f2 6933 0538 e03f  #....D.?..6....?.4;..!.?..i3.8.?
    000000c0: 9005 24c9 4c36 ae3f 3a36 abde 4629 d33f bc43 6b94 f85f c33f 58ba 416e 2fd6 ce3f  ..$.L6.?:6..F).?.Ck.._.?X.An/..?
    000000e0: 1c27 e452 83ac d63f ee11 880a 7da0 ea3f 7019 deba 5555 c23f 0025 2c72 6cbe e13f  .'.R...?....}..?p...UU.?.%,rl..?


    In [26]: a.size
    Out[26]: 16000000

    In [21]: a.nbytes
    Out[21]: 128000000

    In [22]: a.nbytes/1e6
    Out[22]: 128.0

    In [24]: commands.getoutput("ls -al a.npy")
    Out[24]: '-rw-r--r--  1 blyth  staff  128000128 Sep 10 15:09 a.npy'



"""
import numpy as np, commands

seed = 0 

np.random.seed(seed)
a = np.random.rand( 1000000,4,4 )

np.random.seed(seed)
b = np.random.rand( 1000000,4,4 )

assert np.all( a == b )

np.save( "a.npy", a )
np.save( "b.npy", b )

a2 = np.load("a.npy")

assert np.all( a2 == a ) 





