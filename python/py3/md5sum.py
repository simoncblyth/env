#!/usr/bin/env python
"""
https://stackoverflow.com/questions/7829499/using-hashlib-to-compute-md5-digest-of-a-file-in-python-3

::

    epsilon:py3 blyth$ ./md5sum.py 
    07ff6ccca996639954d48032b0703f1d

    epsilon:py3 blyth$ md5 md5sum.py 
    MD5 (md5sum.py) = 07ff6ccca996639954d48032b0703f1d
    epsilon:py3 blyth$ 



"""
import hashlib
from functools import partial

def md5sum(filename):
    with open(filename, mode='rb') as f:
        d = hashlib.md5()
        for buf in iter(partial(f.read, 4096), b''):
            d.update(buf)
    return d.hexdigest()

print(md5sum('md5sum.py'))
