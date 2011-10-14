#!/usr/bin/env python
"""
Emits to stdout the hexdigest for file path provided, using chunked reading to avoid memory 
issues with large files, usage::

  ./digestpath.py /path/to/file/to/digest 

http://stackoverflow.com/questions/1131220/get-md5-hash-of-a-files-without-open-it-in-python
"""
import sys

try: 
    from hashlib import md5
except ImportError: 
    from md5 import md5

def digestpath( path ):
    hash = md5()
    size = 64*128   # 8192
    f = open(path,'rb') 
    for chunk in iter(lambda: f.read(size), ''): 
        hash.update(chunk)
    f.close()
    return hash.hexdigest()

if __name__ == '__main__':
    print digestpath(sys.argv[1])


