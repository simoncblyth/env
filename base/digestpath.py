#!/usr/bin/env python
"""
Emits to stdout the hexdigest for file path provided, using chunked reading to avoid memory 
issues with large files, usage::

  ./digestpath.py /path/to/file/to/digest 

http://stackoverflow.com/questions/1131220/get-md5-hash-of-a-files-without-open-it-in-python
"""
import sys, hashlib
def digestpath( path ):
    md5 = hashlib.md5()
    size = 64*128   # 8192
    with open(path,'rb') as f: 
        for chunk in iter(lambda: f.read(size), ''): 
            md5.update(chunk)
    return md5.hexdigest()

if __name__ == '__main__':
    print digestpath(sys.argv[1])


