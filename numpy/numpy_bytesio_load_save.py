#!/usr/bin/env python
"""
io.BytesIO provides a file-like interface to memory, 
allowing code expecting to write/read from files
to do so to memory  

* http://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database

"""
import numpy as np
import io

def save_to_buffer( arr ):
    stream = io.BytesIO()
    np.save(stream, a)
    stream.seek(0)
    buf = buffer(stream.read())
    return buf

def load_from_buffer( buf ):
    stream = io.BytesIO(buf)
    stream.seek(0)
    return np.load(stream)

if __name__ == '__main__':
    a = np.identity(4)
    buf = save_to_buffer(a)
    b = load_from_buffer(buf)

    print a
    print b 



