#!/usr/bin/env python
"""
http://stackoverflow.com/questions/5681747/how-to-write-a-sorted-function-in-python-for-version-before-2-4
"""

def _sorted(iterable, key=lambda _:_, reverse=False):
    """  
    sorted for py23, caution returns lists not tuples
    """ 
    temp = [(key(x), x) for x in iterable]
    temp.sort()
    if reverse:
        return [temp[i][1] for i in xrange(len(temp) - 1, -1, -1)] 
    return [t[1] for t in temp]

try:
    sorted
except NameError: 
    sorted = _sorted 


if __name__ == '__main__':
    demo = [1,300,7,10]
    expect = [1,7,10,300]
    assert sorted(demo) == expect , ( sorted(demo), expect )

