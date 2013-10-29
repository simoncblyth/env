#!/usr/bin/env python
"""

::

    Traceback (most recent call last):
      File "test_IndexedList.py", line 14, in <module>
        C = deepcopy(L)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/copy.py", line 189, in deepcopy
        y = _reconstruct(x, rv, 1, memo)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/copy.py", line 329, in _reconstruct
        y.append(item)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/pycollada-0.4-py2.6.egg/collada/util.py", line 226, in append
        self._addindex(obj)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/pycollada-0.4-py2.6.egg/collada/util.py", line 152, in _addindex
        _idx = self._index
    AttributeError: 'IndexedList' object has no attribute '_index'

"""

from copy import deepcopy 
from collada.util import IndexedList

class obj(object):
    def __init__(self, id):
        self.id = id
    def __str__(self):
        return "obj %s " % self.id
    __repr__ = __str__

a = obj("a")
b = obj("b")
c = obj("c")

L = IndexedList([], ('id',))

#L.append(a)
#L.append(b)
#L.append(c)

if len(L) == 0:
    L.append(a)
else:   
    L.insert(0,a) 
L.insert(0,b) 
L.insert(0,c) 

print L
print "a in L", a in L
print "a.id in L", a.id in L


#C = deepcopy(L)




