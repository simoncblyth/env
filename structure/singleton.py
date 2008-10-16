"""

   http://coding.derkeiler.com/Archive/Python/comp.lang.python/2005-02/3532.html

"""



class C(object):
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwds):  
        print 'In C init. only called once  '
        pass

    def __init__(self):  
        print 'In C __init__ called each time '



class D(C):
    def init(self, *args, **kwds):  
        print 'In D init.  only called once  '
        pass

    def __init__(self): 
        print 'In D __init__ called each time ' 





if '__main__'==__name__:
   print "c... "
   c = C()

   print "c... again  "
   c = C()

   print "d... "
   d = D() 

   print "d... again  "
   d = D() 

