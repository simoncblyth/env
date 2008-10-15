""" 
   Attempt to factor out pickling of persistent dicts 

"""

import cPickle as pickle
import os

class PerDict(dict):
    @classmethod
    def pget(cls,**kwa):
        instance = cls.pload(**kwa)
        if instance:
             print "providing pickled"
             return instance
        else:
            return cls(**kwa)
    
    @classmethod
    def pid(cls,**kwa):
        """
            Identity based on ctor keyword arguments
        """
        import md5
        return md5.new(repr(kwa)).hexdigest()
    
    @classmethod
    def pdir(cls,**kwa):
        dir = os.path.join('/tmp/env', cls.pid(**kwa) )
        if not(os.path.exists(dir)):
            os.makedirs(dir)
        return dir
    
    @classmethod
    def ppath(cls,**kwa):
        return os.path.join(cls.pdir(**kwa), "%s.p" % cls.__name__ )
    
    @classmethod
    def psave(cls, obj , **kwa):
        pp = cls.ppath(**kwa)
        print "saving to %s " % pp
        pickle.dump( obj , file(pp,'w') )
    
    @classmethod
    def pload(cls, **kwa):
        pp = cls.ppath(**kwa)
        if os.path.exists(pp):
            print "loading from %s " % pp
            return pickle.load(file(pp))
        else:
            print "failed to load from %s " % pp
            return None

"""
    def __new__ ( cls, *args, **kwargs ):
        newobj = object.__new__( cls, *args, **kwargs )
        cls.__init__(newobj, *args, **kwargs)
"""


