""" 
   Attempt to factor out pickling of persistent dicts 

   
   Found something similar ...
        http://www.mems-exchange.org/software/durus/
        http://www.mems-exchange.org/software/durus/Durus-3.7.tar.gz/Durus-3.7/persistent_dict.py
        http://www.mems-exchange.org/software/durus/Durus-3.7.tar.gz/Durus-3.7/persistent.py

   Zope also has a persistent dict 
        http://pypi.python.org/pypi/zope.app.keyreference/3.5.0b2


   When getting serious... best to use SQLAlchemy approach, but for now just want 
   something simple/lightweight for smallish dicts.

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

    def __new__ ( cls, *args, **kwa ):
        obj = super(PerDict,cls).__new__( cls, *args, **kwa )
        obj['trace'] = [] 
        obj['trace'].append('PerDict::__new__ after birth of obj %s' %  type(obj) ) 
        cls.__init__(obj, *args, **kwa )
        obj['trace'].append('PerDict::__new__ exit') 
        return obj

    def __init__(self, **kwa ):
        self['trace'].append('PerDict::__init__') 

    def init(self, **kwa):
        



class Test(PerDict):
    def __init__(self, **kwa ):
        super(Test, self).__init__(**kwa)
        self['trace'].append('Test::__init__')      
 
 
 
    def parse(self):
        self['hello'] = 'world'




if __name__=='__main__':
    print "--------"
    t1 = Test(mango='chutney')
    t1.parse() 
    print t1

