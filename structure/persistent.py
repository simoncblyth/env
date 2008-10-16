"""
   Attempt to adapt the singleton pattern to persistency 
   handling 

"""


import cPickle as pickle
import os


class Persistent(object):
    """
    
     Subclasses of Persistent should implement a general form of __init__ method :
          __init__(*args, **kwargs)
    
     When using the singleton form, the __init__ method should be :
     
          def __init__(self, *args, **kwa ):pass
       POSSIBILITES : enforce this somehow ?
     
     
     and the state of the object should be set up in an init method, eg: 
     
         def init(self, *args, **kwa ):self.parse() ## or whatever 
     
     If you do not do this you miss out on the singleton benefits 
     of only doing the state setup once only.
    
     The identity of persisted files is based on the kwargs only 
     Two of the kwargs have special meaning :
    
         singleton=True  : 
                  operate in singleton manner, ie try to use a preexisting 
                  instance in memory already 
          remake=True     : 
                  force remake, ie do not attempt to reuse a persisted instance 
          
       In singleton mode you should set the state of the object in the init method and
       leave the __init__ as pass, otherwise set the state in the __init__ method and leave 
       the init as pass


    

    """
    
    @classmethod
    def _get(cls,*args, **kwa):
        instance = cls._load(*args, **kwa)
        if instance:
             print "providing pickled"
             return instance
        else:
            return cls(*args, **kwa)
    
    @classmethod
    def _idstring(cls, *args, **kwa):
        import pprint
        return pprint.pformat( kwa )
    
    @classmethod
    def _id(cls,*args, **kwa):
        """
            Identity based on ctor keyword arguments
        """
        import md5
        return md5.new(cls._idstring(*args,**kwa)).hexdigest()
    
    @classmethod
    def _dir(cls,*args, **kwa):
        dir = os.path.join('/tmp/env', cls._id(*args, **kwa) )
        if not(os.path.exists(dir)):
            os.makedirs(dir)
        return dir
    
    @classmethod
    def _path(cls,*args, **kwa):
        return os.path.join(cls._dir(*args, **kwa), "%s.p" % cls.__name__ )
    
    @classmethod
    def _save(cls, obj , *args, **kwa):
        pp = cls._path(*args, **kwa)
        print "saving to %s using identity %s " % ( pp , cls._idstring(*args, **kwa))
        pickle.dump( obj , file(pp,'w') )
    
    @classmethod
    def _load(cls, *args, **kwa):
        pp = cls._path(*args, **kwa)
        if os.path.exists(pp):
            print "loading from %s  using identity %s " % ( pp , cls._idstring(*args, **kwa))
            return pickle.load(file(pp))
        else:
            print "failed to load from %s " % pp
            return None


    def __new__(cls, *args, **kwds):
        
        # check if have the singleton instance yet
        if kwds.get('singleton',False) == True:
            print "singleton mode ON "
            it = cls.__dict__.get("__it__")
        else:
            print "singleton mode OFF "
            it = None
        
        if it is not None:
            return it            
        
        # attempt to access the persisted it, if find it set the singleton slot and return
        if kwds.get('remake',False) == True:
            print "remake mode ON ... forcing a remake "
            it = None
        else:
            print "remake mode OFF ... try to load from persisted "
            it = cls._load(*args, **kwds)  
        
        if it is not None:
            cls.__it__ = it
            return it 
        
        # still dont find it ... so make it and save it 
        it = object.__new__(cls)
        
        if hasattr(it, 'init'):
            it.init(*args, **kwds)
        else:
            print "skipping init method ... as you didnt implement one "
            
        cls._save(it, *args, **kwds)
    
        cls.__it__ = it
        return it


    def __init__(self,*args, **kwds): # called each time
        print 'In Persistent __init__.'




class NonSingletonExample(Persistent):
    """
          Using remake=True will inhibit loading of a persisted version 
          
          if an init method is implemented it will be invoked once only for the type
          hence... when not wanting singleton operation, it is clearer not to implement 
          said method
    """
    def __init__(self, *args, **kwds):
        self.args = args
        self.kwds = kwds  
    def __repr__(self):
        import pprint
        return "<%s %s %s>" % ( self.__class__.__name__ , pprint.pformat(self.args), pprint.pformat(self.kwds) )

    

class SingletonExample(Persistent):
    """
         You must provide the argument  singleton=True
           TODO: 
                how to avoid this without using another class
    """
    def __init__(self, *args, **kwds ):pass
    def init(self, *args, **kwds): 
        print "SingletonExample.init leave as pass for non-singleton usage "
        self.args = args
        self.kwds = kwds        
        pass
    def __repr__(self):
        import pprint
        return "<%s %s %s>" % ( self.__class__.__name__ , pprint.pformat(self.args), pprint.pformat(self.kwds) )



if '__main__'==__name__:
 
   print "nse... "
   nse = NonSingletonExample(hello="one", remake=True) 
   print nse


   print "se...   "
   se1 = SingletonExample(hello="two",singleton=True) 
   print se1
   
   se2 = SingletonExample(hello="two",singleton=True) 
   print se2
   
   assert se1 == se2 , "NOT a singletonian %s != %s " % ( se1 , se2 )  
   
   

