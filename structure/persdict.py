import cPickle as pickle
import os
import re

class Persdict(dict):
    """
        Basic idea :
           * objects can be identified by their keyword arguments alone, 
             which translates into a file path 
           * using __new__ (which takes precedence over __init__) the
             object instantiation is controlled to load from a persisted version 
             if such files exists 
             
        Persdict sub classes must obey the contract 
           * implement a "populate' method where object identity 
             is fully specified by the kwa
             
        In return they gain transparent persistency, and avoidance of 
        repeating expensive object constructions, such a DB lookups. The use of 
        timestamp arguments allows simple history tracking for object changes.    
        
    """
    _dbg = 0
    def _get(cls,*args, **kwa):
        instance = cls._load(*args, **kwa)
        if instance:
             if cls._dbg > 0:
                 print "providing pickled"
             return instance
        else:
            return cls(*args, **kwa)
    _get = classmethod( _get )   
 
    def _idstring(cls, *args, **kwa):
        """
            Identity string based on ctor keyword arguments only 
        """
        import pprint
        return pprint.pformat( kwa )
    _idstring = classmethod( _idstring )   

    def _id( cls, *args, **kwa ):
        if kwa.get('hex',False) == True:
            import md5
            return md5.new(cls._idstring(*args,**kwa)).hexdigest()
        else:
            return "%s" % ( kwa.get('stamp','nostamp') )
    _id = classmethod( _id )

    def _clsdir(cls, *args, **kwa):
       persdir = kwa.get( 'persdir', '/tmp/env'  )
       clsdir = os.path.join( persdir , cls.__name__ )
       return clsdir 
    _clsdir = classmethod( _clsdir )

    def _grpdir(cls, *args, **kwa):
        clsdir = cls._clsdir( *args, **kwa )
        dir = os.path.join( clsdir , kwa.get('group','nogroup')  )
        if not(os.path.exists(dir)):
            os.makedirs(dir)
        return dir
    _grpdir = classmethod( _grpdir )   

    _patn = re.compile("(?P<stamp>\d*)")
    def _parse(cls, name ):
        m = cls._patn.match( name )
        if m:
            return m.groupdict()
        return None   
    _parse = classmethod( _parse )    
    
    def _groups(cls):
        clsdir = cls._clsdir()
        for grp in os.listdir( clsdir ):
            grpdir = os.path.join( clsdir, grp )
            if os.path.isdir(grpdir):
                yield grp
    _groups = classmethod( _groups )

    def _instances(cls, group=None ):        
        """ allows iteration over the persisted instances """
        clsdir = cls._clsdir()
        for grp in os.listdir(clsdir):
            grpdir = os.path.join( clsdir, grp )
            if os.path.isdir(grpdir) and ( group == None or grp == group ):
                for ins in os.listdir(grpdir):
                    path = os.path.join( grpdir, ins )
                    if os.path.isfile(path):
                        d = cls._parse(ins)
                        if d:
                            yield d 
    _instances = classmethod( _instances )

    def _summary(cls):
        for g in cls._groups():
            print "(base)%s._summary instances in group \"%s\" :" % (cls.__name__, g )
            for i in cls._instances(group=g):
                print i
        print "(base)%s._summary all instances :" % ( cls.__name__ )
        for i in cls._instances():
            print i
    _summary = classmethod( _summary )

    def _path(cls,*args, **kwa):
        return os.path.join(cls._grpdir(*args, **kwa), "%s.p" % cls._id(*args,**kwa) )
    _path = classmethod( _path )   
 
    def _save(cls, obj , *args, **kwa):
        pp = cls._path(*args, **kwa)
        if cls._dbg > 0:
            print "(base)%s._save obj %s to %s using identity %s " % ( cls.__name__ , obj, pp , cls._idstring(*args, **kwa))
        pickle.dump( obj , file(pp,'w') )
    _save = classmethod( _save )   
 
    def _load(cls, *args, **kwa):
        pp  = cls._path(*args, **kwa)
        ii = cls._idstring(*args, **kwa)
        if os.path.exists(pp):
            it = pickle.load(file(pp))
            if cls._dbg > 0:print "(base)%s._load loading from %s  using identity %s :  %s " %  ( cls.__name__ , pp , ii , it )
            return it 
        else:
            if cls._dbg > 0:print "(base)%s._load failed from %s using identity %s  " % ( cls.__name__ , pp , ii )
            return None
    _load = classmethod( _load )


    def __new__(cls, *args, **kwds):
        it = None
        if kwds.get('singleton',False) == True:
            if cls._dbg > 1:print "(base)%s.__new__ singleton mode ON " % cls.__name__
            it = cls.__dict__.get("__it__")
        if it is not None:
            return it
                    
        if kwds.get('remake',False) == True:
            if cls._dbg > 1:print "(base)%s.__new__ remake mode ON ... forcing remake " % cls.__name__
        else:
            if cls._dbg > 1:print "(base)%s.__new__ remake mode OFF ... try to load from persisted " % cls.__name__
            it = cls._load(*args, **kwds)  
            
        if it is not None:
            cls.__it__ = it       ## set the singleton slot as a class variable 
            if cls._dbg > 0:print "(base)%s.__new__ returning persisted instance : %s " % ( cls.__name__ , it )
            return it 
        
        if cls._dbg > 0:print "(base)%s.__new__ no persisted instance ... so invoke the __init__ (via __new__ ) " % cls.__name__
        it = dict.__new__(cls, *args, **kwds )
        
        if hasattr( cls , 'populate'):
            getattr( cls , 'populate' )( it , *args , **kwds )            
            cls._save( it , *args, **kwds  )
        else:
            print "client class is expected to have a populate method "
        if cls._dbg > 0:print "(base)%s.__new__ after dict.__new__  %s " %  ( cls.__name__ , it )
        return it

    def __init__(self,*args, **kwds): 
        if self.__class__._dbg > 0:
            print '(base)%s.__init__ THIS IS CALLED AFTER THE __new__ ? %s ' % ( self.__class__.__name__, self  )

    def diff(self, other):
        if self == other:
            return None
        return CfDict(self, other )
            

class CfDict(dict):
    def __init__(self, a , b ):
        ak = set(a.keys())
        bk = set(b.keys())
        for k in ak.intersection(bk):
            if b[k] > a[k]:
                self[k] = [ "gt", a[k], b[k] ]
            elif b[k] == a[k]:
                self[k] = [ "eq" , a[k] ]
            elif b[k] < a[k]:
                self[k] = [ "lt", a[k] , b[k] ]
        for k in ak.difference(bk):         ## ak but not bk
            self[k] = [ "lhs only"]    
        for k in bk.difference(ak):         ## bk but not ak
            self[k] = ["rhs only"]
            

class NonSingletonExample(Persdict):
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

    

class SingletonExample(Persdict):
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
   
   

