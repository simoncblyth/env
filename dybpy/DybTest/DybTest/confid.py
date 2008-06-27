
import os
import pprint

class PersistableRepr(object):

    def identity(self):
        """ string of length 32, containing only hexadecimal digits ... 
            that can be used to represent the identity of this object so long as 
            all pertinent properties are included in the repr 
        """
        import md5
        return md5.new(repr(self)).hexdigest()

    def path(self, n ):
        """   relative path based on the identity  """
        name = n == -1 and "conf" or "%0.3d" % n 
        return "%s/%s.py" % ( self.identity() , name ) 
        
    def save(self, n , obj):
        """ 
           persist the repr of the object 
           for n of zero the repr of the conf is also saved 
        """
        assert type(n) == int and n > -2
        if n==0:
            self.save( -1, self)
        
        p = self.path(n)
        #print "saving to %s " % p
        pp = os.path.dirname(p)
        if not(os.path.exists(pp)):
            os.makedirs(pp)
        file(p,"w").write(pprint.pformat(obj)+"\n")

    def load(self, n ):
        """ revivify the repr with eval ... assumes it is a valid expression """  
        p = self.path(n)
        #print "loading from %s " % p 
        if os.path.exists(p):
            r = file(p).read()
            o = eval(r)
            return o
        return None

    def hdr(self):
        return "<%s [0x%08X] > " % ( self.__class__.__name__ , id(self) )
    
    def log(self, *args , **kwargs ):
        import pprint
        print "%s %s %s" % ( self.hdr() , " ".join(args),  pprint.pformat(kwargs) )
        
        
        
class ConfigIdentity(PersistableRepr):

    def __init__(self, **atts ):
        self.atts = atts
        
    def __props__(self):
        d = {}
        for k,v in self.atts.items():
            d[k]=v.__props__() 
        return d
    
    def update(self, **xtts ):
        self.atts.update( **xtts )
    
    def __getitem__( self , k ):
        return self.atts[k]
    
    def __repr__(self):
        import pprint 
        return pprint.pformat( self.__props__() )
