
import GaudiPython
import genrepr 
import gprepr 
import pprint
import os
import pyutil
import consistency 


class GenToolsTestConfig(object,pyutil.PrintLogger):
    """ 
         intermediary to present a standard interface for disparate configs to the tests 
    """
    __tesmap__={ 
          '/Event/Gen/HepMCEvents':
               'KeyedContainer<DayaBay::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >' 
               }
    
    __singleton = None
    
    @classmethod
    def configure(cls):
        print "_configure "
        print "instantiating gttc "
        gttc = GenToolsTestConfig(volume="/dd/Geometry/Pool/lvFarPoolIWS")
        return gttc
    
    def __new__(cls, *args, **kwargs):
        """
             ensure the instantiation of the conf only gets done once ...
             forced by incomplete GaudiPython cleanup
        """
        if cls.__singleton==None:
            print "instanciating singleton "
            obj = object.__new__( cls, *args, **kwargs )
            print "calling __init__ "
            cls.__init__(obj, *args, **kwargs)
            print "__init_ done "
            cls.__singleton = obj
        print "returning singleton "
        return cls.__singleton

    def __init__(self, **kwargs):
        if self.__class__.__singleton:
            self.log("skipping __init__ ")
            return
        self.log( "proceeding to  __init__ " )
        self.config(**kwargs)

    def config(self, **atts):
    
        global g
        
        ol = 'outputlevel' in atts and atts['outputlevel'] or 5
        
        g = GaudiPython.AppMgr(outputlevel=ol)        
        g.EvtSel = "NONE"
        
        self.log( "appmgr before config %s " % repr(g) , ol=ol ) 
        
        import xmldetdesc
        xddc = xmldetdesc.XmlDetDescConfig()
        import gentools
        self.conf = gentools.GenToolsConfig(**atts)
        
        self.gen = gen = g.algorithm("Generator")
        
        ## modify the config after the fact
        
        g.removeAlgorithm('GtHepMCDumper/Dumper')
        msv = g.service("MessageSvc")
        msv.OutputLevel = ol
        gun = g.property("ToolSvc.GtGunGenTool")
        gun.OutputLevel = ol
    
        print g 
        
        print "instanciate and init the alg  "
        self.alg = consistency.ConsistencyAlg().init(self)
        print "addAlgorithm %s " % repr(self.alg)
        g.addAlgorithm(self.alg)
       
        self.log( "appmgr after config %s " % repr(g) )
         
                   
        
    def cleanup(self):
        """ otherwise accumulate algs """
        for alg in self.g.TopAlg:
            self.log("cleanup removeAlgoritm %s " % alg )
            self.g.removeAlgorithm(alg)
    
                    
    def nevents(self):
        return self.conf.nevents
    def location(self):
        return self.gen.Location
    def classname(self):
        return self.__tesmap__[self.location()]

    def __digest__(self):
        """ string of length 32, containing only hexadecimal digits ... 
            that can be used to represent the identity of this object so long as 
            all pertinent properties are included in the repr 
        """
        import md5
        return md5.new(repr(self)).hexdigest()

    def __props__(self):
        """ most of the properties are auditing/operational things that will not
            change the events generated ... so cannot automate the choice of properties 
            that constitute the identity of the "run" 
            
            TODO: ? need to control the formatting of numbers to properly ignore 
              architecture differences / rounding errors ...
            
        """
 
        d = {}
        for gtp in ["Location","GenName"]:
            d[gtp] = getattr(self.gen, gtp )
        
        for gtn in self.gen.GenTools:
            tool = g.property("ToolSvc.%s" % gtn)
            d[gtn] = {}
            for k,v in tool.properties().items():
                if k not in ['OutputLevel']:
                    d[gtn][k] = v.value()
        return d
        
    def __repr__(self):
        return pprint.pformat( self.__props__() )
        

    def identity(self):
        """  
           need to introspect all the properties that go into defining the run and digest it down 
           to an identity that provides the context for the events and their tests
        """
        return self.__digest__()  


    def path(self, n ):
        """   relative based on the identity of the configuration """
        #return "%s/%s/%s.py" % ( self.identity() , self.location().lstrip("/") , n ) 
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
        



gttc = GenToolsTestConfig.configure()




