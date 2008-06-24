

import GaudiPython

class GenToolsTestConfig(object):
    """ 
         intermediary to present a standard interface for disparate configs to the tests 
    """
    __tesmap__={ 
          '/Event/Gen/HepMCEvents':
               'KeyedContainer<DayaBay::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >' 
               }
    
    __singleton = None
    
    def __new__(cls, *args, **kwargs):
        """
            ensure the instantiation of the conf only gets done once ...
            forced by incomplete GaudiPython cleanup
        """
        if not cls.__singleton:
            obj = object.__new__( cls, *args, **kwargs )
            cls.__init__(obj, *args, **kwargs)
            cls.__singleton = obj
        return cls.__singleton

    
    def __init__(self, **atts ):
        """
                http://dayabay.phys.ntu.edu.tw/tracs/env/ticket/43             
        """
        
        global g
        self.g = g = GaudiPython.AppMgr(outputlevel=5)        
        #g.initialize()      http://dayabay.phys.ntu.edu.tw/tracs/env/ticket/44
        g.EvtSel = "NONE"
        
        import xmldetdesc
        xddc = xmldetdesc.XmlDetDescConfig()
        import gentools
        self.conf = gentools.GenToolsConfig(**atts)
        
        g.initialize()
        import genrepr 
        self.gen = gen = g.algorithm("Generator")
        
        
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
                d[gtn][k] = v.value()
        return d
        
    def __repr__(self):
        import pprint
        return pprint.pformat( self.__props__() )
        

    def identity(self):
        """  
           need to introspect all the properties that go into defining the run and digest it down 
           to an identity that provides the context for the events and their tests
        """
        return self.__digest__()  


