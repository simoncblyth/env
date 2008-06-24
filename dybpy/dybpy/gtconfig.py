

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

    def identity(self):
        """  
           need to introspect all the properties that go into defining the run and digest it down 
           to an identity that provides the context for the events and their tests
        """
        return "klop"   


