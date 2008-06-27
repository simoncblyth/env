
import GaudiPython
import genrepr 
import gprepr 
import pprint
import os
import pyutil


def reload_():
    import sys
    reload(sys.modules[__name__])

class ConfigWrapper(DybTest.PersistableRepr):
    """ 
         intermediary to present a standard interface for disparate configs to the tests 
    """
    __tesmap__={ 
          '/Event/Gen/HepMCEvents':'KeyedContainer<DayaBay::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >', 
            '/Event/Gen/GenHeader':'DayaBay::GenHeader'
                 }
     
    def __init__(self, *args, **kwargs):
        self.log( "proceeding to  __init__ " )
        self.config(*args, **kwargs)

   
    def config(self, *args, **atts):
        
        ol = 'outputlevel' in atts and atts['outputlevel'] or 5
        
        self.algs = {}        
        global g
        g = GaudiPython.AppMgr(**atts)     
        self.log( "appmgr before config %s " % repr(g) , **atts ) 
        
        import inhibit_run
        import gentools
                    
        self.gen = gen = g.algorithm("GenAlg")
        gsq = g.algorithm("GenSeq")
        
        dmp = "GtHepMCDumper/GenDump"
        if dmp in gsq.Members:
            g.removeAlgorithm(dmp)

        print "reset the positioner tool volume " 
        #volume = '/dd/Geometry/AD/lvAD'
        #volume = '/dd/Structure/steel-2/water-2'
        #volume = '/dd/Structure/Pool/la-iws'
        #volume="/dd/Geometry/Pool/lvFarPoolIWS"
        volume = '/dd/Structure/AD/la-gds1'
        
        poser = g.property("ToolSvc.GtPositionerTool")
        poser.Volume = volume
        trans = g.property("ToolSvc.GtTransformTool")
        trans.Volume = volume
        
        
        gen = g.algorithm("GenAlg")
#gen.OutputLevel = 1
        gen.GenTools = [ "GtGunGenTool", "GtTimeratorTool" ]
        ## get rid of position and tranform tools as they are not finding the detector element
        
        #pot = g.toolsvc().create("GtPositionerTool")
        #trt = g.toolsvc().create("GtTransformTool")
        #trt.Volume = pot.Volume = None
        #msv = g.service("MessageSvc")
        #msv.OutputLevel = ol
        #gun = g.property("ToolSvc.GtGunGenTool")
        #gun.OutputLevel = ol
    
        print g 
        print "instanciate and init the alg  "
        
        for arg in args:
            if issubclass(arg.__class__,GaudiPython.PyAlgorithm):
                alg = arg
                print "adding alg %s " % repr(alg)
                alg.init(self)
                g.addAlgorithm(alg)
                self.algs[alg.name()] = alg
            else:
                print "skipping arg %s " % arg
        
        self.log( "appmgr after config %s " % repr(g) )
         
        
    def cleanup(self):
        """ otherwise accumulate algs """
        for alg in self.g.TopAlg:
            self.log("cleanup removeAlgoritm %s " % alg )
            self.g.removeAlgorithm(alg)
    
                    
    def location(self):
        return self.gen.Location
    def classname(self):
        return self.__tesmap__[self.location()]


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
        

        





