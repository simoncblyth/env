"""
 
"""

import unittest
import GaudiPython
import genrepr
import gputil 

class LookAlg(GaudiPython.PyAlgorithm):
    
    def init(self, conf ):
        self.conf = conf
        self.esv = g.evtsvc()
        self.esv.dump()
        self.items = {}
        return self
    
    def execute(self):
        loc = self.conf.location()
        print "<%s %s>" % (self.__class__.__name__ , loc )
    
        kco = self.esv[loc]
        assert kco.__class__.__name__ == self.conf.classname()
        if hasattr(kco,"__props__"):
            self.items[len(self)]=kco.__props__()
        print kco
        print repr(kco)
        return True
            
    def __repr__(self):
        return "<%s>" % self.__class__.__name__
    def __getitem__(self,key):
        return key in self.items and self.items[key] or None
    def __len__(self):
        return len(self.items)


def _configure():
    import gtconfig
    conf = gtconfig.GenToolsTestConfig( volume="/dd/Geometry/Pool/lvFarPoolIWS" )
    global g
    g = conf.g
    alg = LookAlg().init(conf)
    g.addAlgorithm(alg)
    return conf



class GenEventLook():
    """ 
         looking into the result of dybgaudi/InstallArea/python/gentools.py  
         this is now primarily to provide some objs for interactive inspection
    """
    def __init__(self,conf):
        self.conf = conf
 
    def run(self):
        g.run(self.conf.nevents())
        self.discovery()
        return self

    def discovery(self):
        """ all the assertions are done by genrepr when printing kco  
          ... still here mainly to provide objs for interactive examination 
        """
        
        self.esv = g.evtsvc()
        kco = self.kco = self.esv[self.conf.location()]
        #assert kco.size() == 1 == len(kco)
        assert kco.__class__.__name__ == 'KeyedContainer<DayaBay::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >'
        print "\nkco", kco
       
        hme = self.hme = kco[0]
        assert hme.__class__.__name__ == 'DayaBay::HepMCEvent'
        #print "\nhme", hme
 
        evt = self.evt = hme.event()
        assert evt.__class__.__name__ == 'HepMC::GenEvent'
        assert evt.particles_size() == 1
        #print "\nevt", evt
 
        pdg_ids=[]
        for prt in gputil.irange(evt.particles_begin(),evt.particles_end()):
            self.prt = prt 
            assert prt.__class__.__name__ == 'HepMC::GenParticle'
            #print "\nprt",prt
            pdg_ids.append(prt.pdg_id())      
       

if __name__ == '__main__':
    conf = _configure()
    self = GenEventLook(conf).run()
    
    esv = self.esv
    kco = self.kco
    hme = self.hme
    evt = self.evt
    prt = self.prt






