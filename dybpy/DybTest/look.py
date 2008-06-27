"""
 
"""
import GaudiPython

import genrepr
import gtconfig
import gputil

from GaudiPython import AppMgr ; g = AppMgr()


def reload_():
    import sys
    reload(sys.modules[__name__])

class LookAlg(GaudiPython.PyAlgorithm):
    
    def init(self, conf ):
        self.conf = conf
        self.esv = g.evtsvc()
        self.esv.dump()
        self.items = []
        return self
    
    def execute(self):
        loc = self.conf.location()
        print "<%s %s>" % (self.__class__.__name__ , loc )    
        #kco = self.esv[loc]
        #assert kco.__class__.__name__ == self.conf.classname()
        #if hasattr(kco,"__props__"):
        #    self.items.append(kco.__props__())
        #print kco
        #print repr(kco)
        return True
            

gttc = None 
def _configure():
    global gttc
    if gttc:
        print "_configure reusing gttc "
        return gttc
    else:
        print "_configure instantiating gttc "
        alg = LookAlg()
        gttc = gtconfig.GenToolsTestConfig(alg)
        return gttc


if __name__ == '__main__':

    conf = _configure()
    g.run(10)
    
    esv = g.evtsvc()
    kco = esv[conf.location()]
  
    ##assert kco.__class__.__name__ == 'KeyedContainer<DayaBay::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >'
    #assert kco.__class__.__name__ == 'DayaBay::GenHeader'
    #print "\nkco", kco
       
    #hme =  kco[0]
    #assert hme.__class__.__name__ == 'DayaBay::HepMCEvent'
     
    #evt = hme.event()
    #assert evt.__class__.__name__ == 'HepMC::GenEvent'
    #assert evt.particles_size() == 1
        
    #pdg_ids=[]
    #for prt in gputil.irange(evt.particles_begin(),evt.particles_end()):
    #    assert prt.__class__.__name__ == 'HepMC::GenParticle'
    #    pdg_ids.append(prt.pdg_id())      

    
 





