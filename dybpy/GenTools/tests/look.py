"""
   The point of this is to provide objects for interactive inspection ...
   invoke with
       ipython look.py
     
"""
import genrepr
import DybTest.gputil as gputil
from GaudiPython import AppMgr, PyAlgorithm 
import config
from config import cid as cid
g = AppMgr()

class LookAlg(PyAlgorithm):
    def execute(self):
        global cid
        loc = cid['location']
        self.print_(loc)  
        global g
        esv = g.evtsvc()
        kco = esv[loc]
        print kco
        print repr(kco)
        return True

    def print_(self, *msg):
        print  "<%s %s>" % (self.__class__.__name__, " ".join(msg) )

            
g.EvtMax = 5
g.addAlgorithm(LookAlg()) 

if __name__ == '__main__':

    g.run(g.EvtMax)    
    esv = g.evtsvc()
    kco = esv[cid['location']]
  
    assert kco.__class__.__name__ == 'DayaBay::GenHeader'
    print "\nkco", kco
       
    hme =  kco[0]
    assert hme.__class__.__name__ == 'DayaBay::HepMCEvent'
     
    evt = hme.event()
    assert evt.__class__.__name__ == 'HepMC::GenEvent'
    assert evt.particles_size() == 1
        
    pdg_ids=[]
    for prt in gputil.irange(evt.particles_begin(),evt.particles_end()):
        assert prt.__class__.__name__ == 'HepMC::GenParticle'
        pdg_ids.append(prt.pdg_id())      

    
 





