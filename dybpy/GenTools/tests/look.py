"""
   The point of this is to provide objects for interactive inspection ...
   and demonstrate a PyAlgorithm 
   
   invoke with
         ipython look.py
     
        
     
"""
import genrepr
import DybTest.gputil as gputil
from GaudiPython import AppMgr, PyAlgorithm 
import config
g = AppMgr()

loc = '/Event/Gen/GenHeader'

class LookAlg(PyAlgorithm):
    def beginRun(self):
        self.print_("beginRun")
        return True
    def endRun(self):
        self.print_("endRun")
        return True
    def initialize(self):
        self.print_("initialize")
        return True
    def finalize(self):
        self.print_("finalize")
        return True
    def execute(self):
        global loc
        self.print_(loc)  
        global g
        esv = g.evtsvc()
        kco = esv[loc]
        print kco
        print repr(kco)
        return True

    def print_(self, *msg):
        print  ">>>>>>>>>>   <%s %s>" % (self.__class__.__name__, " ".join(msg) )

            
g.EvtMax = 3
g.addAlgorithm(LookAlg()) 
msv = g.service("MessageSvc")
msv.OutputLevel = 5
gen = g.algorithm("GenAlg")
gen.OutputLevel = 5 


if __name__ == '__main__':

    g.run(g.EvtMax)    
    esv = g.evtsvc()
    ghr = esv[loc]
  
    assert ghr.__class__.__name__ == 'DayaBay::GenHeader'
    print "\nghr", ghr
       
    evt = ghr.event()
    assert evt.__class__.__name__ == 'HepMC::GenEvent'
    assert evt.particles_size() == 1
        
    pdg_ids=[]
    for prt in gputil.irange(evt.particles_begin(),evt.particles_end()):
        assert prt.__class__.__name__ == 'HepMC::GenParticle'
        pdg_ids.append(prt.pdg_id())      

    
 





