"""
   The point of this is to provide objects for interactive inspection ...
   and demonstrate a PyAlgorithm 
   
   invoke with
         ipython look.py         #### HIGHLY RECOMMENDED ####
    OR   python -i look.py
     
   Then examine the objects in your scope :  
     
     >>> locals().keys()
['genrepr', 'loc', 'PyAlgorithm', 'ghr', 'g', 'test_look', '__builtins__', 
 'msv', '__file__', 'gen', 'sys', 'evt', 'esv', 'prt', '__name__', 'AppMgr', 
 'config', '__doc__', 'LookAlg', 'gputil']
     
     
   Also contains a test to attempt to ensure that this stays operational
         nosetests test_look.py 
         nosetests test_look.py -s    
                use option -s or  --nocapture to see stdout 
   
         nosetests --help   
"""
import genrepr
import dybtest.gputil as gputil
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
        self.print_("execute")  
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

def test_look():
    g.run(g.EvtMax)    
    esv = g.evtsvc()
    ghr = esv[loc]
    assert ghr.__class__.__name__ == 'DayaBay::GenHeader'
    print "\nghr\n", ghr
    evt = ghr.event()
    assert evt.__class__.__name__ == 'HepMC::GenEvent'
    assert evt.particles_size() == 1
    for prt in gputil.irange(evt.particles_begin(),evt.particles_end()):
        assert prt.__class__.__name__ == 'HepMC::GenParticle'

if __name__ == '__main__':
    import sys
    print sys.modules[__name__].__doc__
    g.run(g.EvtMax)    
    esv = g.evtsvc()
    ghr = esv[loc]
    print "\nghr\n", ghr
    evt = ghr.event()
    for prt in gputil.irange(evt.particles_begin(),evt.particles_end()):
        assert prt.__class__.__name__ == 'HepMC::GenParticle'
             

    
 





