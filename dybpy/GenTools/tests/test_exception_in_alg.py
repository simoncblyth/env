
import genrepr
import dybtest.gputil as gputil
from GaudiPython import AppMgr, PyAlgorithm 
import config

g = AppMgr()

loc = '/Event/Gen/GenHeader'

class ExceptionalAlg(PyAlgorithm):
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
g.addAlgorithm(ExceptionalAlg()) 
msv = g.service("MessageSvc")
msv.OutputLevel = 5
gen = g.algorithm("GenAlg")
gen.OutputLevel = 5 


def test_exc_in_alg():
    global g 
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


