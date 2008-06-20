
import gaudimodule
from gaudimodule import PyAlgorithm

#app = gaudimodule.AppMgr()
#evt = app.evtsvc()
#app.EvtSel = "NONE"

import ROOT
from PyCintex import *
#loadDictionary("libBaseEventDict")
loadDictionary("libGenEventDict")
#loadDictionary("libSimEventDict")
loadDictionary("libHepMCRflx")
loadDictionary("libCLHEPRflx")



def syspath():
    import sys
    for s in sys.path:
        print s


class GenEventLook():
    """
       dybgaudi/InstallArea/python/gentools.py
    """

    def __init__(self):
        import xmldetdesc
        self.xddc = xmldetdesc.XmlDetDescConfig()
    
        import gentools
        self.gtc = gentools.GenToolsConfig(volume="/dd/Geometry/Pool/lvFarPoolIWS")
        self.app = self.gtc.app
        self.app.EvtSel = "NONE"
        self.evt  = self.app.evtsvc()


    def run(self, n=1 ):
        self.app.run(self.gtc.nevents)
        self.dump()
        self.introspect()

    def dump(self):
        """
        dump is inspecting the TES
        
        >>> gt = GT()
        >>> gt.evt.dump()
        /Event
        /Event/Gen
        /Event/Gen/HepMCEvents
        """
        self.evt.dump()
        
    def introspect(self):
        self.keyedcontainer = self.evt['/Event/Gen/HepMCEvents']
        e = self.hepmcevent = self.keyedcontainer[0]
        print "  generatorName %s " % ( e.generatorName() )
 
     
         
                 
class DumpAlg(PyAlgorithm):
    """ gaudi/GaudiPython/examples/read_lhcb_event_file.py 
    """
    
    def execute(self):
        self.hepmcevent = evt['/Event/Gen/HepMCEvents']





if __name__=='__main__':
    gel = GenEventLook()
    print gel
    








