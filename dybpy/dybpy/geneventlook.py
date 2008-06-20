"""
    http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/EventIntrospection
   
    Useful slides from Thomas Ruf :
    
    http://lhcb-reconstruction.web.cern.ch/lhcb-reconstruction/Python/GaudiPython_and_RawData.pdf 
    http://lhcb-reconstruction.web.cern.ch/lhcb-reconstruction/Python/Dst_as_Ntuple_files/frame.htm
   

"""


import GaudiPython
from GaudiPython import PyAlgorithm

import ROOT
ROOT.gSystem.Load("libMathCore")   ## workaround for http://dayabay.phys.ntu.edu.tw/tracs/env/ticket/42


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


class irange(object):
    """  cranking the iterator from py side  """
    def __init__(self, begin, end):
        self.begin, self.end = begin, end
    def __iter__(self):
        it = self.begin
        while it != self.end:
            yield it.__deref__()
            it.__postinc__(1)


class GenEventLook():
    """
       dybgaudi/InstallArea/python/gentools.py
    """
    def __init__(self):
        self.conf = self.configure()
        self.app = self.conf.app
        self.app.EvtSel = "NONE"
        self.esv  = self.app.evtsvc()

    def configure(self):
        """
            the xmldetdesc is required ...  would be nicer to tuck that away elsewhere than here
        """
        import xmldetdesc
        self.xddc = xmldetdesc.XmlDetDescConfig()
        import gentools
        return gentools.GenToolsConfig(volume="/dd/Geometry/Pool/lvFarPoolIWS")


    def run(self, n=1 ):
        self.app.run(self.conf.nevents)
        self.dump()
        self.introspect()

    def dump(self):
        """
        dump is inspecting the TES
        
        >>> gel = GenEventLook()
        >>> gel.evt.dump()
        /Event
        /Event/Gen
        /Event/Gen/HepMCEvents
        """
        self.esv.dump()
        
    def introspect(self):
        """
             tis tedious 
                  but the best possible documentation that will be gauranteed to 
                  be correct when put into automated testing
             
             need to get expectations from the conf  
             
             make less tedious and error prone by implementing the checking
             within the classes in question ... by dynamically adding methods to 
             the py counterparts
             
                
        """
    
        loc = '/Event/Gen/HepMCEvents'
        gen = self.app.algorithm("Generator")
        
        assert gen.Location == loc
    
        kco = self.kco = self.esv[loc]
        assert kco.size() == 1 == len(kco)
        
        name = 'DayaBay::HepMCEvent' 
        for i in kco:
            assert i.__class__.__name__ == name
        
        hme = self.hme = kco[0]
        assert hme.__class__.__name__ == name 
 
        assert hme.generatorName() == gen.GenName
 
 
        evt = self.evt = hme.event()
        assert evt.__class__.__name__ == 'HepMC::GenEvent'
               
        assert evt.particles_size() == 1 
            
        pdg_ids=[]
        for p in irange(evt.particles_begin(),evt.particles_end()):
            pdg_ids.append(p.pdg_id())      
               
        
        
        
                             
        print "  generatorName %s " % ( hme.generatorName() )
 
 
 
 
     
         
                 
class DumpAlg(PyAlgorithm):
    """ gaudi/GaudiPython/examples/read_lhcb_event_file.py 
    """
    
    def execute(self):
        self.hepmcevent = evt['/Event/Gen/HepMCEvents']





if __name__=='__main__':
    gel = GenEventLook()
    print gel
    








