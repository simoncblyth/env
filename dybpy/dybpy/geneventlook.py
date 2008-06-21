"""
    Testing ideas ...
        * get expectations from the conf 
        * store the string repr for comparison 
        * tests in the Dumper ? 
"""

import GaudiPython
from genrepr import *

# treat globals as such ... esv doesnt work here though 
app = GaudiPython.AppMgr()

class Dumper(GaudiPython.PyAlgorithm):
    """  gaudi/GaudiPython/examples/read_lhcb_event_file.py  NB misusage of the name to feed in a location to dump """        
    
    def execute(self):
        dump = repr(self.esv[self.name()])
        self.dumps.append(dump)
        print dump
        return True
        
    def init(self):
        """ cannot use an __init__ so use this instead """
        self.esv = app.evtsvc()
        self.dumps = []
        return self

    def __repr__(self):
        dmp=[]
        dmp.append("<%s> " % self.__class__.__name__)
        for dump in self.dumps:
            dmp.append(dump)
        return "\n".join(dmp)
         

class GenEventLook():
    """ looking into the result of dybgaudi/InstallArea/python/gentools.py  """
    
    def configure(self):
        """ the xmldetdesc is required ...  would be nicer to tuck that away in GenToolConfig  """
        import xmldetdesc
        self.xddc = xmldetdesc.XmlDetDescConfig()
        import gentools
        return gentools.GenToolsConfig(volume="/dd/Geometry/Pool/lvFarPoolIWS")

    def __init__(self):
        self.conf = self.configure()
        app.EvtSel = "NONE"
        self.esv = app.evtsvc()
        self.esv.dump()
        self.gen = gen = app.algorithm("Generator")
        assert gen.Location == '/Event/Gen/HepMCEvents'
        self.dumper = dumper = Dumper(gen.Location).init()
        app.addAlgorithm(dumper)

    def run(self):
        app.run(self.conf.nevents)
        self.discovery()
        return self

    def __repr__(self):
        """ hmmm how to control the run such that this to gives string rep of the whole shebang """
        dmp = []
        dmp.append( "<%s> " % self.__class__.__name__ )
        dmp.append( repr(self.dumper) )
        return "\n".join(dmp)

    def discovery(self):
        """ all the assertions are done by genrepr when printing kco  ... still here mainly to provide objs for interactive examination """
        kco = self.kco = self.esv[self.gen.Location]
        assert kco.size() == 1 == len(kco)
        assert kco.__class__.__name__ == 'KeyedContainer<DayaBay::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >'
        print "\nkco", kco
       
        hme = self.hme = kco[0]
        assert hme.__class__.__name__ == 'DayaBay::HepMCEvent'
        assert hme.generatorName() == self.gen.GenName
        #print "\nhme", hme
 
        evt = self.evt = hme.event()
        assert evt.__class__.__name__ == 'HepMC::GenEvent'
        assert evt.particles_size() == 1
        #print "\nevt", evt
 
        pdg_ids=[]
        for prt in irange(evt.particles_begin(),evt.particles_end()):
            self.prt = prt 
            assert prt.__class__.__name__ == 'HepMC::GenParticle'
            #print "\nprt",prt
            pdg_ids.append(prt.pdg_id())      
               

if __name__=='__main__':
    self = GenEventLook()
    self.run()
    print "self", self
    
    








