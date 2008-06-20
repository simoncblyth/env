
def syspath():
    import sys
    for s in sys.path:
        print s


class GT():
    """
       dybgaudi/InstallArea/python/gentools.py
    """

    def __init__(self):
        import xmldetdesc
        self.xddc = xmldetdesc.XmlDetDescConfig()
    
        import gentools
        self.gtc =gentools.GenToolsConfig(volume="/dd/Geometry/Pool/lvFarPoolIWS")
    
        import gaudimodule as gm
        self.app= gm.AppMgr()
        self.app.EvtSel = "NONE"
    
    
    def run(self, n=1 ):
        self.app.run(tg.nevents)




class DumpAlg(gaudimodule.PyAlgorithm):
    """ gaudi/GaudiPython/examples/read_lhcb_event_file.py """
    
    def execute(self):
        evh  = evt['Header']
        mcps = evt['MC/Particles']
        print 'event # = ',evh.evtNum()




if __name__=='__main__':
    gt = GT()
    








