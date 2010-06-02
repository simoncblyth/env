import ROOT

# for debugging the plumbing without the GUI 

from ROOT import kTRUE, kFALSE

from evctrl import EvController
print "imported EvController"
from evtree  import EvTree
print "imported EvTree"
from evonline import EvOnline
print "imported EvOnline"


class Controller(EvController):
    def __init__(self):
        EvController.__init__(self)
        self.src = None

    def istree(self):
        return self.GetSource().endswith(".root")
 
    def propagateTreeChange(self):
        if len(self.src)==0:return False
        if len(self.src)>0:
            self.SetEntryMinMax( 0, len(self.src) - 1 )

    def handleRefreshSource(self):
        if not(self.src):return False
        self.src.refresh() 
        self.propagateTreeChange()
        self.SetEntry(-1)

    def handleChangedSource(self):
        if self.istree():
            self.src = EvTree(self.GetSource())
            self.propagateTreeChange()
        else:
            self.src = EvOnline(self.GetSource()) 
        self.SetEntry(0)

    def handleChangedEntry(self):
        entry = self.GetEntry()
        self.src(entry)
        print "handleChangedEntry %s\n%s" % ( entry, self.src.edm()  )

    def __call__(self, *args ):
        if len(args)>0:
            self.SetEntry(args[0])
        else:     
            self.NextEntry()

    def __repr__(self):
        return "%s entry %s " % ( self.__class__.__name__ , self.GetEntry() )



if __name__=='__main__':

    g = Controller()
    from ROOT import g_
    print "created %s " % g 
    g.SetSource("online")

    # stay alive while debugging with : gdb which python
    while kTRUE:
        ROOT.gSystem.Sleep(100)
        ROOT.gSystem.ProcessEvents()



    

