import ROOT
from ROOT import kTRUE, kFALSE, TFile
from evdatamodel import EvDataModel

class EvTree(list):
    def __init__( self , path ):
        self.status = ""
        self.path = path
        self.tree = None
        self.edm = EvDataModel()
        self.load_tree()
        self.load_entry(0)

    def refresh(self):
        if not(self.tree):
            ROOT.Error("EvTree.refresh" , "no tree to refresh" )
            return False
        print "EvTree.refresh tree "
        self.tree.Refresh()

    def load_tree(self, cache=False):
        if cache:
            import os
            TFile.SetCacheFileDir(os.getcwd())
            f = TFile.Open(self.path,"CACHEREAD")
        else:
            f = TFile.Open(self.path)
        if not(f):
            self.status = "failed to open path "
            return False
        tree = f.Get(self.edm.treename)
        print "EvTree.load_tree entries : %s " % tree.GetEntries()
        self.tree = tree
        self.edm.branch_addresses( self.tree )

    def __call__(self, i):
        return self.load_entry(i)

    def load_entry(self , i ):
        if not(self.tree):return None
        self.tree.GetEntry(i)
        return True

    def pmt_response(self,**kwa):return self.edm.pmt_response(**kwa)
    def tracker_hits(self,**kwa):return self.edm.tracker_hits(**kwa)
    def evt_summary(self,**kwa):return self.edm.evt_summary(**kwa)
    def run_summary(self,**kwa):return self.edm.run_summary(**kwa)
    def ndr_summary(self,**kwa):return self.edm.ndr_summary(**kwa)
    ## nasty : this is duplicated with evonline , rethink inheritance from list to avoid ?

    def __getitem__(self, i ):
        if self.load_entry(i):
            return self.edm() 

    def __len__(self):
        if not(self.tree):
            print "tree is not loaded "
            return 0 
        return self.tree.GetEntries()

    def __repr__(self):
        if self.tree:vers=self.edm.version(self.tree) 
        return "<Abt %s %s %s >" % (self.path , self.status , vers  )  



if __name__=='__main__':

    #src =  "http://dayabay.phys.ntu.edu.tw/aberdeen/run00027.root"
    src =  "$ABERDEEN_HOME/DataModel/sample/run00027.root"

    et = EvTree(src)
    print et 

    n = len(et)
    for i in range(10):
        print et[i]




