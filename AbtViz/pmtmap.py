
import os
import cPickle as pickle

class PMT:
    def __repr__(self):return "<PMT %-2s ( %10.3f %10.3f %10.3f ) >" % ( self.id , self.x , self.y , self.z )

class PMTMap(list):
    pmtmap = "$ENV_HOME/aberdeen/root/abergeo.root"
    pmtree = "PMTPositionTree"
    def __init__(self):
        if os.path.exists(self.path()):
            self.pyload()
        else:
            self.rootload()
            self.pysave()
    
    def rootload(self):
        """
             tutorials/pyroot/staff.py
        """
        from ROOT import gROOT
        gROOT.ProcessLine( "struct pmt_t { Int_t id ; Double_t x ; Double_t y ; Double_t z ; } ; " )
        from ROOT import pmt_t
        from ROOT import TFile, AddressOf
        f = TFile.Open(PMTMap.pmtmap) 
        t = f.Get(PMTMap.pmtree )
        n = t.GetEntries()

        pj = pmt_t()
        atts = "id x y z".split()
        for a in atts:
            t.SetBranchAddress( "%sPMT" % a[0] , AddressOf( pj , a ) )

        for i in range(n):
            t.GetEntry(i)
            p = PMT()
            for a in atts:setattr( p , a ,   getattr( pj , a , None ))    
            self.append(p)


    def path(self):return os.path.join( "/tmp/aberdeen/AbtViz"  , "PMTMap.pickle" )
    def pysave(self):
        dir = os.path.dirname(self.path())
        if not(os.path.isdir(dir)):os.makedirs(dir) 
        print "%s.pysave to %s " % (self.__class__.__name__ , self.path() )
        pickle.dump( list(self) , file(self.path(), "wb") , pickle.HIGHEST_PROTOCOL )
    def pyload(self):
        self[:] = []
        print "%s.pyload from %s " % (self.__class__.__name__ , self.path() )
        obj = pickle.load( file(self.path(),"rb") )
        for item in obj:
            self.append(item)


    def __repr__(self):
        return "\n".join( [repr(p) for p in self] )


if __name__=='__main__':

    #
    # NB without the import __file__ is undefined and pickling fails  :
    #      PicklingError: Can't pickle __main__.PMT: attribute lookup __main__.PMT failed 
    #
    from pmtmap import PMTMap
    pm = PMTMap()
    print pm
    



