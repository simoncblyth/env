

class KeyList:
    """
         Recursive walk the TDirectory structure inside 
         a single TFile providing list access to all keys that 
         hold instances of classes within the cls list or all classes
         if cls is an empty list 

         Usage examples:

              kl = KeyList( "path/to/histos.root" , ['TH1F', 'TH2F'] )
              list(kl)
              print len(kl)
              for k in kl:
                 print k
              print kl[0], kl[-1]
 
    """
    def __init__(self, path, cls ):
        from ROOT import TFile, gDirectory
        f = TFile.Open(path)
        self.keys = self.keys_( f , cls)

    def keys_(self, tdir, cls):
        sk = []
        for k in tdir.GetListOfKeys():
            kls = k.GetClassName()
            kok = kls in cls or cls == []
            if kls == 'TDirectoryFile' or kok:
                if kls == 'TDirectoryFile':
                    o = k.ReadObj()
                    for kk in self.keys_(o, cls):
                        sk.append( kk )
                elif kok:
                    sk.append( k )            
        return sk 

    def __len__(self):
        return len(self.keys)
    def __getitem__(self, n ):
        return self.keys[n]
    def __repr__(self):
        return "<KeyList %s > " % ( len(self) ) 




# make TKey easier to work with 
from ROOT import TKey
def TKey_GetCoords(self):
    """
        provides filename, directory within the file and key name
           ['ex2.root', 'red/aa/bb/cc', 'h2', 'TH1F' ]
    """ 
    apath = self.GetMotherDir().GetPath() 
    return apath.split(":/") + [self.GetName(), self.GetClassName()]
def TKey_GetIdentity(self):
    """
         skip the file name 
    """
    cds = self.GetCoords()
    return ":".join( cds[1:] )

def TKey_GetTag(self):
    cds = self.GetCoords()
    return ":".join(cds)
    
def TKey__repr__( self ):return "<TKey %s >" % ( self.GetCoords() )
TKey.__repr__  = TKey__repr__
TKey.GetCoords   = TKey_GetCoords
TKey.GetIdentity = TKey_GetIdentity
TKey.GetTag      = TKey_GetTag





class CfRoot:
    """
          Facilitate comparisons between multiple root files by holding 
          KeyList's into each of them, allowing list access 
          to corresponding objects from all the files 

          Usage examples : 
              cf = CfRoot(["ex1.root","ex2.root","ex3.root"], ['TH1F','TH2F'] )
              if not(cf.is_valid()):
                  print cf
              else:           
                  cf.compare()    

    """
    kolmogorov_cut = 0.9

    def __init__(self, paths, cls ):
        assert len(paths) > 1, "CfRoot: requires 2 or more paths to root files "
        keymap = {}
        for p in paths:
            keymap[p] = KeyList(p, cls )  

        self.keymap = keymap
        self.log = []
        self.rc  = 0

    def cf_len(self):
        nk = {}
        for p, kl in self.keymap.items():
            nk[p] = len(kl)
        if len(set(nk.values())) != 1:
            self.log.append( "inconsistent numbers of items %s " % str(nk) )
            self.rc = 1

    def cf_identity(self):
        for i in range(len(self)):
            self.cf_identity_(i)
    def cf_identity_(self, i):
        kid = {}
        for p, kl in self.keymap.items():
            kid[p] = kl[i].GetIdentity()
        if len(set(kid.values())) != 1:
            self.log.append( "inconsistent identities for index %s ...  %s " % (i, str(kid)) )
            self.rc = 2

    def is_valid(self):
        self.cf_len()
        if self.rc != 0:return False
        self.cf_identity()
        if self.rc != 0:return False
        return True


    def compare(self):
        for i in range(len(self)):
            self.compare_(i)
    def compare_(self, i ):
        ks = self[i]
        
        k0 = ks[0]
        h0 = k0.ReadObj() 
        if not(h0.InheritsFrom("TH1")): 
            self.log.append("class not supported in comparison %s %s " % ( i , k0 ) )
            self.rc = 3 
            return False           
       
        cut = CfRoot.kolmogorov_cut
        kmt = {}
        for k in ks[1:]:
            h = k.ReadObj()
            cft = "compare_%s_%s_%s" % ( k0.GetTag() , k.GetTag() , cut ) 
            kmt[cft] = h0.KolmogorovTest(h)

        if min(kmt.values()) < cut:
            self.log.append("kolmogarov histo consistency failure %s %s " % ( i, str(kmt) ) )
            self.rc = 4
            return False


    def summary(self):
        return "<CfRoot rc:%s >" % self.rc
    def __repr__(self):
        return "\n".join( [self.summary()] + self.log )
    def __len__(self):
        return len(self.keymap.values()[0])
    def __getitem__(self, i ):
        cf = [] 
        for p, kl in self.keymap.items():
            cf.append( kl[i] )
        return cf        
 




def create(path):
    """
        Create test .root files containing TH1F, TH2F within a directory structure
    """ 
    from ROOT import TFile, TH1F,TH2F, gDirectory
    f = TFile.Open( path ,"recreate")   
    for dir in "/ red/aa/bb/cc green/dd/ee blue".split():
        f.cd()
        if dir!="/":
            for e in dir.split("/"):
                gDirectory.mkdir(e)
                gDirectory.cd(e)

        for i in range(3):
            n = "h%s" % ( i )
            p = gDirectory.GetPath()
            if i == 2:
                h = TH2F( n, p, 2,-1,1, 2, -1, 1)
            else:
                h = TH1F( n, p, 2,-1,1)
            h.FillRandom("gaus")
            h.Write()
            h.Delete()
    f.Close()
    print "created %s " % path 





if __name__=='__main__':
    import os
    klss = ['TH1F', 'TH2F' ]
    paths = ["ex1.root","ex2.root"] 
    for p in paths:
        if not(os.path.exists(p)):
            create(p)

    #kl = KeyList(paths[0], klss )
    cf = CfRoot(paths,  klss )
    if cf.is_valid():
        cf.compare()
    print cf
                
       


