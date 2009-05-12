

def create(path):
    """
        test files with directory structure
    """ 
    from ROOT import TFile, TH1F, gDirectory
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
            h = TH1F( n, p, 2,-1,1)
            h.FillRandom("gaus")
            h.Write()
            h.Delete()
    f.Close()



class FileMap:
    """
         Working out approaches to 
         generic resource access from .root file 
    """
    def __init__(self, path):
        from ROOT import TFile, gDirectory
        f = TFile.Open(path)
        self.f = f

    def ls(self, cls=[]):
        for k in self.keys(cls):
            o = k.ReadObj()
            p = k.GetMotherDir().GetPath()
            print "%-40s %-20s %s" % ( p, o.GetName(), o.GetTitle() ) 

    def keys(self, cls=[]):
        """ recursively list keys of desired cls """ 
        return self.keys_(self.f, cls)
 
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


if __name__=='__main__':
    path = "ex.root" 
    import os
    if not(os.path.exists(path)):
        create(path)
    fm = FileMap(path)
    fm.ls()
    


