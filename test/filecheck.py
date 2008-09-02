
import ROOT

def assert_dirs( filepath , paths ):
    f = ROOT.TFile(filepath)
    assert not(f.IsZombie()), "missing filepath %s " % filepath  
    for path in paths:
        dir = f.GetDirectory(path)
        assert dir != None , "missing path %s in filepath %s  " %  ( path , filepath ) 

if __name__=='__main__':
    assert_dirs( "gen.root" , [ "/Event" , "/Event/Gen" ] )



