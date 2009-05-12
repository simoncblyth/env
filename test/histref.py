
import os

class HistRef:
    def __init__(self, path ):
        self.path = path

    def refpath(self):
        dir = os.path.dirname(self.path)
        base = os.path.basename(self.path)
        return os.path.join( dir, "histref_"+base )

    def __call__(self, **kwa ):
        path = self.path
        if not(os.path.exists(path)):
            print "HistRef ABORT no path %s " % path 
            return 1      

        refp = self.refpath()
        if not(os.path.exists(refp)):
            print "HistRef no reference, so blessing path %s as the future reference " % path 
            import shutil
            shutil.copy( "%s %s" % ( path , refp))  
            return 0

        x = XmlRoot(refp)
        y = XmlRoot(path)
        xy = XmlCfRoot( x, y )

    def __repr__(self):
        return "<HistRef %s >" % self.path 


