"""
   Provide vital statistics of the trac instances to python 


   Whats a good way to factor out this persistency stuff ?

"""

import cPickle as pickle
import os


class TracInstance(dict):
    def __init__(self, **kwa):self.update(kwa)
     
                           
class TracInfo(dict):
    pfile = '/tmp/TracInfo.p'
    
    @classmethod
    def get(cls):
        ti = TracInfo.load()
        if ti:
             print "providing pickled"
             return ti
        else:
            return TracInfo()
    
    @classmethod
    def save(cls, obj ):
        f = TracInfo.pfile
        print "saving to %s " % f
        pickle.dump( obj , file(f,'w') )
    
    @classmethod
    def load(cls):
        f = TracInfo.pfile
        if os.path.exists(f):
            print "loading from %s " % f
            return pickle.load(file(f))
        else:
            print "failed to load from %s " % f
            return None
    
    def __init__(self):
        self.parse()
        self.save(self)

    def parse(self):
        from env.bash import Bash
        b = Bash()
        tps = b("trac- tracinter- tracinter-triplets-")
        import re
        for line in tps.split("\n"):
            a = re.split("\s*", line.strip() )
            if len(a) == 3:
                i = TracInstance( tag=a[0], name=a[1], url=a[2] )
                self[a[1]] = i
            else:
                print "failed to interpret line [%s] [%s] " % ( line.strip() , a )


if __name__=='__main__':
    from env.trac import TracInfo
    ti = TracInfo.get()
    print ti