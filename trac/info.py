"""
   Provide vital statistics of the trac instances to python, using the 
   intertrac triplet source accessed from bash calls

   A persistent singleton is used to avoid needless rerunning of 
   the expensive bash call 
   
"""


import os

class TracInstance(dict):
    def __init__(self, **kwa):self.update(kwa)
 
from env.structure import Persistent
class TracInfo(Persistent):
    def __init__(self, *args, **kwa ):pass
        
    def init(self, *args, **kwa ):
        self.parse()

    def parse(self):
        from env.bash import Bash
        b = Bash()
        tps = b("trac- tracinter- tracinter-triplets-")
        import re
        self.info={}
        for line in tps.split("\n"):
            a = re.split("\s*", line.strip() )
            if len(a) == 3:
                i = TracInstance( tag=a[0], name=a[1], url=a[2] )
                self.info[a[1]] = i
            else:
                print "failed to interpret line [%s] [%s] " % ( line.strip() , a )

    def __repr__(self):
        from pprint import pformat 
        return pformat(self.info)



if __name__=='__main__':
    from env.trac import TracInfo
    tia = TracInfo(singleton=True)
    tib = TracInfo(singleton=True)
    assert tia == tib 
    print tia
    