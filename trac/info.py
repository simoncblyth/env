"""
   Provide vital statistics of the trac instances to python, using the 
   intertrac triplet source accessed from bash calls

   A persistent dict is used to avoid needless rerunning of 
   the expensive bash call 
   
"""


import os

class TracInstance(dict):
    def __init__(self, **kwa):self.update(kwa)
 
from env.structure import PerDict
class TracInfo(PerDict):
    def __init__(self):
        self.parse()
        self.psave(self)

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
    ti = TracInfo.pget()
    print ti