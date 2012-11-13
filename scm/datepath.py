#!/usr/bin/env python
import re
from datetime import datetime




class Path(str):
    """    
    paths of interest are assumed to contain a datetime encoding string such as 2012/02/21/093002
    NB pattern extent and strptime fmt extent must correspond precisely to allow correct parsing of
    string into datetime

    #. ``dir`` of the path is the portion before the encoded datetime

    .. warn:: implicit assumption that timezone of monitored and monitor machines are the same

    """
    ptn = re.compile("\/\d{4}\/\d{2}\/\d{2}\/\d{6}\/")
    fmt = "/%Y/%m/%d/%H%M%S/"

    def __init__(self, path):
        str.__init__(self, path)    
        m = self.ptn.search(path)
        assert m, "failed to match path %s " % path 
        start, end = m.span()
        self.dir = path[:start]
        self.dat = path[start:end]
        self.aft = path[end:]
        dt  = datetime.strptime(self.dat,self.fmt) 
        self.dt = dt 
        self.date = dt.strftime("%Y-%m-%dT%H:%M:%S")   # SQLite can handle this 

    def __repr__(self):
        return "%s %s(%s)%s [%s]" % ( self.__class__.__name__, self.dir, self.dat, self.aft, self.date )     



if __name__ == '__main__':
    pass    

    p = Path("/hello/world/2012/11/13/120001/ok")
    print p.date


