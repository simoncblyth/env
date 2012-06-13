#!/usr/bin/env python
"""
Provide the checks used in the fabfile, 

#. keep fabric specifics in the fabfile 
#. minimize remote connections by

    #. doing a single remote find to get the paths and sizes   
    #. pulling datetime info encoded into path rather than querying remote file system.

Curiously:

#. there are windows newlines ``\r\n`` in the returned string not ``\n`` 


"""
import os, re, platform, logging
log = logging.getLogger(__name__)
from datetime import datetime
from simtab import Table


class Path(str):
    """	
    paths of interest are assumed to contain a datetime encoding string such as 2012/02/21/093002
    NB pattern extent and strptime fmt extent must correspond precisely to allow correct parsing of
    string into datetime

    #. ``dir`` of the path is the portion before the encoded datetime

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
        self.size = None

    def __repr__(self):
	return "%s %s(%s)%s [%s] [%s]" % ( self.__class__.__name__, self.dir, self.dat, self.aft, self.date, self.size )     



class GZCheck(object):
    def __init__(self, dbpath):
        """
        :param dbpath: path to SQLite db 
        """
        localnode = platform.node()
        self.cmd = "find $SCM_FOLD/backup/%s -name '*.gz' -exec du --block-size=1M {} \;" % localnode.split(".")[0]
        self.tgzs = Table(dbpath, "tgzs", nodepath="text primary key", node="text", dir="text", date="text", size="real" )
   
    def __call__(self, lines, node ):
        """
        Parse the response from the remote command and update local database,    

        :param lines: list of strings response from the cmd
        :param node: ssh tag or alias of remote node on which the command was performed  

        """
        log.info("%s lines from node %s " % ( len(lines), node ) ) 
        for line in lines: 
            fields = line.split("\t")
            assert len(fields) == 2, "unexpected field count : %s " % repr(fields)
            size, path_ = fields
            path = Path(path_)
            path.size = size
            nodepath="%s:%s" % ( node, path )
            self.tgzs.add( node=node, nodepath=nodepath, size=size, dir=path.dir, date=path.date )
            pass 
        self.tgzs.insert()    # new entries added, changed entries replace older ones

    def check(self):
        self._check_current()

    def _check_current(self):
        """
        Checking the current entries, without using persisted prior entries
        """
        # group paths according to their folder, ie string before the date
        dex = {}
        for d in self.tgzs:  
            dir = d['dir']
            if dir not in dex:
                dex[dir]=[]
            dex[dir].append(d)

        # dump paths within each folder ordered by date
        for k in dex.keys():
            print k
	    for p in sorted(dex[k],key=lambda _:_['date'] ):
                print repr(p)		


if __name__ == '__main__':
    pass
