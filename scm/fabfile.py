"""
Usage::

    fab        scm_backup_check    # on the env.hosts nodes
    fab -R svn scm_backup_check    # on nodes with role 'svn'


For bare ipython interactive tests::

   from fabric.api import run, env
   env.use_ssh_config = True
   env.hosts = ["WW"]
   paths = run("find $SCM_FOLD/backup/$LOCAL_NODE -name '*.gz' ")


"""

import os, re
from datetime import datetime
from fabric.api import run, env, abort
from fabric.state import output
output.stdout = False
env.use_ssh_config = True
env.hosts = ["WW"]


class Path(str):
    """	
    paths of interest are assumed to contain a datetime encoding string such as 2012/02/21/093002
    NB pattern extent and strptime fmt extent must correspond precisely to allow correct parsing of
    string into datetime
    """
    ptn = re.compile("\/\d{4}\/\d{2}\/\d{2}\/\d{6}\/")
    fmt = "/%Y/%m/%d/%H%M%S/"

    def __init__(self, path):
	str.__init__(self, path)    
	m = self.ptn.search(path)
	assert m, "failed to match path %s " % path 
	start, end = m.span()
        self.bef = path[:start]
	self.dat = path[start:end]
	self.aft = path[end:]
	self.dt  = datetime.strptime(self.dat,self.fmt) 

    def __repr__(self):
	return "%s %s(%s)%s [%s]" % ( self.__class__.__name__, self.bef, self.dat, self.aft, self.dt.strftime("%c") )     



def scm_backup_check():
    """
    Avoid many remote connections by pulling datetime info encoded into path
    rather than querying remote file system.

    #. maybe mac related there are ``\r\n`` in the returned string not ``\n`` 

    """
    paths = run("find $SCM_FOLD/backup/$LOCAL_NODE -name '*.gz' ")
    assert paths.return_code == 0, paths.return_code
    
    pp = map(Path,paths.split())   
    #befs = list(set(map(lambda _:_.bef, pp)))

    # group paths according to their folder, ie string before the date
    dex = {}
    for path in pp:  
        if path.bef not in dex:
            dex[path.bef]=[]
        dex[path.bef].append(path)

    # dump paths within each folder ordered by date
    for k in dex.keys():
        print k
	for p in sorted(dex[k],key=lambda _:_.dt):
            print repr(p)		



