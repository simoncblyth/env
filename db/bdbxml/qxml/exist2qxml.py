#!/usr/bin/env python
"""
Hmm add to config

[container.source] 

"""
import os, logging
log = logging.getLogger(__name__)
from dbxml import *
from config import qxml_config

def ingest( root , dbxml ):
    """
    :param root: exist backup directory to ingest into dbxml container
    :param dbxml: path of dbxml container to be created
    """
    try:
        mgr = XmlManager()
	cont = mgr.createContainer(dbxml)
	ctx = mgr.createUpdateContext()

        for p,n in walk(root):
	    stm = mgr.createLocalFileInputStream(p)
            cont.putDocument( n, stm, ctx , 0)
	    pass

    except XmlException, e:
	print "XmlException (", e.exceptionCode,"): ", e.what
	if e.exceptionCode == DATABASE_ERROR:
	    print "Database error code:",e.dbError
    pass 


# os.path.relpath only from py26
relpath = lambda path,root:path[len(root):]      # keep leading slash or not ?
	
def walk( root ):
    for (dirpath, dirnames, filenames) in os.walk( root ):
	 rdir = relpath(dirpath, root)
	 for name in filenames:   
             if name != '__contents__.xml':
                 p = os.path.join(dirpath,name)  
                 n = os.path.join(rdir,name)
		 yield p,n 

if __name__ == '__main__':
    cfg = qxml_config()
    print cfg
    #for p,n in walk("/data/heprez/data/backup/part/localhost/last"):  
    #    print "%-80s %s" % ( n, p )
    #
