#!/usr/bin/env python
"""

./transfer.py -u http://localhost/servlet/db/hfagc_system/qtag2latex.xml 

"""
import os, logging
log = logging.getLogger(__name__)
from dbxml import *
from config import qxml_config
from common import existsDoc, urlDoc, ExistDirQuery

if __name__ == '__main__':
    pass	
    cfg = qxml_config()
    url = cfg['cli']['url'] 
    tgt = cfg['cli']['target']
    dbxml = cfg['containers'][tgt]

    assert url and url.startswith('http://')
    assert os.path.exists(dbxml)
	
    name = os.path.basename( url )
    dirurl = os.path.dirname( url )

    mgr = XmlManager()
    uctx = mgr.createUpdateContext()   # do i need 2 of these ?

    # look at directory to determine the metadata of the resource 
    edq = ExistDirQuery(mgr)
    collections, resources = edq( dirurl )
    dir = dict((d['name'],d) for d in resources)
    meta = dir[name]

    log.info("url %s dirurl %s name %s meta %s " % ( url, dirurl, name, meta ))

    doc = urlDoc(mgr, url, name=name, meta=meta )
    cont = mgr.openContainer(dbxml)
    if existsDoc( name, cont):
        log.info("deleting pre-existing document %s from container %s " % (name, dbxml) )	
	cont.deleteDocument( name , uctx ) 	

    log.info("putDocument %s from %s into container %s " % (name,url,dbxml) )	
    cont.putDocument( doc , uctx, 0 ) 





