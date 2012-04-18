#!/usr/bin/env python
"""
GETs a single resource from an eXist DB and adds it to target dbxml container, default 'sys'
A directory GET is also performed in order to propagate the metadata of the resource.

Usage::

     ./transfer.py -u http://localhost/servlet/db/hfagc_system/qtag2latex.xml 
     ./transfer.py -u http://cms01.phys.ntu.edu.tw/servlet/db/test/lhcb_winter2011_BcX.xml

TODO: fold into QXML methods

"""
import os, logging
log = logging.getLogger(__name__)

from qxml import QXML
from common import existsDoc, urlDoc, ExistDirQuery

if __name__ == '__main__':
    pass	
    qx = QXML()


    url = qx.cfg['cli']['url'] 
    tgt = qx.cfg['cli']['target']
    dbxml = qx.cfg['containers'][tgt]

    assert os.path.exists(dbxml)
    assert url and url.startswith('http://')
    name = os.path.basename( url )
    dirurl = os.path.dirname( url )

	
    # pull directory xml of resource to get metadata 
    edq = ExistDirQuery(qx.mgr)
    collections, resources = edq(dirurl)
    dir = dict((d['name'],d) for d in resources)
    meta = dir[name]
    log.info("url %s dirurl %s name %s meta %s " % ( url, dirurl, name, meta ))


    cont = qx.mgr.openContainer(dbxml)
    uctx = qx.mgr.createUpdateContext()   
    if existsDoc( name, cont):
        log.info("deleting pre-existing document %s from container %s " % (name, dbxml) )	
    	cont.deleteDocument( name , uctx ) 	
    else:
        log.info("no pre-existing document %s from container %s " % (name, dbxml) )	

    doc = urlDoc(qx.mgr, url, name=name, meta=meta )
    log.info("putDocument %s from %s into container %s " % (name,url,dbxml) )	
    cont.putDocument( doc , uctx, 0 ) 





