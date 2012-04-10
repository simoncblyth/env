#!/usr/bin/env python
"""
Hmm add to config

[container.source] 

"""
import os, logging
log = logging.getLogger(__name__)
from dbxml import *
from config import qxml_config

# os.path.relpath only from py26
relpath = lambda path,root:path[len(root):]      # keep leading slash or not ?
	
from existmeta import ExistMeta

def ingest_root( tag, srcdir , dbxml ):
    """
    :parm tag: alias string of the container to be created
    :param srcdir: exist backup directory to ingest into dbxml container
    :param dbxml: path of dbxml container to be created
    """
    if srcdir == "":
	log.debug("skipping tag %s dbxml %s as invalid srcdir " % ( tag, dbxml ))
	return
    elif not(os.path.isdir(srcdir)):
	log.warn("srcdir \"%s\" does not exist skip ingest into %s " % ( srcdir , dbxml ))     
	return
    else:
        log.info("ingest %s creating %s from xml files from %s " % ( tag, dbxml, srcdir ))
	pass

    try:
        mgr = XmlManager()
	xmeta = ExistMeta(mgr)
        metaname = ExistMeta.metaname
	cont = mgr.createContainer(dbxml)
	ctx = mgr.createUpdateContext()
    
        for (dirpath, dirnames, filenames) in os.walk( srcdir ):
	    rdir = relpath(dirpath, srcdir )

            if metaname in filenames:
		dirmeta = xmeta( os.path.join( dirpath, metaname ) ) 

	    for name in filter(lambda _:_ != metaname, filenames):
                p = os.path.join(dirpath,name)  
                n = os.path.join(rdir,name)
		xm = dirmeta[name]
                print xm
	        stm = mgr.createLocalFileInputStream(p)
                doc = mgr.createDocument()
		doc.setName(n)
		doc.setContentAsXmlInputStream(stm)
		for key, val in xm.items():
                    doc.setMetaData( ExistMeta.namespace , key, XmlValue(val) )
		    pass
                cont.putDocument( doc , ctx, 0 ) 
                #cont.putDocument( n, stm, ctx , 0)
	    pass


    except XmlException, e:
	print "XmlException (", e.exceptionCode,"): ", e.what
	if e.exceptionCode == DATABASE_ERROR:
	    print "Database error code:",e.dbError
    pass 

if __name__ == '__main__':
    cfg = qxml_config()

    tagsrc = cfg['srcdir'].keys()
    tagcon = cfg['containers'].keys()
    assert tagsrc == tagcon , (tagsrc, tagcon )

    for tag in tagsrc:
	srcdir = cfg['srcdir'][tag]    
	dbxml  = cfg['containers'][tag]    
        ingest_root( tag, srcdir, dbxml )	    
