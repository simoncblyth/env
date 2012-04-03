#!/usr/bin/env python
"""
Pythonic equivalent to qxml.cc 

  ./monolith.py test/extmixed.xq

Monolithic approach is much easier wrt memory management as nothing 
goes out of scope, but makes it inconvenient to pull up into ipython session 


"""
from __future__ import with_statement 
import os, logging
log = logging.getLogger(__name__)

from bsddb3.db import *
from dbxml import *
from extfun import myResolver
from config import qxml_config


if __name__ == '__main__': 
    cfg = qxml_config()
    try:
        environment = DBEnv()
        envdir = cfg["dbxml"]["dbxml.environment_dir"]
        environment.open(envdir, DB_CREATE|DB_INIT_MPOOL, 0)
	mgr = XmlManager(environment,DBXML_ALLOW_EXTERNAL_ACCESS) 

	resolver = myResolver()
	mgr.registerResolver(resolver)

        for tag,path in cfg['containers'].items():
	    log.debug(" containers %s = %s  " % ( tag, path )) 	
            cont  = mgr.openContainer(path)
            cont.addAlias(tag) 

        qc = mgr.createQueryContext()       

	qc.setNamespace("my", resolver.getUri())
        qc.setDefaultCollection( cfg["dbxml"]["dbxml.default_collection"] )
        qc.setBaseURI( cfg["dbxml"]["dbxml.baseuri"])

        for name,uri in cfg['namespaces'].items():
	    log.debug(" namespaces %s = %s  " % ( name, uri )) 	
	    qc.setNamespace(name, uri)

	for k,v in cfg['variables'].items():
	    log.info(" setVariableValue $%s := %s  " % ( k, v )) 	
	    qc.setVariableValue( k, XmlValue(v) )

        res = mgr.query( cfg['query'] , qc )

	for value in res:
            print "value: ", value.asString() 

        del mgr
        #environment.close(0)

    except XmlException, e:
	print "XmlException (", e.exceptionCode,"): ", e.what
	if e.exceptionCode == DATABASE_ERROR:
	    print "Database error code:",e.dbError
    pass 

	
