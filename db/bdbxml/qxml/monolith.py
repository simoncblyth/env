#!/usr/bin/env python
"""
Pythonic equivalent to qxml.cc 

  ./qxml.py test/extmixed.xq

"""
from __future__ import with_statement 
import os, logging
log = logging.getLogger(__name__)

from bsddb3.db import *
from dbxml import *
from extfun import myResolver
from config import qxml_config, remove_droppings


def test_monolithic():
    cfg = qxml_config()
    try:
        environment = DBEnv()
        environment.open(None, DB_CREATE|DB_INIT_MPOOL, 0)
	mgr = XmlManager(environment,DBXML_ALLOW_EXTERNAL_ACCESS) 
	resolver = myResolver()
	mgr.registerResolver(resolver)

        for tag,path in cfg['containers'].items():
	    log.info(" containers %s = %s  " % ( tag, path )) 	
            cont  = mgr.openContainer(path)
            cont.addAlias(tag) 

        qc = mgr.createQueryContext()       

	qc.setNamespace("my", resolver.getUri())
        qc.setDefaultCollection( cfg["dbxml"]["dbxml.default_collection"] )
        qc.setBaseURI( cfg["dbxml"]["dbxml.baseuri"])

        for name,uri in cfg['namespaces'].items():
	    log.info(" namespaces %s = %s  " % ( name, uri )) 	
	    qc.setNamespace(name, uri)

	for k,v in cfg['variables'].items():
	    log.info(" setVariableValue $%s := %s  " % ( k, v )) 	
	    qc.setVariableValue( k, XmlValue(v) )

        res = mgr.query( cfg['query'] , qc )

	for value in res:
            print "value: ", value.asString() 

        del mgr
        #environment.close(0)
        #remove_droppings()

    except XmlException, e:
	print "XmlException (", e.exceptionCode,"): ", e.what
	if e.exceptionCode == DATABASE_ERROR:
	    print "Database error code:",e.dbError
    pass 

if __name__ == '__main__': 
    test_monolithic()	
	
