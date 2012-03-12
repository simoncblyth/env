#!/usr/bin/env python
"""
Pythonic equivalent to qxml.cc 

"""

from optparse import OptionParser
from bsddb3.db import *
from dbxml import *
from extfun import myResolver


def parse_args():
    """
    Try to duplicate the boost_program_options interface in C++ qxml
    """
    parser = OptionParser()
    parser.add_option("--xqpath",   default=None )
    parser.add_option("--dbxmldir", default="/tmp/hfagc" )
    parser.add_option("--baseuri",  default=""  )
    parser.add_option("-k", "--keys" , action="append" )
    parser.add_option("-v", "--vals" , action="append" )

    cfg, args = parser.parse_args()
    if not cfg.xqpath and len(args)==1:
        cfg.xqpath = args[0]	    

    assert len(cfg.keys) == len(cfg.vals), "number of keys must match vals" 
    cfg.kv = dict(zip(cfg.keys,cfg.vals))
    print cfg.kv
    return cfg, args 

if __name__ == '__main__':

    cfg, args = parse_args()	  

    xqpath = cfg.xqpath 
    dbxmldir = cfg.dbxmldir
    baseuri = cfg.baseuri
    vars = cfg.kv

    lines = open(xqpath, "r").readlines()
    if lines[0][0] == "#":  # ignore 1st line when 1st char is '#' allowing shebang running  
        lines = lines[1:]
    q = "".join(lines)

    print q

    environment = DBEnv()
    #environment.open(None, DB_CREATE|DB_INIT_LOCK|DB_INIT_LOG|DB_INIT_MPOOL|DB_INIT_TXN, 0)
    environment.open(None, DB_CREATE|DB_INIT_MPOOL, 0)
    
    try:
	mgr = XmlManager(environment,DBXML_ALLOW_EXTERNAL_ACCESS) 

	resolver = myResolver()
	mgr.registerResolver(resolver)

        hfagc = mgr.openContainer( dbxmldir + "/hfagc.dbxml")
        hfagc.addAlias("hfc") 

        sys_ = mgr.openContainer( dbxmldir + "/hfagc_system.dbxml")
        sys_.addAlias("sys") 

        qc = mgr.createQueryContext()        

	qc.setNamespace("rez","http://hfag.phys.ntu.edu.tw/hfagc/rez")

	qc.setNamespace("my", resolver.getUri())

        qc.setDefaultCollection("dbxml:///" + dbxmldir + "/hfagc.dbxml")
        qc.setBaseURI( baseuri )

	for k,v in vars.items():
            xv = XmlValue(v)		
	    print " setVariableValue $%s := %s  %s " % ( k, v , xv ) 	
	    qc.setVariableValue( k, xv )

        res = mgr.query( q , qc)

	for value in res:
            print "Value: ", value.asString() 

    except XmlException, inst:
	print "XmlException (", inst.exceptionCode,"): ", inst.what
	if inst.exceptionCode == DATABASE_ERROR:
	    print "Database error code:",inst.dbError

