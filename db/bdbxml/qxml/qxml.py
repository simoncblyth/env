#!/usr/bin/env python
"""
Pythonic equivalent to qxml.cc 

TODO: pull out config hfagc specialization into ini file 
     (that can be shared with the qxml C++ boost_program_options ) 

"""
import logging
log = logging.getLogger(__name__)
from optparse import OptionParser
from bsddb3.db import *
from dbxml import *
from extfun import myResolver

def parse_args():
    """
    Try to duplicate the boost_program_options interface in C++ qxml
    """
    parser = OptionParser()
    parser.add_option("-l", "--level", default="INFO" )
    parser.add_option("--xqpath",   default=None )
    parser.add_option("--dbxmldir", default="/tmp/hfagc" )
    parser.add_option("--baseuri",  default=""  )
    parser.add_option("-k", "--keys" , action="append" )
    parser.add_option("-v", "--vals" , action="append" )

    cfg, args = parser.parse_args()
    if not cfg.xqpath and len(args)==1:
        cfg.xqpath = args[0]	    

    if cfg.keys and cfg.vals:
        assert len(cfg.keys) == len(cfg.vals), "number of keys must match vals" 
	kv = dict(zip(cfg.keys,cfg.vals))
    else:
	kv = {}    
    pass	
    cfg.kv = kv
    return cfg, args 


def remove_droppings():
    """
    Suspect the need for this to clean up the __db.001
    indicates are missing some memory cleanup ?
    """
    import os, glob	
    files = glob.glob("__db.*")
    for file in files:
        os.remove(file)


if __name__ == '__main__':

    cfg, args = parse_args()	  
    logging.basicConfig(level=getattr(logging, cfg.level.upper()))

    xqpath = cfg.xqpath 
    dbxmldir = cfg.dbxmldir
    baseuri = cfg.baseuri
    vars = cfg.kv

    lines = open(xqpath, "r").readlines()
    if lines[0][0] == "#":  # comment 1st line when 1st char is '#' allowing shebang running  
        lines[0] = "(: %s :)\n" % lines[0].rstrip()
    q = "".join(lines)

    print q

   
    try:
        environment = DBEnv()
        environment.open(None, DB_CREATE|DB_INIT_LOCK|DB_INIT_LOG|DB_INIT_MPOOL|DB_INIT_TXN, 0)
        #environment.open(None, DB_CREATE|DB_INIT_MPOOL, 0)
        #environment.open(None, DB_CREATE, 0)

	mgr = XmlManager(environment,DBXML_ALLOW_EXTERNAL_ACCESS) 

	resolver = myResolver()
	mgr.registerResolver(resolver)

        hfagc_path = dbxmldir + "/hfagc.dbxml"
        hfagc = mgr.openContainer( hfagc_path )
        hfagc.addAlias("hfc") 

        sys_path = dbxmldir + "/hfagc_system.dbxml"
        sys_ = mgr.openContainer( sys_path )
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

        del hfagc
	del sys_
	del qc
        del mgr
        # removeContainer DELETES THEM !!
        #mgr.removeContainer( hfagc_path )
        #mgr.removeContainer( sys_path )
        environment.close(0)
        remove_droppings()


    except XmlException, inst:
	print "XmlException (", inst.exceptionCode,"): ", inst.what
	if inst.exceptionCode == DATABASE_ERROR:
	    print "Database error code:",inst.dbError


    pass 


