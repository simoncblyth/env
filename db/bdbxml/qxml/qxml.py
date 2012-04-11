#!/usr/bin/env python
"""
Pythonic equivalent to qxml.cc 

  ./qxml.py test/extmixed.xq


SWIG PYTHON MEMORY MANAGEMENT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    When DBXML wrapped python objects goes out of scope
    on python side it goes away on C++ side : unless measures are 
    taken to prevent this.

    http://www.swig.org/Doc1.3/Python.html#Python_nn30

    31.4.2 Memory management

    Associated with proxy object, is an ownership flag .thisown 
    The value of this flag determines who is responsible for deleting the underlying C++ object. 
    If set to 1, the Python interpreter will destroy the C++ object when the proxy class is garbage collected. 
    If set to 0 (or if the attribute is missing), then the destruction of the proxy class has no effect on the C++ object.

    When an object is created by a constructor or returned by value, Python automatically takes ownership of the result. 

"""
from __future__ import with_statement 
import os, logging
log = logging.getLogger(__name__)

from bsddb3.db import *
from dbxml import *
from extfun import myResolver
from config import qxml_config, remove_droppings
from pprint import pformat

resolver = myResolver()    ## letting resolver go out of scope, results in XmlExceptions

class QXML(dict):
    def __init__(self, *args, **kwargs ):	
	dict.__init__(self, *args, **kwargs)    
	if len(args) == 0:
            d = qxml_config()
            self.update(d)
	log.debug("QXML __init__ after qxml_config")    
	self.bootstrap()
	log.debug("QXML __init__ DONE")    

    def __repr__(self):
	return pformat(dict(self))

    def bootstrap(self):
	"""
        Bootstrap Berkeley DB XML environment, manager and query context 
	according to the configurations prescription.

	``mgr.env`` provides the enviroment 

	"""
        env = DBEnv()   # no thisown for C wrapper  
        envdir = self["dbxml"]["dbxml.environment_dir"]
        env.open(envdir, DB_CREATE|DB_INIT_MPOOL, 0)
	mgr = XmlManager(env, DBXML_ALLOW_EXTERNAL_ACCESS|DBXML_ADOPT_DBENV) 
	mgr.thisown = False

        resolver.xqmpath = self["dbxml"]["dbxml.xqmpath"]
	mgr.registerResolver(resolver)
	self._containers(mgr) 	

	ctx = mgr.createQueryContext()       
	ctx.thisown = False
	ctx.setNamespace("my", resolver.getUri() )
        self._ctx( ctx )

        self.mgr = mgr 
	self.ctx = ctx
	log.debug("QXML.bootstrap DONE")    

    def __call__(self, q=None ):
	if not q:
	    q = self['query']	
        return self.mgr.query( q , self.ctx )
	#for i, value in enumerate(res):
        #    log.info("value %-3s: %s", i, value.asString()) 
	#return res

    def _containers(self, mgr):
	"""
        Must arrange to prevent python garbage collection of swig proxies 
	from killing the underlying C++ objects.  Normally python objects
	get collected when they go out of scope, giving plenty of opportunities
	for dropping the C++ objects on the floor.

	Approaches that work: 

	#. set ``.thisown = False`` on the proxies
	#. assign to a slot in a global dict, so they never go out of scope 
        #. use monolithic coding style  

	There are not enough containers to worry about leaking.
	"""
	for tag,path in self['containers'].items():
	    if os.path.exists(path):
		log.debug("openContainer %s : %s " % ( tag, path) )    
                cont = mgr.openContainer(path)
                cont.addAlias(tag) 
		cont.thisown = False
	    else:
		log.warn("container %s does not exist " % path )	 
		raise Exception("No such container %s " % path )

    def _ctx( self, ctx ):
        ctx.setDefaultCollection( self["dbxml"]["dbxml.default_collection"] )
        ctx.setBaseURI( self["dbxml"]["dbxml.baseuri"])

        for name,uri in self['namespaces'].items():
	    log.debug("namespaces %s = %s  " % ( name, uri )) 	
	    ctx.setNamespace(name, uri)

	for k,v in self['variables'].items():
	    log.info(" setVariableValue $%s := %s  " % ( k, v )) 	
	    ctx.setVariableValue( k, XmlValue(v) )


class QX(object):
    def __init__(self):
	"""
        A contextlib jacket for QXML, motivation was to avoid duplication of 
	exception handling... unsure if need that now.

        http://www.doughellmann.com/PyMOTW/contextlib/
	"""
        qxm = QXML()
	self.qxm = qxm

    def __enter__(self):
	return self.qxm

    def __exit__(self, etype, e , etb):
	"""
	:return: True to handle expection here, or False to propagate it on 
	"""
	handled = False
	if isinstance(e,XmlException): 
	    log.warn("XmlException (%s) %s " % (  e.exceptionCode, e.what) )
	    if e.exceptionCode == DATABASE_ERROR:
	        log.warn("Database error code: %s " % e.dbError)
            handled = True
        return handled


def test_qx():
    with QX() as q: 
	print q    
        for value in q():
            print "Value: ", value.asString() 

if __name__ == '__main__':
    pass
    qx = QXML()
    res = qx()
    for v in res:
        print v


