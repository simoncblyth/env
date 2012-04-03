"""
http://docs.oracle.com/cd/E17276_01/html/api_reference/CXX/frame.html
"""
import os, logging
log = logging.getLogger(__name__)

from bsddb3.db import *
from dbxml import *

from pyextfun import MyExternalFunctionPow 
from pyextfun import MyExternalFunctionSqrt 


class MyExternalFunction(XmlExternalFunction):
    def __init__(self):
	XmlExternalFunction.__init__(self)
	self.thisown = False
	log.debug("%s __init__ " % self.__class__.__name__ )
    def __del__(self):
	log.debug("%s __del__ " % self.__class__.__name__ )
    def close(self):
        log.debug("%s close" %  self.__class__.__name__ )
        del self
     

class Foo(MyExternalFunction):
    def execute(self, txn, mgr, args):
        results = mgr.createResults()
        results.add(XmlValue("foo"))
        results.add(XmlValue("bar"))
        return results

class Dumper(MyExternalFunction):
    """
    Checking the passing of XML nodes to extension functions
    """
    def execute(self, txn, mgr, args):
        """
        :param args: XmlArguments

        Iterating over XmlResults provides XmlValue instances which have DOM like methods
        """ 
        nargs = args.getNumberOfArgs()
	if nargs != 1: raise Exception("unexpected number of arguments %s " % nargs)

	iresults = args.getArgument(0)   
        assert type(iresults) == XmlResults

        for i, value in enumerate(iresults):
            log.debug("Dumper.execute %s %s " % (i, value.asString()) ) 

        results = mgr.createResults()
        #results.add(XmlValue(""))
        return results



class Quote2Values(MyExternalFunction):
    """	


           <rez:value>
                <rez:number>1.5</rez:number>
                <rez:unit>0.00001</rez:unit>
                <rez:signif>4.9</rez:signif>
                <rez:ltype>-</rez:ltype>
                <rez:limit/>
                <rez:cl/>
            </rez:value>
            <rez:errors>
                <rez:err>
                    <rez:errname>stat</rez:errname>
                    <rez:type>absolute</rez:type>
                    <rez:symm/>
                    <rez:plus>0.5</rez:plus>
                    <rez:minus>0.4</rez:minus>
                </rez:err>
                <rez:err>
                    <rez:errname>syst</rez:errname>
                    <rez:type>absolute</rez:type>
                    <rez:symm>0.1</rez:symm>
                    <rez:plus/>
                    <rez:minus/>
                </rez:err>
            </rez:errors>
            <rez:xerrors>
                <rez:xerr>
                    <rez:errname>Ds BR</rez:errname>
                    <rez:type>absolute</rez:type>
                    <rez:symm>0.2</rez:symm>
                    <rez:plus/>
                    <rez:minus/>
                </rez:xerr>
            </rez:xerrors>


    """
    def execute(self, txn, mgr, args):
        """
        :param args: XmlArguments

        Iterating over XmlResults provides XmlValue instances which have DOM like methods
        """ 
        nargs = args.getNumberOfArgs()
	if nargs != 1: raise Exception("unexpected number of arguments %s " % nargs)

	arg0 = args.getArgument(0)   
        assert type(arg0) == XmlResults
        value = arg0.next()
        log.info("Q2V arg %s %s" % (value.__class__.__name__, value.asString()) ) 
	rdr = value.asEventReader()

	i = 0 
        while rdr.hasNext():
            i += 1		 
	    typ = rdr.next()	
	    if   typ == XmlEventReader.StartElement:
		log.info("Q2V %s start %s" % (i, rdr.getLocalName()) )     
	    elif typ == XmlEventReader.EndElement:
		log.info("Q2V %s end %s" % (i, rdr.getLocalName()) )     
	    else:	     
                log.info("Q2V %s typ %s %s " % (i, typ,""))		 


        results = mgr.createResults()
        #results.add(XmlValue(""))
        return results

	


class myResolver(XmlResolver):
    """
    Hmm not a very sustainable approach
    """
    def __init__(self):
	XmlResolver.__init__(self)
        self.uri_ = "http://my"
        log.debug("init resolver with uri %s" % self.uri_)
	self.thisown = False

        self.foo = Foo()
        self.dumper = Dumper()
        self.quote2values = Quote2Values()

	self.pow = MyExternalFunctionPow()
	self.sqrt = MyExternalFunctionSqrt()
        self.xqmpath = ""

        log.info("%s __init__" % self.__class__.__name__ )

    def getUri(self): return self.uri_

    def find_entity(self, systemId, publicId):
	"""
	Search directories of QXML_ENTITYPATH for named entity, typically an xqm module
	"""
	xqmpath = filter(os.path.exists, self.xqmpath.split(":"))
	if len(xqmpath) == 0:
            log.warn("no existing directories in xqmpath ")		
        for dir in xqmpath:
            path = os.path.join( dir, systemId )
	    if os.path.exists(path):
		return path
	return None

    def resolveEntity(self, txn, mgr, systemId, publicId):
	"""
	allows importing xqm off an directory path
	"""
        path = self.find_entity( systemId, publicId )
	log.info("resolveEntity %s %s => %s " % (systemId, publicId, path))
	return mgr.createLocalFileInputStream(path) if path else None

    #def resolveModule(self, txn, mgr, moduleLocation, nameSpace):
    #	log.info("resolveModule %s %s " % (moduleLocation, nameSpace))
    #	return None

    def resolveExternalFunction(self, txn, mgr, uri, name, numArgs):
	"""    
        verify the number of arguments, uri and name which uniquely
        identify a function in XQuery
	"""
	log.info("resolveExternalFunction name %s numArgs %s " % ( name, numArgs) )
        if uri != self.uri_:
            log.warn("myResolver -- wrong uri ")
	    raise Exception("resolver wrong uri")
    	    
        sig = (numArgs, name)
        log.debug("resolve external with sig %s " % repr(sig) )

        if sig == (0,"foo"):
            return self.foo
        elif sig == (1,"quote2values"):
            return self.quote2values
        elif sig == (1,"dumper"):
            return self.dumper
        elif sig == (1,"sqrt"):
            return self.sqrt
        elif sig == (2,"pow"):
            return self.pow
        else:
            log.warn("myResolver -- could not resolve function")
	    return None
    
    def __del__(self):
        log.info("%s __del__" % self.__class__.__name__ )


if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.DEBUG)
    r = myResolver()


