"""
http://docs.oracle.com/cd/E17276_01/html/api_reference/CXX/frame.html
"""
import logging
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

        log.info("%s __init__" % self.__class__.__name__ )

    def getUri(self): return self.uri_

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


