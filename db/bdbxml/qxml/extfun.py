"""
http://docs.oracle.com/cd/E17276_01/html/api_reference/CXX/frame.html
"""
import logging
log = logging.getLogger(__name__)

from bsddb3.db import *
from dbxml import *

class myFunction(XmlExternalFunction):
    def __init__(self):
	XmlExternalFunction.__init__(self)
	log.debug("myFunction constructor")

    def execute(self, txn, mgr, args):
        results = mgr.createResults()
        results.add(XmlValue("foo"))
        results.add(XmlValue("bar"))
        return results

    def close(self):
        log.debug("myFunction -- close")
        del self
        
    def __del__(self):
        log.debug("myFunction -- del")


class myDumperFunction(XmlExternalFunction):
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

    def close(self):
        log.debug("Dumper.close")
        del self


class myResolver(XmlResolver):
    def __init__(self):
	XmlResolver.__init__(self)
        self.fun = myFunction()
        self.dumper = myDumperFunction()
        self.uri_ = "http://my"

    def getUri(self): return self.uri_

    def resolveExternalFunction(self, txn, mgr, uri, name, numArgs):
	"""    
        verify the number of arguments, uri and name which uniquely
        identify a function in XQuery
	"""
        if numArgs == 0 and uri == self.uri_ and name == "foo":
            return self.fun
        elif numArgs == 1 and uri == self.uri_ and name == "dumper":
            return self.dumper
        else:
            log.warn("myResolver -- could not resolve function")
    
    def __del__(self):
        log.info("myResolver -- del")


if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.DEBUG)
    r = myResolver()


