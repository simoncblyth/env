from bsddb3.db import *
from dbxml import *

class myFunction(XmlExternalFunction):
    def __init__(self):
	XmlExternalFunction.__init__(self)
	print "myFunction constructor"

    def execute(self, txn, mgr, args):
        #print "myFunction -- execute"
        results = mgr.createResults()
        results.add(XmlValue("foo"))
        results.add(XmlValue("bar"))
        return results

    def close(self):
        print "myFunction -- close"
        del self
        
    def __del__(self):
        print "myFunction -- del"







class myResolver(XmlResolver):
    def __init__(self):
	XmlResolver.__init__(self)
        self.fun = myFunction()
        self.uri_ = "http://my"

    def getUri(self): return self.uri_

    def resolveExternalFunction(self, txn, mgr, uri, name, numArgs):
        # verify the number of arguments, uri and name which uniquely
        # identify a function in XQuery
        if numArgs == 0 and uri == self.uri_ and name == "foo":
            return self.fun
        else:
            print "myResolver -- could not resolve function"
    
    def __del__(self):
        print "myResolver -- del"


if __name__ == '__main__':
    pass
    r = myResolver()


