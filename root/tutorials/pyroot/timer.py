import ROOT

class Timer(ROOT.TTimer):
    def __init__(self, *args):
        ROOT.TTimer.__init__(self, *args)
        self._connect( "Timeout()", self.timeout )
        self._connect( "TurnOn()",  self.turnoff )
        self._connect( "TurnOff()", self.turnon )
    def _connect(self, sign , method ):
        """
            dynamically adds methods to this class with the requisite TPyDispatcher wrapping 
            that enables them to be invoked when sigs from the timer are emitted
        """
        handlerName = "_%s" % method.__name__
        setattr( self , handlerName , ROOT.TPyDispatcher( method ) )
        self.Connect( sign , "TPyDispatcher", getattr( self , handlerName )  , "Dispatch()" ) 
    def timeout(self):
        print "timeout"
    def turnoff(self):
        print "turnoff"
    def turnon(self):
        print "turnon"


if __name__=='__main__':
    t = Timer(1000)
    t.TurnOn()

 
