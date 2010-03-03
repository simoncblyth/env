import ROOT

def soext():
	"""
	    Smth funny with Darwin ROOT library loading ... 
         ROOT.gSystem.GetSoExt()  is giving "so" when expect "dylib"      

      from TSystem.cxx this SOEXT is set at compile time ...
         $ROOTSYS/include/compiledata.h

         workaround :
	          ln -s libAbtDataModel.dylib libAbtDataModel.so
     
	"""
	soext = "so"
	if ROOT.gSystem.GetBuildArch() == "macosx":soext = "dylib"
	return soext

class EvMQ:
    def __init__(self, key="default.routingkey"):
        """
              Need library path to include 
                 $ENV_HOME/notifymq/lib
                 $ABERDEEN_HOME/DataModel/lib
        """
        ROOT.gSystem.Load("libnotifymq" )
        ROOT.gSystem.Load("libAbtDataModel" )
 
        ROOT.gMQ.Create()
        self.key = key 
        self.mq = ROOT.gMQ
        self.timer = ROOT.TTimer(1000)
        self._connect( self.timer, "TurnOn()",  self.On ) 
        self._connect( self.timer, "Timeout()", self.Check ) 
        self._connect( self.timer, "TurnOff()", self.Off ) 
        self.timer.TurnOn()
        self.obj = None

    def On(self):
        print "EvMQ.On"
        self.mq.StartMonitorThread()
    def Check(self):
        #print "EvMQ.Check"
        if self.mq.IsMonitorRunning():
            if self.mq.IsUpdated(self.key):
                obj = self.mq.Get(self.key, 0)
                if obj:
                    obj.Print("")
                    self.obj = obj
    def Off(self):
        print "EvMQ.Off"

    def stop(self):
        self.timer.TurnOff()

    def _connect(self, obj, sign , method ):
        handlerName = "_%s" % method.__name__
        setattr( self , handlerName , ROOT.TPyDispatcher( method ) )
        obj.Connect( sign , "TPyDispatcher", getattr( self , handlerName )  , "Dispatch()" ) 

    def __repr__(self):
        return repr(self.mq)



if __name__=='__main__':
    from evmq import EvMQ
    emq = EvMQ()
    #ROOT.gSystem.Sleep(1000*10)
    #emq.stop()


    



