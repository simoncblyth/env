import ROOT

class EvMQ:
    """
        Timer based architecture to handle controlled maximum frequency 
        updating of an event display 

        As new event messages can arrive faster than would want to 
        display them, establish a "pulse" via a timer allowing new messages 
        to be checked for every second (or so) while still providing 
        interactive ipython 

    """
    def __init__(self, keys=["default.routingkey"]):
        """
              Need library path to include 
                 $ENV_HOME/rootmq/lib
                 $ABERDEEN_HOME/DataModel/lib
        """
        if ROOT.gSystem.Load("librootmq" ) < 0:ROOT.gSystem.Exit(10)
        if ROOT.gSystem.Load("libAbtDataModel" ) < 0:ROOT.gSystem.Exit(10)
 
        ROOT.gMQ.Create()
        self.keys = keys 
        self.mq = ROOT.gMQ
        self.timer = ROOT.TTimer(1000)
        self._connect( self.timer, "TurnOn()",  self.On ) 
        self._connect( self.timer, "Timeout()", self.Check ) 
        self._connect( self.timer, "TurnOff()", self.Off ) 
        self.obj = None


    def Launch(self):
        self.timer.TurnOn()

    def On(self):
        print "EvMQ.On : starting monitor thread "
        self.mq.StartMonitorThread()

    def Check_(self, key):
        if self.mq.IsUpdated(key):
            obj = self.mq.Get(key, 0)
            if obj:
                print "EvMQ.Check_ finds update in queue %s " % key  
                obj.Print("")
                self.obj = obj
    
    def Check(self):
        #print "EvMQ.Check : looking for updates  "
        if self.mq.IsMonitorRunning():
            for key in self.keys:
                self.Check_(key)

    def Off(self):
        print "EvMQ.Off "

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
    keys = ['default.routingkey','womble']
    emq = EvMQ(keys)
    emq.Launch()
    #ROOT.gSystem.Sleep(1000*10)
    #emq.stop()


    



