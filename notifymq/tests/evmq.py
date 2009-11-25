import ROOT

class EvMQ:
    def __init__(self):
        ROOT.gSystem.Load("$ENV_HOME/notifymq/lib/libnotifymq.%s" % ROOT.gSystem.GetSoExt() )
        ROOT.gSystem.Load("$ABERDEEN_HOME/DataModel/lib/libAbtDataModel.%s" % ROOT.gSystem.GetSoExt() )
        ROOT.gMQ.Create()
        self.mq = ROOT.gMQ
        self.timer = ROOT.TTimer(1000)
        self._connect( self.timer, "TurnOn()",  self.On ) 
        self._connect( self.timer, "Timeout()", self.Check ) 
        self._connect( self.timer, "TurnOff()", self.Off ) 
        self.timer.TurnOn()

    def On(self):
        print "EvMQ.On"
        self.mq.StartMonitorThread()
    def Check(self):
        #print "EvMQ.Check"
        if not(self.mq.IsMonitorFinished()):
            if self.mq.IsBytesUpdated():
                obj = self.mq.ConstructObject()
                if obj:obj.Print()
    def Off(self):
        print "EvMQ.Off"

    def _connect(self, obj, sign , method ):
        handlerName = "_%s" % method.__name__
        setattr( self , handlerName , ROOT.TPyDispatcher( method ) )
        obj.Connect( sign , "TPyDispatcher", getattr( self , handlerName )  , "Dispatch()" ) 

    def __repr__(self):
        return repr(self.mq)



if __name__=='__main__':
    from evmq import EvMQ
    emq = EvMQ()
    



