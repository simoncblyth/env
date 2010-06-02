import ROOT
from ROOT import kTRUE, kFALSE
from evdatamodel import EvDataModel

class EvOnline(list):   
    """
      Architectural issue ... 
         tis OK to loose evt objs when monitoring, as when too many are received
         BUT : do not want to loose run objs, how to avoid ? split streams / multiple-keys 
 
      Probably best to configure multiple queues having different routing keys ... 
        mq.Abt.Event
        mq.Abt.RunInfo
        mq.Abt.Text
     
      maybe need to improve rootmq key dynamism to facilitate this 
      (key laid down in the config ? nope seems not )
 
      Need some MQ status display in the GUI to help with this 
      monitoring  the different messages received ... 
    
    
    """
    def __init__(self, key="default.routingkey", dbg=0 ):

        self.status = "online"
        self.key = key
        self.edm = EvDataModel()
        self.dbg = dbg

        ROOT.gSystem.Load("librootmq")
        ROOT.gSystem.Load("libAbtDataModel")
        
        ROOT.gMQ.Create()
        self.mq = ROOT.gMQ
        self.timer = ROOT.TTimer(5000)
        self._connect( self.timer, "TurnOn()",  self.On ) 
        self._connect( self.timer, "Timeout()", self.Check ) 
        self._connect( self.timer, "TurnOff()", self.Off ) 
        
        ROOT.gSystem.Sleep(1000)  ## pause before starting the timer... avoid possible startup issue ?
        self.timer.TurnOn()

    def On(self):
        print "EvMQ.On"
        self.mq.StartMonitorThread()
    def Check(self):
        if self.mq.IsMonitorRunning():
            if self.mq.IsUpdated(self.key):
                if self.dbg>0:print "EvOnline.Check mq %s updated " % self.key  
                obj = self.mq.Get(self.key,0)
                if obj == None:
                    print "EvOnline.Check failed to Get obj "
                    return 
                if self.dbg > 1:
                    obj.Print("")
                if obj.ClassName() == 'AbtEvent':
                    self.edm.set_autoevent( obj )
                elif obj.ClassName() == 'AbtRunInfo':
                    self.edm.set_autorun( obj )
                self.obj = obj 
            else:
                if self.dbg>2:print "EvOnline.Check key \"%s\" no update " % self.key 

    def Off(self):
        print "EvMQ.Off"

    def _connect(self, obj, sign , method ):
        handlerName = "_%s" % method.__name__
        setattr( self , handlerName , ROOT.TPyDispatcher( method ) )
        obj.Connect( sign , "TPyDispatcher", getattr( self , handlerName )  , "Dispatch()" ) 

    def __repr__(self):
        return repr(self.mq)

    def refresh(self):
        print "EvOnline.refresh nop "

    def __call__(self, i):
        pass
        #print "EvOnline.call by handleChangedEntry %d " % i 
        #if self.auto.IsEventUpdated():
        #    self.edm.set_autoevent(self.auto.GetEvent())  

    def pmt_response(self,**kwa):return self.edm.pmt_response(**kwa)
    def tracker_hits(self,**kwa):return self.edm.tracker_hits(**kwa)
    def evt_summary(self,**kwa):return self.edm.evt_summary(**kwa)
    def run_summary(self,**kwa):return self.edm.run_summary(**kwa)
    def ndr_summary(self,**kwa):return self.edm.ndr_summary(**kwa)

    def __getitem__(self, i ):
        return self.edm() 

    def __len__(self):
        return 1

    def __repr__(self):
        return "<Abt %s %s >" % (self.key , self.status )  


if __name__=='__main__':
    ROOT.gSystem.Load("libAbtViz" )
    ROOT.EvModel.Create()
    eo = EvOnline("default.routingkey")
    print eo 
    n = len(eo)
    for i in range(n):
        print eo[i]




