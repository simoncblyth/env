import ROOT
from ROOT import kTRUE, kFALSE
from evdatamodel import EvDataModel

class EvOnline(list):   
    """
      In order to partition obj handling in the MQ the routing key is
      used to control which dq the messages are collected in, avoiding loss of runinfo 
      in a cloud of events
 
        abt.test.event
        abt.test.runinfo
        abt.test.string
        abt.test.other
     
     Intanciation of EvOnline creates a TTimer configured to "Check"
     the mq every period (default 5000ms)
     
     
      TODO:
        MQ status display in the GUI to monitor message counts received ... 
    
    """
    def __init__(self, keys=["abt.test.event"], dbg=0, period=5000 ):

        self.status = "online"
        self.keys = ['default.routingkey','abt.test.string','abt.test.runinfo','abt.test.event','abt.test.other']
        self.edm = EvDataModel()
        self.msg = None
        
        self.dbg = dbg
        self.period = period
 
        if ROOT.gSystem.Load("librootmq") < 0:ROOT.gSystem.Exit(10)
        if ROOT.gSystem.Load("libAbtDataModel") < 0:ROOT.gSystem.Exit(10)
        
        ROOT.gMQ.Create()
        self.mq = ROOT.gMQ
        self.timer = ROOT.TTimer(self.period)
        self._connect( self.timer, "TurnOn()",  self.On ) 
        self._connect( self.timer, "Timeout()", self.Check ) 
        self._connect( self.timer, "TurnOff()", self.Off ) 
        
        ROOT.gSystem.Sleep(1000)  ## pause before starting the timer... avoid possible startup issue ?
        self.timer.TurnOn()

    def On(self):
        print "EvMQ.On"
        self.mq.StartMonitorThread()
        
    def Off(self):
        print "EvMQ.Off"
        self.mq.StopMonitorThread()
   
    def Check(self):
        """
            Called by the TTimer/Timeout every self.period microseconds 
            if there is a new obj in the MQ then get it and pass it on 
            to self.edm based on its class 
        """
        for key in self.keys:
            self.Check_(key)

    def Check_(self, key):
         """
              it will usually be a while before runinfo gets updated ... so 
              if the autorun has not yet been set 
              try exploratory get popping off the top of the dq :

                   self.mq.Get("abt.test.runinfo", 0 )

              following the routing key standardization can now
              also respond to abt.test.string ... an fill in text message view

         """
         if self.mq.IsUpdated(key):
             if self.dbg>0:print "EvOnline.Check dq %s updated " % key  
             obj = self.mq.Get(key,0)
             if obj == None:
                 print "EvOnline.Check failed to Get obj "
                 return 
             if self.dbg > 1:
                 obj.Print("")
             if obj.ClassName() == 'AbtEvent':
                 self.edm.set_autoevent( obj )
             elif obj.ClassName() == 'AbtRunInfo':
                 self.edm.set_autorun( obj )
             elif obj.ClassName() == 'TObjString':
                 self.msg = str(obj) 
             self.obj = obj 
         else:
             if self.dbg>2:print "EvOnline.Check dq \"%s\" no update " % key 

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
        return "<Abt %s %s >" % (self.keys , self.status )  


if __name__=='__main__':
    if ROOT.gSystem.Load("libAbtViz" ) < 0:ROOT.gSystem.Exit(10)
    ROOT.EvModel.Create()
    eo = EvOnline("default.routingkey")
    print eo 
    n = len(eo)
    for i in range(n):
        print eo[i]




