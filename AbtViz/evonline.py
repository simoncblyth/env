import ROOT
from ROOT import kTRUE, kFALSE
from evdatamodel import EvDataModel


def mq_pop(key="abt.test.string", n=-1 ):
    """     
       defaults to popping the tail  
       ... n=0 for popping the head             
    """
    while 1:
        obj = ROOT.gMQ.Pop(key,n)
        if not obj:
            break
        yield obj



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
    def __init__(self, src ):

        self.status = "online"
        self.keys = eval(src)    # converts repr of dict of lifo/fifo keys back into dict  
        self.edm = EvDataModel()
        self.msg = None
        self.msgi = []
        self.period = 5000
 
        if ROOT.gSystem.Load("librootmq") < 0:ROOT.gSystem.Exit(10)
        if ROOT.gSystem.Load("libAbtDataModel") < 0:ROOT.gSystem.Exit(10)
        
        ROOT.gMQ.Create(1)      # CONSUMER
        self.dbg = ROOT.gMQ.GetDebug()
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
        for key in self.keys.get('lifo',[]):
            self.Check_lifo(key)
        for key in self.keys.get('fifo',[]):
            self.Check_fifo(key)

    def Check_fifo(self, key):        
        if key == 'abt.test.string':
            self.msgi = mq_pop(key)   ## hmm this is a generator ... does not need to be here .. can do it once only
            accessed = self.mq.GetAccessed(key, 0) 
            if accessed == 0:
                ROOT.g_.SetOther()    ## something fresh ready for popping ....  kick the GUI
        else:
            self.msgi = []
            print "Check_fifo not implemented for key %s " % key 
            
 
    def Check_lifo(self, key):
         """
         
              PROBLEMS WITH THIS ARCHITECTURE...
              
                 1) misses AbtEvent ... thats more of a feature : it is meant to do this
                 2) misses text msgs ...
                       sending multiple messages close together within the pulse window
                       results in some missed messages (different messages get missed
                       by each consumer)  
                                     
                    the messages are all in the local dq ... the monitor thread blocks
                    on updates and places messages in the dq collections based on 
                    routing key (up to a current limit of ~10 msgs in each dq)
                    
                    just need to read them off..  but issue then becomes avoiding duplicates    
                    .... to deal with this add .accessed to the msg

         """
         accessed = self.mq.GetAccessed(key, 0) 
         if accessed == 0:
             if self.dbg>0:print "EvOnline.Check_lifo dq %s accessed %s " % ( key , accessed )  
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
             if self.dbg>4:print "EvOnline.Check_lifo dq \"%s\" accessed %s " % ( key , accessed ) 

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


    def fitted_track(self,*args,**kwa):return self.edm.fitted_track(*args,**kwa)
    def vertex_position(self,*args,**kwa):return self.edm.vertex_position(*args,**kwa)
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




