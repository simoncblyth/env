# http://root.cern.ch/phpBB2/viewtopic.php?t=6685&highlight=tpydispatcher

import ROOT
import os

class DemoDelegate:
    def __init__(self):
        pass
    def handleQueueUpdated(self):
        print "DemoDelegate.handleQueueUpdated"
    def handleQueueUpdatedIndex(self, index):
        print "DemoDelegate.handleQueueUpdatedIndex", index

def handleQueueUpdatedIndex(index):
    print "Demo callback handleQueueUpdatedIndex ", index



class pMQ(dict):
	"""
	     Creates the message queue singleton ROOT.gMQ and wires up notifications
	     from it via ROOT signals to user supplied callback function or delegate object
	     methods. 
	
	     Configuration of the message queue and server to connect to is done 
	     using the private library which is implemented in C (using glib), 
	     avoiding multi-language duplication.
	     This reads config data from the file pointed to by the ENV_PRIVATE_PATH envvar
	
	     SEE evmq.py for alternative that avoids the polling loop using timers 
	
	"""
    def __init__(self, *args ):
        path = os.path.abspath( os.path.join( os.path.dirname( __file__ ), "lib" , "librootmq.%s" %  ROOT.gSystem.GetSoExt()  ) ) 
        ROOT.gSystem.Load( path )
        ROOT.gMQ.Create(*args)

    def _connect_obj( self, obj , signal , slot , dispatchName ):
        handlerName = "_%s" % slot.__name__
        setattr( obj , handlerName , ROOT.TPyDispatcher( slot ) )
        ROOT.gMQ.Connect( signal , "TPyDispatcher", getattr( obj , handlerName ) , dispatchName ) 

    def _connect_fnc(self,   fnc , signal , dispatchName ):
        self[fnc.__name__] = ROOT.TPyDispatcher( fnc )
        ROOT.gMQ.Connect( signal , "TPyDispatcher", self[fnc.__name__] , dispatchName ) 

    def setCallback( self, fnc ):
        self._connect_fnc( fnc , "QueueUpdatedIndex(Long_t)" , "Dispatch(Long_t)" )

    def setDelegate( self , obj ):
        """
            The delegate must implement all the slots, TPyDispatcher 
            wrapped versions of these are dynamically added to the delegate object
            When signals are emitted by gMQ they are dispatched onwards to the handlers

        """
        self._connect_obj( obj , "QueueUpdated()" ,            obj.handleQueueUpdated      , "Dispatch()" )
        self._connect_obj( obj , "QueueUpdatedIndex(Long_t)" , obj.handleQueueUpdatedIndex , "Dispatch(Long_t)" )
        
    def idle(self, t=1000 ):
        while(True):
            ROOT.gSystem.Sleep(t)
            ROOT.gSystem.ProcessEvents()



if __name__=='__main__':
    from env.rootmq.pmq import pMQ
    q = pMQ(True)
    q.setCallback( handleQueueUpdatedIndex )
    d = DemoDelegate()
    q.setDelegate( d )

    q.idle()     ##  observing segmentation faults from ipython without the sleeping 




