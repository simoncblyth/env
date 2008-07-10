"""
    Attempting to bring the Gaudi message svc under python logger control

     assign the below __call__ method as a callback stream buf 
            and set the default stream of the message service to this
            ... this means that all gaudi messages go thru the below 
            call method allowing them to be directed to logfiles or whateve


    CURRENTLY SEGMENTING WITH ABANDON 

"""

import ROOT
ROOT.gSystem.Load("libMathCore")  
import GaudiPython as gp 
import PyCintex as pc



class Log:
    """ bring Gaudi message service under python control for logger integration """
    def __init__(self):
        pass
    
    def __call__(self, msg ):
        print "%s %s" % ( self.__class__.__name__, msg )

    def __del__(self):
        pass
        #self.msv.setDefaultStream(self.ori)


     
if __name__=='__main__':
    
    g = gp.AppMgr()
    g.initialize()  
        
    def echo(msg):print "echo:%s"%msg 

    log = Log()
    
    buf = gp.CallbackStreamBuf(log)
    ost = gp.gbl.ostream(buf)
    msv = g.service("MessageSvc", "IMessageSvc")
    ori = msv.defaultStream()
    
    msv.setDefaultStream(ost)
    
    g.initialize()
        
    msv.setDefaultStream(ori)
    

    
    
#
#msv.reportMessage('TEST',7,'This is a test message')
#
#
#g.initialize()






    