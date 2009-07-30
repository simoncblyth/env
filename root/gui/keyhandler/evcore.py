import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
ROOT.gROOT.ProcessLine(".L EvManager.cxx+")
from ROOT import EvManager, g_, gEve

def Handler():
    """
        Messages from the C++ side such as
            - load next event 
        are emitted as TQObject signals 
        which are picked up by the dispatcher on the py side 

        gTQSender holds pointer to the last object that sent a signal 
        so the obj should be g_ : huh not very useful ...

        well the usefulness is that it pings the python side
        to do an update ... when the user presses the shortcut key 

    """
    obj = ROOT.BindObject( ROOT.gTQSender, ROOT.EvManager )   ## create object of given type from given address
    print obj
    print "eid %s " % obj.GetEventId()

dispatcher = ROOT.TPyDispatcher( Handler )

if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    EvManager.Create()

    g_.Connect("NextEvent()","TPyDispatcher", dispatcher , "Dispatch()" ) 
    g_.Connect("PrevEvent()","TPyDispatcher", dispatcher , "Dispatch()" ) 
    g_.Connect("LoadEvent()","TPyDispatcher", dispatcher , "Dispatch()" ) 


    tf = gEve.GetBrowser().GetToolbarFrame()

   

