import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
ROOT.gROOT.ProcessLine(".L EvManager.cxx+")
from ROOT import EvManager, g_

"""
   How to get messages from the C++ side ... like load next event 
   onto the py side ?


"""
def Handler():
   btn = ROOT.BindObject( ROOT.gTQSender, ROOT.TGTextButton )

m = ROOT.TPyDispatcher( Handler )

if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    from ROOT import gEve
    EvManager.Create()

    tf = gEve.GetBrowser().GetToolbarFrame()

   

