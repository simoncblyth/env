import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
ROOT.gROOT.ProcessLine(".L EvManager.cxx+")
from ROOT import EvManager, g_

if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    from ROOT import gEve
    EvManager.Create()

    tf = gEve.GetBrowser().GetToolbarFrame()

   

