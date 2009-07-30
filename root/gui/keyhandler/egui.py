import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
ROOT.gROOT.ProcessLine(".L KeyHandler.cxx+")
from ROOT import KeyHandler

if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    from ROOT import gEve

    kh = KeyHandler()
    tf = gEve.GetBrowser().GetToolbarFrame()

   

