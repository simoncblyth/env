import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    from ROOT import gEve

    tf = gEve.GetBrowser().GetToolbarFrame()


