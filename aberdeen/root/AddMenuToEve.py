import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

def MenuHandler():
    obj = ROOT.BindObject( ROOT.gTQSender, ROOT.TGPopupMenu )
    if obj.ClassName() == "TGPopupMenu":
        entry = obj.GetCurrent()
        print "entry %s " % entry 
    print "MenuHandler %s " % obj

dispatcher = ROOT.TPyDispatcher( MenuHandler )

def add_menu( frame ):
    """
        Based on $ROOTSYS/gui/gui/src/TRootBrowser.cxx
        which TEveBrowser inherits from 

    """
    fLH1 = ROOT.TGLayoutHints(ROOT.kLHintsTop | ROOT.kLHintsLeft, 0, 4, 0, 0)
    fLH2 = ROOT.TGLayoutHints(ROOT.kLHintsTop | ROOT.kLHintsExpandX, 1, 1, 1, 3)
    
    fMenuBar = ROOT.TGMenuBar( frame , 10, 10, ROOT.kHorizontalFrame)
   
    fNavMenu = ROOT.TGPopupMenu(ROOT.gClient.GetDefaultRoot())
    fNavMenu.AddEntry("&Next", 0)
    fNavMenu.AddEntry("&Prev", 1)
    fNavMenu.AddEntry("&Jump", 2)
    fNavMenu.AddEntry("&Load", 3)
    fNavMenu.Connect("Activated(Int_t)", "TPyDispatcher", dispatcher , "Dispatch()") 

    fMenuBar.AddPopup("&Navigate" , fNavMenu , fLH1 )
    frame.AddFrame(fMenuBar, fLH2)
 
if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    from ROOT import gEve

    b = gEve.GetBrowser()
    fToolbarFrame = b.GetToolbarFrame()
    add_menu( fToolbarFrame )

    b.MapSubwindows()
    b.Resize(b.GetDefaultSize())
    b.MapWindow()
 
"""
    vf = gEve.GetGLViewer().GetFrame()  ; assert vf.ClassName() == 'TGLSAFrame'
    fe = vf.GetList().At(0)             ; assert fe.ClassName() == 'TGFrameElement'
    mb = fe.fFrame                      ; assert mb.ClassName() == 'TGMenuBar'

    f1 = vf.GetClient().GetRoot()       ; assert f1.ClassName() == 'TGFrame' and f1.GetName() == 'fFrame1' 

"""
