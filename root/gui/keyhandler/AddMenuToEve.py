import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

def MenuHandler():
    obj = ROOT.BindObject( ROOT.gTQSender, ROOT.TGPopupMenu )
    if obj.ClassName() == "TGPopupMenu":
        entry = obj.GetCurrent()
        print "entry %s " % entry 
    print "MenuHandler %s " % obj
menu_dispatcher = ROOT.TPyDispatcher( MenuHandler )

def ButtHandler():
    obj = ROOT.BindObject( ROOT.gTQSender, ROOT.TGTextButton )
    #if obj.ClassName() == "TGTextButton":
    #    entry = obj.GetCurrent()
    #    print "entry %s " % entry 
    print "ButtHandler %s " % obj
butt_dispatcher = ROOT.TPyDispatcher( ButtHandler )

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
    fNavMenu.Connect("Activated(Int_t)", "TPyDispatcher", menu_dispatcher , "Dispatch()") 

    fMenuBar.AddPopup("&Navigate" , fNavMenu , fLH1 )
    frame.AddFrame(fMenuBar, fLH2)


def add_buttons( frame ):
    fNextButton = ROOT.TGTextButton( frame , "&Next" )
    fNextButton.Connect("Clicked()", "TPyDispatcher", butt_dispatcher , "Dispatch()") 
    fPrevButton = ROOT.TGTextButton( frame , "&Prev" )
    fPrevButton.Connect("Clicked()", "TPyDispatcher", butt_dispatcher , "Dispatch()") 
    f0 = ROOT.TGLayoutHints( ROOT.kLHintsCenterX , 5, 5, 3, 4)  
    frame.AddFrame( fNextButton ,  f0 )
    frame.AddFrame( fPrevButton ,  f0 )


 
if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    from ROOT import gEve

    b = gEve.GetBrowser()
    fToolbarFrame = b.GetToolbarFrame()
    add_menu(    fToolbarFrame )
    add_buttons( fToolbarFrame )

    b.MapSubwindows()
    b.Resize(b.GetDefaultSize())
    b.MapWindow()
 
"""
    vf = gEve.GetGLViewer().GetFrame()  ; assert vf.ClassName() == 'TGLSAFrame'
    fe = vf.GetList().At(0)             ; assert fe.ClassName() == 'TGFrameElement'
    mb = fe.fFrame                      ; assert mb.ClassName() == 'TGMenuBar'

    f1 = vf.GetClient().GetRoot()       ; assert f1.ClassName() == 'TGFrame' and f1.GetName() == 'fFrame1' 

"""
