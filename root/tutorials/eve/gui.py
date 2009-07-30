"""
      based on tutorials/pyroot/gui_ex.py
      hangs and crashes in gui.txt

"""

import ROOT
eve = True
#eve = False 
if eve:
    ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

def ButtHandler():
    btn = ROOT.BindObject( ROOT.gTQSender, ROOT.TGTextButton )
    print "ButtHandler %s " % btn.WidgetId() 

m = ROOT.TPyDispatcher( ButtHandler )

class NavMainFrame( ROOT.TGMainFrame ):
    def __init__( self, parent, width, height ):
        ROOT.TGMainFrame.__init__( self, parent, width, height )

        self.hf = ROOT.TGHorizontalFrame( self )   

        self.prevb = ROOT.TGTextButton( self.hf, "&Prev", 10 )
        self.prevb.Connect("Clicked()", "TPyDispatcher", m , "Dispatch()")
        self.hf.AddFrame(self.prevb, ROOT.TGLayoutHints() )

        self.nextb = ROOT.TGTextButton( self.hf, "&Next", 20 )
        self.nextb.Connect("Clicked()", "TPyDispatcher", m,  "Dispatch()")
        self.hf.AddFrame(self.nextb, ROOT.TGLayoutHints()  )

        self.AddFrame(self.hf , ROOT.TGLayoutHints() )

        self.SetWindowName("XX GUI")
        #self.SetCleanup(ROOT.kDeepCleanup)  ## buttons do not appear with this 
        self.MapSubwindows()
        self.Resize( self.GetDefaultSize() )
        self.Resize()
        self.MapWindow()

    def __del__(self):
        self.Cleanup()


def make_gui():
    nf = NavMainFrame( ROOT.gClient.GetRoot(), 1000, 600 ) 
    return nf

def e_make_gui():
    from ROOT import gEve, TRootBrowser
    browser = gEve.GetBrowser()
    browser.StartEmbedding(TRootBrowser.kLeft)
    nf = NavMainFrame( ROOT.gClient.GetRoot(), 1000, 600 ) 
    browser.StopEmbedding()
    browser.SetTabTitle("Event Control", 0)


if __name__=='__main__':
    if eve:
        ROOT.PyGUIThread.finishSchedule()
        nf = e_make_gui()
    else:
        nf = make_gui()   


