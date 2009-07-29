import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

class ButtHandler:
    def __call__(self):
        btn = ROOT.BindObject( ROOT.gTQSender, ROOT.TGTextButton )
        print "ButtHandler %s " % btn.WidgetId() 
       
m = ROOT.TPyDispatcher( ButtHandler )

## based on tutorials/pyroot/gui_ex.py

class NavFrame( ROOT.TGMainFrame ):
    def __init__( self, parent, width, height ):
        ROOT.TGMainFrame.__init__( self, parent, width, height )

        hf = ROOT.TGHorizontalFrame( self )   
        b = ROOT.TGTextButton( hf, "&Prev", 10 )
        hf.AddFrame(b)
        b.Connect("Clicked()", "TPyDispatcher", m , "Dispatch()")

        b = ROOT.TGTextButton( hf, "&Next", 20 )
        hf.AddFrame(b)
        b.Connect("Clicked()", "TPyDispatcher", m,  "Dispatch()")

        self.AddFrame(hf)
        self.SetWindowName("XX GUI")
        self.SetCleanup(ROOT.kDeepCleanup)

        self.MapSubwindows()
        self.Resize( self.GetDefaultSize() )
        self.MapWindow()

    def __del__(self):
        self.Cleanup()


def make_gui():

    from ROOT import gEve, TRootBrowser
    browser = gEve.GetBrowser()
    browser.StartEmbedding(TRootBrowser.kLeft)

    nf = NavFrame( ROOT.gClient.GetRoot(), 1000, 600 ) 

    browser.StopEmbedding()
    browser.SetTabTitle("Event Control", 0)


if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    make_gui()



