import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
ROOT.gROOT.ProcessLine(".L EvManager.cxx+")
from ROOT import g_ , gEve 

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

class Enum(list):
    def __init__(self, names):
        self.extend( names.split() )
        for value, name in enumerate(self):
            setattr(self, name, value)
    def __call__(self, value ):
        return [name for v, name in enumerate(self) if value == v][0] 


class EvGUI(ROOT.TQObject):
    def __init__(self):
        browser = gEve.GetBrowser()
        fToolbarFrame = browser.GetToolbarFrame()

        self.menu = Enum("kPrev kNext kFirst kLast kAnother")
        self.add_navmenu( fToolbarFrame )

        self.butt = Enum("kPrev kNext kFirst kLast kAnother")
        self.add_buttons( fToolbarFrame )

        browser.MapSubwindows()
        browser.Resize(browser.GetDefaultSize())
        browser.MapWindow()
  
    def add_navmenu(self, frame ):
        """
          Based on $ROOTSYS/gui/gui/src/TRootBrowser.cxx
          which TEveBrowser inherits from 

        """
        fLH1 = ROOT.TGLayoutHints(ROOT.kLHintsTop | ROOT.kLHintsLeft, 0, 4, 0, 0)
        fLH2 = ROOT.TGLayoutHints(ROOT.kLHintsTop | ROOT.kLHintsExpandX, 1, 1, 1, 3)
    
        fMenuBar = ROOT.TGMenuBar( frame , 10, 10, ROOT.kHorizontalFrame)
   
        fNavMenu = ROOT.TGPopupMenu(ROOT.gClient.GetDefaultRoot())
        for value, name in enumerate(self.menu):
            fNavMenu.AddEntry( name[1:] , value )

        self._handleNavMenu  = ROOT.TPyDispatcher( self.handleNavMenu  )
        fNavMenu.Connect("Activated(Int_t)", "TPyDispatcher", self._handleNavMenu , "Dispatch()") 

        fMenuBar.AddPopup("&Navigate" , fNavMenu , fLH1 )
        frame.AddFrame( fMenuBar, fLH2)

    def handleNavMenu(self):
        obj = ROOT.BindObject( ROOT.gTQSender, ROOT.TGPopupMenu )
        eid = obj.GetCurrent().GetEntryId()

        if   eid == self.menu.kNext :  g_.NextEntry()
        elif eid == self.menu.kPrev :  g_.PrevEntry()
        elif eid == self.menu.kFirst:  g_.FirstEntry()
        elif eid == self.menu.kLast :  g_.LastEntry()
        else:
            name = self.menu(eid)
            print "handleNavMenu ?... %s %s " % ( obj , name )



    def add_buttons(self, frame ):
        fLH0 = ROOT.TGLayoutHints( ROOT.kLHintsCenterX , 5, 5, 3, 4)  
        self._handleButtons  = ROOT.TPyDispatcher( self.handleButtons  )
        for value, name in enumerate(self.butt):
            button = ROOT.TGTextButton( frame , name[1:] , value )
            button.Connect("Clicked()", "TPyDispatcher", self._handleButtons , "Dispatch()") 
            frame.AddFrame( button ,  fLH0 )

    def handleButtons(self):
        obj = ROOT.BindObject( ROOT.gTQSender, ROOT.TGTextButton )
        wid = obj.WidgetId()

        if   wid == self.butt.kNext:  g_.NextEntry()
        elif wid == self.butt.kPrev:  g_.PrevEntry()
        elif wid == self.butt.kFirst: g_.FirstEntry()
        elif wid == self.butt.kLast : g_.LastEntry()
        else:
            name = self.butt(wid)
            print "handleButtons ?... %s %s " % ( obj , name )


class EvController(ROOT.TQObject):
    """
        EvController is notified of changes to instrumented properties of the model (EvManager.cxx)

        NB the structure of the Connect hookup is rather prescribed, small changes
           will cause segv
    """
    def __init__(self):
        ROOT.EvManager.Create()

        self._handleChangedEntry  = ROOT.TPyDispatcher( self.handleChangedEntry  )
        g_.Connect("SetEntry(Int_t)", "TPyDispatcher", self._handleChangedEntry  , "Dispatch()" ) 

        self._handleChangedEntryMinMax  = ROOT.TPyDispatcher( self.handleChangedEntryMinMax  )
        g_.Connect("SetEntryMinMax(Int_t,Int_t)", "TPyDispatcher", self._handleChangedEntryMinMax  , "Dispatch()" ) 

        self._handleChangedSource = ROOT.TPyDispatcher( self.handleChangedSource )
        g_.Connect("SetSource(char*)","TPyDispatcher", self._handleChangedSource , "Dispatch()" ) 

    def handleChangedEntry(self):
        sender = ROOT.BindObject( ROOT.gTQSender, ROOT.EvManager ) 
        print "ChangedEntry %s %s " % ( sender.GetEntry() , g_.GetEntry() )

    def handleChangedEntryMinMax(self):
        sender = ROOT.BindObject( ROOT.gTQSender, ROOT.EvManager ) 
        print "ChangedEntryMinMax %s %s " % ( sender.GetEntry() , g_.GetEntry() )
     
    def handleChangedSource(self):
        sender = ROOT.BindObject( ROOT.gTQSender, ROOT.EvManager ) 
        print "ChangedSource %s %s " % ( sender.GetSource() , g_.GetSource() )


if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    ec = EvController()
    eg = EvGUI()

   

