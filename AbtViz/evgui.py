import os

import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

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


class EvGui(ROOT.TQObject):
    """
        TEveBrowser is based on TRootBrowser which has a 3-Tab based layout
            left : Eve, Files
           right : GLViewer
          bottom : Command
    
          http://root.cern.ch/root/html/TRootBrowser.html
          http://root.cern.ch/root/html/TGTab.html
        
        Customize the interface by adding/removing tabs 
        
        use TGHtml for textual representation   
           http://root.cern.ch/root/html/tutorials/gui/calendar.C.html
        links ?
        
        Next steps ...
           * top left corner of new tabs ... display glitch until resized
           * switching to the new tabs exhibits display glitch until resized    
           * how to efficiently update these tabs for each event 
        
        
    
    """
    def __init__(self, br):
        
        if ROOT.gSystem.Load("librootmq") < 0:ROOT.gSystem.Exit(10)
        name = ROOT.MQ.NodeStamp()      
        br.SetWindowName( name )
        
        fToolbarFrame = br.GetToolbarFrame()
        self.keyhandler = ROOT.KeyHandler( br )
       
        self.menu = Enum("kOnline kOffline")
        self.add_navmenu( fToolbarFrame )

        self.butt = Enum("kPrev kNext kFirst kLast kRefresh")
        self.add_buttons( fToolbarFrame )

	self.add_numberEntry( fToolbarFrame )

	self.html = {}
        self.tab = {}
        self.field = {}
        
        self.add_htmltab( br.GetTabLeft() ,   "AbtEvent" )
        self.add_htmltab( br.GetTabLeft() ,   "AbtRunInfo" ) 
        self.add_htmltab( br.GetTabBottom() , "AbtNdResponse" )       
        
        self.add_texttab( br.GetTabBottom() , "AbtText" )
        
        br.Layout()
        br.MapSubwindows()
        br.MapWindow()
        br.Resize(br.GetDefaultSize())   ## avoids display glitch at top left of new tabs
      	
    def update_summary(self, name, smy=None , html=None):
        if not(html):
            assert len(smy) % 2 == 0 
            html = "<dl> " + "\n".join([ "<dt> %s <dd> %s" % ( smy[i],smy[i+1] )  for i in range(0,len(smy),2) ])
        self.update_htmltab( name , html  )

    def update_evt_summary(self, **kwa ):
        return self.update_summary( "AbtEvent", **kwa )
    def update_run_summary(self, **kwa ):
        return self.update_summary( "AbtRunInfo", **kwa )
    def update_ndr_summary(self, **kwa):
        return self.update_summary( "AbtNdResponse", **kwa )

    def update_htmltab(self, name, text ):
        tgh = self.html.get(name, None)
        assert tgh, "Error no htmltab for name %s" % name        
        tgh.Clear()
        tgh.ParseText(text)
        tgh.Layout() 

    def add_htmltab(self, tab , name="Info" ):
        """
          Very easy way to display status text ...    
              ROOT.gEve.GetBrowser().SetStatusText("yo",0)    
              ROOT.gEve.GetBrowser().SetStatusText("yo",1)   
          
          To remove a prior tab, such as the useless command tab .. ipython is much better
             tab.RemoveTab(0)    
              
        """
        ROOT.gEve.GetBrowser().SetStatusText("Online",0)    
        ROOT.gEve.GetBrowser().SetStatusText("",1) 

        it = tab.AddTab( name )
        tab.SetTab( name )     ## select the tab 
        
        ht = ROOT.TGHtml( it , 100, 100 , -1  )  ## window,w,h,id  dimensions dont matter, as expands to fill container frame ? 
        it.AddFrame( ht , ROOT.TGLayoutHints(ROOT.kLHintsExpandX | ROOT.kLHintsExpandY, 5, 5, 2, 2) )  ## hints,padleft,right,top,bottom
        
        self.html[name] = ht
        self.tab[name] = it
        self.update_htmltab( name , "<html><head></head><body><h1>Initial content of %s tab </h1></body></html>" % name )
 
 
    def add_texttab( self, tab , name="Default"):

        it = tab.AddTab( name )
        tab.SetTab( name )
        
        te = ROOT.TGTextEntry( it , "enter message here" )
        it.AddFrame( te , ROOT.TGLayoutHints(ROOT.kLHintsExpandX , 5, 5, 2, 2)) 
        self.msg_entry_dispatch = ROOT.TPyDispatcher( self.do_msg_entry )
        te.Connect(  "ReturnPressed()", "TPyDispatcher", self.msg_entry_dispatch, "Dispatch()" )
        
        tv = ROOT.TGTextView( it )
        it.AddFrame( tv  , ROOT.TGLayoutHints(ROOT.kLHintsExpandX | ROOT.kLHintsExpandY, 5, 5, 2, 2) ) 
        
        self.field["%s_te" % name ] = te
        self.field["%s_tv" % name ] = tv
        self.tab[name] = it 
  
    def do_msg_entry(self):
        te = self.field.get('AbtText_te',None)
        if not(te):return
        txt = te.GetText()
        print "do-msg-entry ... %s " % txt
        ROOT.gMQ.SendAString(txt, "abt.test.string")
        te.SetText("")
  
    def do_msg_display(self, txt):
        tv = self.field.get('AbtText_tv',None)
        if not(tv):return
        tv.AddLine(txt)
        tv.ShowBottom()
        #tvt = tv.GetText()  # TGText 
  
  
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
        from ROOT import g_ 

        if   eid == self.menu.kOnline :
	    online = dict(lifo=['default.routingkey','abt.test.runinfo','abt.test.event'],fifo=['abt.test.string'] )
	    g_.SetSource( repr(online) )
	    g_.RefreshSource()
	    ROOT.gEve.GetBrowser().SetStatusText("Online",0)
	    ROOT.gEve.GetBrowser().SetStatusText("",1)
        elif eid == self.menu.kOffline :
	    #FileBrowser = ROOT.gEve.GetBrowser().MakeFileBrowser()
 	    offline = "$ABERDEEN_HOME/DataModel/sample/run00027_mc_interim.root"
	    g_.SetSource( offline )
	    g_.RefreshSource()
	    ROOT.gEve.GetBrowser().SetStatusText("Offline",0)
	    ROOT.gEve.GetBrowser().SetStatusText(offline,1)        
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
        from ROOT import g_ 

        if   wid == self.butt.kNext:	g_.NextEntry()
        elif wid == self.butt.kPrev:	g_.PrevEntry()
	elif wid == self.butt.kFirst:	g_.FirstEntry()
        elif wid == self.butt.kLast:	g_.LastEntry()
        elif wid == self.butt.kRefresh:	g_.RefreshSource()
        else:
            name = self.butt(wid)
            print "handleButtons ?... %s %s " % ( obj , name )

    def add_numberEntry(self,frame):
	self.fNumber = ROOT.TGNumberEntry( frame, 0, 9,999, ROOT.TGNumberFormat.kNESInteger, ROOT.TGNumberFormat.kNEANonNegative, ROOT.TGNumberFormat.kNELLimitMinMax, 0, 1000000 )
	self._handleNumberEntry = ROOT.TPyDispatcher( self.handleNumberEntry )
	self.fNumber.Connect( "ValueSet(Long_t)", "TPyDispatcher", self._handleNumberEntry, "Dispatch()" )
      	self.fNumber.GetNumberEntry().Connect( "ReturnPressed()", "TPyDispatcher", self._handleNumberEntry, "Dispatch()" )
	frame.AddFrame( self.fNumber, ROOT.TGLayoutHints( ROOT.kLHintsTop | ROOT.kLHintsLeft, 5, 5, 5, 5 ) )

    def handleNumberEntry(self):
	from ROOT import g_	
	g_.SetEntry(self.fNumber.GetNumberEntry().GetIntNumber())


if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()

    from evctrl import EvController
    ec = EvController()
    br = ROOT.gEve.GetBrowser()
    eg = EvGui(br)






