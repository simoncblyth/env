#!/usr/bin/env python


import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

from ROOT import kTRUE, kFALSE, gEve 

import MultiView

from evctrl import EvController
print "imported EvController"
from evgeom  import EvGeom
print "imported EvGeom"
from evdigi  import EvDigi
print "imported EvDigi"
from evtrk import EvTrk
print "imported EvTrk"
from evvrtx import EvVrtx
print "imported EvVrtx"
from evgui   import EvGui
print "imported EvGui"
from evtree  import EvTree
print "imported EvTree"
from evonline import EvOnline
print "imported EvOnline"





class Controller(EvController):
    """
          The hub of Event Display wiring 
              ... pulling together all components
          
          
          Useful attributes to try from ipython commandline :
          
               g.src   EvOnline OR EvTree instance 
                       depending on g.istree() which depends on g.GetSource()
                       
               g.src.mq   
                       when online the local collection of message queue obj updates
             
               g.src.mq.Get("abt.test.event",9)
               g.src.mq.Get("abt.test.runinfo",0)
               g.src.mq.Get("abt.test.string",0)              
                        
                        only a small number of objects are kept in these collections
                        index 0 corresponds to the last received
                        ... so try it later and you will get a different obj 
                        
                        
                        
               g.viewer
                    ROOT.TGLSAViewer object ("GLViewer") 
               g.browser
                    ROOT.TEveBrowser object ("fEveBrowser2") 
               g.browser.GetTabBottom()        
                    ROOT.TGTab object ("fTab25")
                    
                    t = g.browser.GetTabBottom()
                    t.GetNumberOfTabs()
                    
                    t.GetTabTab("AbtNdResponse")
                           ROOT.TGTabElement object 
                           
                    c = t.GetTabTab("Command")
                    
                    
              g.gui.tab
                     dict of tabs
              g.gui.tab['AbtEvent']
                   
                    
        
    """
    def __init__(self, dbg=0 ):
        self.dbg = dbg 
        EvController.__init__(self)
        self.geom = EvGeom()
        if self.dbg>0:
            print "Controller.__init__ g.geom created : %s " % self.geom
        gEve.AddGlobalElement(self.geom.geo)

        self.digi = EvDigi(pmtmin=0, pmtmax=200)   ## range for palette coloring 
        if self.dbg>0:
            print "Controller.__init__ g.digi created : %s" % self.digi
        for elem in self.digi:
            elem.add_()

        self.trk = EvTrk()

	self.vrtx = EvVrtx()

        browser = gEve.GetBrowser()
        self.gui  = EvGui(browser)
        if self.dbg>0:
            print "Controller.__init__ g.gui created"
        self.browser = browser
        self.src = None
        
        
        if hasattr(gEve, 'GetGLViewer'):
            viewer = gEve.GetGLViewer()
        elif hasattr(gEve, 'GetDefaultGLViewer'):
            #print "WARNING : YOU ARE USING AN UN-TESTED ROOT VERSION : SEE #233 " 
            viewer = gEve.GetDefaultGLViewer()
        else:
            viewer = None
        if viewer:    
            viewer.SetCurrentCamera(ROOT.TGLViewer.kCameraPerspXOY)
            viewer.ResetCurrentCamera()
        else:
            print "%s : WARNING : failed to access the GL Viewer " % self.__class__
        self.viewer = viewer
	        
	#use multiview!
    	MultiView.MultiView()

    def istree(self):
        return self.GetSource().endswith(".root")
 
    def propagateTreeChange(self):
        if len(self.src)==0:return False
        if len(self.src)>0:
            self.SetEntryMinMax( 0, len(self.src) - 1 )

    def handleRefreshSource(self):
        if not(self.src):return False
        self.src.refresh() 
        self.propagateTreeChange()
        self.SetEntry(-1)

    def handleChangedSource(self):
        if self.istree():
            self.src = EvTree(self.GetSource())
            self.propagateTreeChange()
        else:
            self.src = EvOnline(self.GetSource()) 
        self.SetEntry(0)

    def handleChangedOther(self):
        if self.dbg>1:
            print "Controller.handleChangedOther "
            ROOT.g_.Print()
        for tmsg in self.src.msgi:
             self.gui.do_msg_display(str(tmsg))

    def handleChangedEntry(self):
        entry = self.GetEntry()
        self.src(entry)                 ## call the source 
        #self.rnd = ROOT.TRandom(entry)  ## no real tracker hits ... so randomize a storm 
        
        esmy = self.src.evt_summary()
        if len(esmy) > 0:
            if self.dbg>1:print "esmy %s" % esmy
            self.gui.update_evt_summary(  smy=esmy )
        
        nsmy = self.src.ndr_summary()
	if len(nsmy) > 0:
            if self.dbg>1:print "update_ndr_summary with nsmy %s" % len(nsmy)
            self.gui.update_ndr_summary(  html=nsmy )
         
        ## no point doing this for every changed entry ... but need rethink to avoid 
        rsmy = self.src.run_summary()
	if len(rsmy) > 0:
            if self.dbg>1:print "rsmy %s" % len(rsmy)
            self.gui.update_run_summary(  html=rsmy )
        
	self.gui.fNumber.SetIntNumber(g_.GetEntry())
        
        pmtr = self.src.pmt_response() 
	if len(pmtr[2]) > 0:
            if self.dbg>1:print "pmtr %s" % pmtr
            self.digi.update_pmt(  pmtr ) 
            
        trkr = self.src.tracker_hits()
        if len(trkr) > 0:
            if self.dbg>1:print "trkr %s" % trkr
            self.geom.update_hits( trkr ) 
      
        fitk = self.src.fitted_track( [150,0,-150] )
        if len(fitk) > 0:
            if self.dbg>1:print "fitk %s " % repr(fitk)
            self.trk.update( fitk )

    	vrtxp = self.src.vertex_position()
        if len(vrtxp) > 0:
            if self.dbg>1:print "vrtxp %s " % repr(vrtxp)
	    self.vrtx.update( vrtxp )

        #tmsg = self.src.msg
        #if tmsg:
        #    self.src.msg = None
        #    self.gui.do_msg_display(tmsg)
       
 
       
        if self.dbg>1:
            print "Controller.handleChangedEntry %s\n%s" % ( entry, self.src.edm()  )
            ROOT.g_.Print()      

        ## this redraw appears to be where the Darwin crash usually occurs 
        ## gEve.Redraw3D(kFALSE, kFALSE )  ## quick update
        #print "redraw3d"

        gEve.Redraw3D()
        #print "redraw3d.done"

    def __call__(self, *args ):
        if len(args)>0:
            self.SetEntry(args[0])
        else:     
            self.NextEntry()

    def __repr__(self):
        return "%s entry %s " % ( self.__class__.__name__ , self.GetEntry() )



if __name__=='__main__':


    ROOT.PyGUIThread.finishSchedule()

    g = Controller()
    from ROOT import g_

    #offline = "$ABERDEEN_HOME/DataModel/sample/run00027.root"
    #offline = "$ABERDEEN_HOME/DataModel/sample/run00027_mc.root"
    offline = "$ABERDEEN_HOME/DataModel/sample/run00027_mc_interim.root"
    online  = dict(lifo=['default.routingkey','abt.test.runinfo','abt.test.event'],fifo=['abt.test.string'] )
    
    g.SetSource( repr(online) )
    #g.SetSource( offline )
    
    
    gEve.Redraw3D(kTRUE, kTRUE )  ## resetCameras and dropLogicals ... defaults are kFALSE

    try:
        __IPYTHON__
    except NameError:
        from IPython.Shell import IPShellEmbed
        irgs = ['']
        banner = "entering ipython embedded shell, within the scope of the abtviz instance... try g? for help or the locals() command "
        ipshell = IPShellEmbed(irgs, banner=banner, exit_msg="exiting ipython" )
        ipshell()

#        ROOT.gSystem.Sleep(100)
#        ROOT.gSystem.ProcessEvents()
           

