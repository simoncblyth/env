#!/usr/bin/env python

import os


import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

from ROOT import kTRUE, kFALSE, gEve 

from evctrl import EvController
print "imported EvController"
from evgeom  import EvGeom
print "imported EvGeom"
from evdigi  import EvDigi
print "imported EvDigi"
from evgui   import EvGui
print "imported EvGui"
from evtree  import EvTree
print "imported EvTree"
from evonline import EvOnline
print "imported EvOnline"


class Controller(EvController):
    """
          The hub of Event Display wiring ... pulling together all components
 
    """
    def __init__(self, dbg=0 ):
        self.dbg = dbg 
        EvController.__init__(self)
        self.geom = EvGeom()
        if self.dbg>0:
            print "Controller.__init__ g.geom created : %s " % self.geom
        gEve.AddGlobalElement(self.geom.geo)
        
        self.digi = EvDigi(pmtmin=0, pmtmax=600)   ## range for palette coloring 
        if self.dbg>0:
            print "Controller.__init__ g.digi created : %s" % self.digi
        for elem in self.digi:
            elem.add_()

        browser = gEve.GetBrowser()
        self.gui  = EvGui(browser)
        if self.dbg>0:
            print "Controller.__init__ g.gui created"
        
        self.src = None
        
        if hasattr(gEve, 'GetGLViewer'):
            viewer = gEve.GetGLViewer()
        elif hasattr(gEve, 'GetDefaultGLViewer'):
            print "WARNING : YOU ARE USING AN UN-TESTED ROOT VERSION : SEE #233 " 
            viewer = gEve.GetDefaultGLViewer()
        else:
            viewer = None
        if viewer:    
            viewer.SetCurrentCamera(ROOT.TGLViewer.kCameraPerspXOY)
            viewer.ResetCurrentCamera()
        else:
            print "%s : WARNING : failed to access the GL Viewer " % self.__class__

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
        
        pmtr = self.src.pmt_response() 
        if len(pmtr) > 0:
            if self.dbg>1:print "pmtr %s" % pmtr
            self.digi.update_pmt(  pmtr ) 
            
        trkr = self.src.tracker_hits()
        if len(trkr) > 0:
            if self.dbg>1:print "trkr %s" % trkr
            self.geom.update_hits( trkr ) 
       
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



def in_ipython():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True



if __name__=='__main__':

    try:
        import IPython
        ipython = IPython.Shell.start()
    except:
        pass

    ROOT.PyGUIThread.finishSchedule()

    g = Controller()
    from ROOT import g_

    #g.SetSource("$ABERDEEN_HOME/DataModel/sample/run00027.root")   ## offline running from a file
    g.SetSource("default.routingkey")    ## online running tests

    gEve.Redraw3D(kTRUE, kTRUE )  ## resetCameras and dropLogicals ... defaults are kFALSE

    ## stay alive while gdb python debugging, 
    ## but dont idle like this when using ipython, in order to get to the prompt 
    try:
        __IPYTHON__
        ipython.mainloop()
    except:
        ROOT.gSystem.Sleep(100)
        ROOT.gSystem.ProcessEvents()
           

