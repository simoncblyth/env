import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
ROOT.gROOT.ProcessLine(".L EvManager.cxx+")
from ROOT import g_ , gEve , TPyDispatcher

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


class EvController(ROOT.TQObject):
    """
        EvController is notified of changes to instrumented properties of the model (EvManager.cxx)

        NB the structure of the Connect hookup is rather prescribed, small changes
           will cause segv
    """
    def __init__(self):
        ROOT.EvManager.Create()

        self._handleChangedEntry  = TPyDispatcher( self.handleChangedEntry  )
        g_.Connect("SetEntry(Int_t)", "TPyDispatcher", self._handleChangedEntry  , "Dispatch()" ) 

        self._handleChangedEntryMinMax  = TPyDispatcher( self.handleChangedEntryMinMax  )
        g_.Connect("SetEntryMinMax(Int_t,Int_t)", "TPyDispatcher", self._handleChangedEntryMinMax  , "Dispatch()" ) 

        self._handleChangedSource = TPyDispatcher( self.handleChangedSource )
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
    tf = gEve.GetBrowser().GetToolbarFrame()

   

