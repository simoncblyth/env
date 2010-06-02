import ROOT

class EvController(object):
    """
        EvController is notified of changes to instrumented properties of the model (EvModel.cxx)
    """
    def __init__(self):
        """
             the handleChanged* methods here are intended  
             to be overridden in descendant classes like Controller
        """ 
        from evconnectmodel import EvConnectModel 
        self.ecm = EvConnectModel()
        self.ecm( "SetEntry(Int_t)"             , self.handleChangedEntry )
        self.ecm( "SetEntryMinMax(Int_t,Int_t)" , self.handleChangedEntryMinMax )
        self.ecm( "SetSource(char*)"            , self.handleChangedSource )
        self.ecm( "RefreshSource()"             , self.handleRefreshSource )

    def __getattribute__(self, name ):
        """
              controller proxying for the model : methods not implemented in the controller as passed on to the model
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr( ROOT.g_ , name )

    def sender(self):
        return ROOT.BindObject( ROOT.gTQSender, ROOT.EvModel ) 

    def handleChangedEntry(self):       print "ChangedEntry %s %s " %       ( self.sender().GetEntry()  , ROOT.g_.GetEntry() )
    def handleChangedEntryMinMax(self): print "ChangedEntryMinMax %s %s " % ( self.sender().GetEntry()  , ROOT.g_.GetEntry() )
    def handleChangedSource(self):      print "ChangedSource %s %s " %      ( self.sender().GetSource() , ROOT.g_.GetSource() )
    def handleRefreshSource(self):      print "RefreshSource %s %s " %      ( self.sender().GetSource() , ROOT.g_.GetSource() )



if __name__=='__main__':
    g = EvController()

    g.NextEntry()
    g.NextEntry()
    g.NextEntry()
    g.NextEntry()

    g.FirstEntry() 
    g.LastEntry() 

    g.SetEntryMinMax(10,20)
    g.FirstEntry()
    g.NextEntry()
    g.SetSource("hello")
    g.Print()
    g.SetSource("bye")
    g.Print()

    g.SetEntry(100)



