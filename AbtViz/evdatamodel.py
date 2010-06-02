import sys
import ROOT
from ROOT import kTRUE, kFALSE
from aberdeen.DataModel import DataModel

class EvDataModel(DataModel):
    """
        Focus DataModel specifics in this class in order for 
        it to act as buffer against data model changes
    """
    treename = "T"
    def __init__(self):
        super(EvDataModel, self).__init__()  

        if ROOT.gSystem.Load( "libAbtViz" ):
            print "%s : failed to load lib with dynamic path  %s " % ( __file__ , ROOT.gSystem.GetDynamicPath() )
            print "%s : try : LD_LIBRARY_PATH=$(env-libdir) python %s " % ( __file__ , sys.argv[0] )
            #ROOT.gSystem.Exit(13)

        # attempt to avoid deadlocks by creating classes initially in the main thread  
        self.trg = ROOT.AbtEvent()
        self.run = ROOT.AbtRunInfo()
        self.trg = None
        self.run = None

        self.ndrhtml = self.prepare_ndr_summary()
        self.runhtml = self.prepare_run_summary()

    def prepare_ndr_summary(self):
        smry = ROOT.HtmlSummary("ndrhtml")
        qtn = ROOT.AbtNdResponse.__qtn__
        tab = smry.AddTable( "AbtNdResponse Adc/Tdc/Nph/Htime" , len(qtn) , 4 , kTRUE, kTRUE, kFALSE, kTRUE )
        tab.Dump()
        tab.SetSideLabel(0, "Adc" )
        tab.SetSideLabel(1, "Tdc" )
        tab.SetSideLabel(2, "Nph" )
        tab.SetSideLabel(3, "Hti" )
        for i,k in enumerate(qtn):
            tab.SetLabel(i,k)
        return smry

    def update_ndr_summary(self, ev):
        if not(ev):
            print "update_ndr_summary null evt "
	    return   
        tab = self.ndrhtml.GetTable(0)
        for iq,q in enumerate(ROOT.AbtNdResponse.__typ__):
            anr = getattr( ev , 'Get%s'%q )()
            if anr:
                for iv,v in enumerate(anr.__qtv__()):
                    tab.SetValue(iv,iq,v)
                    #print "iv,iq,v %s %s %s " % ( iv,iq,v )

    def prepare_run_summary(self):
        smry = ROOT.HtmlSummary("runhtml")
        qtn = ROOT.AbtRunInfo.__qtn__
        tab = smry.AddTable( "AbtRunInfo" , 0 , len(qtn) , kTRUE, kTRUE, kTRUE, kFALSE )
        tab.Dump()
        for i,l in enumerate(qtn):
            tab.SetSideLabel(i, l)
        #ari.SetLabel(0,"val")
        return smry

    def update_run_summary(self, ri):
        if not(ri):
            print "update_run_summary null run "
	    return   
        tab = self.runhtml.GetTable(0)
        for iv,v in enumerate(ri.__qtv__()):
            #print "iv,v %s %s " % ( iv,v ) 
            tab.SetSideValue(iv,"%s"%v)

    def branch_addresses(self, tree):
        from ROOT import AbtEvent
        evt = AbtEvent() 
        self.trg = evt   ## formerly AbtTriggerEvent
        tree.SetBranchAddress("trigger", ROOT.AddressOf(evt))

    def set_autoevent(self, evt ):
        self.trg = evt 
        ROOT.g_.AutoEventUpdated() 

    def set_autorun(self, run ):
        self.run = run 
        print "set_autorun ... picked up new run object %s " % run 
        ROOT.g_.AutoRunUpdated() 

    def __call__(self):return self.trg

    def version(self, tree):
        if not(tree):
            return "?"
        ri = tree.GetUserInfo().At(0)
        return ri.GetDmVersion()

    def evt_summary(self):
        return [ self.trg.__class__.__name__ , repr(self.trg) ]
    def run_summary_old(self):
        return [ self.run.__class__.__name__ , repr(self.run) ]
    
    def run_summary(self):
        self.update_run_summary(self.run)
        self.runhtml.Build()
        return self.runhtml.Html().Data()
     
    def ndr_summary(self):
        self.update_ndr_summary(self.trg)
        self.ndrhtml.Build()
        return self.ndrhtml.Html().Data()


    def pmt_response_default(self):
        return [0 for i in range(16)]

    def pmt_response(self):
        if not(self.trg):
            ROOT.Error("pmt_response", "NULL trg" )
            return self.pmt_response_default()
        #np = self.trg.GetNPhoton() 
        #if not(np):
        #    ROOT.Error("pmt_response", "NULL NPhoton" )
        #    return self.pmt_response_default()
        adc = self.trg.GetAdc() 
        if not(adc):
            ROOT.Error("pmt_response", "NULL Adc" )
            return self.pmt_response_default()
        return [ adc.GetCh(i) for i in range(16)]

    def tracker_hits(self, random=None):
        hp = None
        if random:
            hp = ROOT.AbtTrackerHitPattern()
            for lay in range(7):
                lhp = hp.Layer(lay)
                ptn = 1 << int(random.Uniform(0, 30))
                lhp.SetPattern(int(ptn))    ## problems with  could not convert argument 1 (long int too large to convert to int)
        else:
            if not(self.trg): 
                ROOT.Error("tracker_hits", "NULL trg" )
                return []
            hp = self.trg.GetTrackerHit()   
        if not(hp):
            #ROOT.Error("tracker_hits", "NULL hit pattern" )
            return [] 
        return [(lay,det) for det in range(32) for lay in range(7) if hp.IsHit(lay,det)]




if __name__=='__main__':
    edm = EvDataModel()

    from aberdeen.DataModel.tests.evs import Evs
    evs = Evs()
    #edm.update_ndr_summary(evs[0])
    #print edm.ndr_summary()
    edm.run = evs.ri
    print edm.run_summary()



