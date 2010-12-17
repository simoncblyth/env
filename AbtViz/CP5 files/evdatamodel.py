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

        if ROOT.gSystem.Load( "libAbtViz" ) < 0:ROOT.gSystem.Exit(10)

        self.trg = None
        self.run = None
        
        self.ndrhtml = self.prepare_ndr_summary()
        self.runhtml = self.prepare_run_summary()

    def prepare_ndr_summary(self):
        smry = ROOT.HtmlSummary("ndrhtml")
        
	#This qtn labels PMT with names Ch00,Ch01,Ch02 ...
	#qtn = ROOT.AbtNdResponse.__qtn__
	
	#This qtn labels PMT with names NW1, NW2, NW3 ...
	qtn = ['NE1','NE2','NE3','NE4','SE1','SE2','SE3','SE4','SW1','SW2','SW3','SW4','NW1','NW2','NW3','NW4','Sum','Mean','MinCh','MaxCh','Overflow']

        tab = smry.AddTable( "AbtNdResponse Adc/Tdc/Nph/Htime" , len(qtn) , 4 , kTRUE, kTRUE, kFALSE, kTRUE )
        #tab.Dump()
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
            
	    #Data might not contain a particular row so need to correct iq (ciq)
	    if q == "Adc":
		ciq = 0
	    elif q == "Tdc":
		ciq = 1
	    elif q == "NPhoton":
		ciq = 2
	    elif q == "HitTime":
		ciq = 3

            anr = getattr( ev , 'Get%s'%q )()
	    if anr:
                for iv,v in enumerate(anr.__qtv__()):
                    tab.SetValue(iv,ciq,v)
                    #print "iv,ciq,v %s %s %s " % ( iv,ciq,v )

    def prepare_run_summary(self):
        smry = ROOT.HtmlSummary("runhtml")
        qtn = ROOT.AbtRunInfo.__qtn__
        tab = smry.AddTable( "AbtRunInfo" , 0 , len(qtn) , kTRUE, kTRUE, kTRUE, kFALSE )
        #tab.Dump()
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
	self.run = tree.GetUserInfo().At(0)
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
            return [self.pmt_response_default() for i in range(4)]
        
	np = self.trg.GetNPhoton() 
        #if not(np):
            #ROOT.Error("pmt_response", "NULL NPhoton" )
        adc = self.trg.GetAdc() 
        #if not(adc):
            #ROOT.Error("pmt_response", "NULL Adc" )
	tdc = self.trg.GetTdc()
	hti = self.trg.GetHitTime()
	
	#To map the correct response of the PMT from data file
	pmtMapping = [11, 15, 3, 7, 10, 14, 2, 6, 9, 13, 1, 5, 8, 12, 0, 4]
        
	#Store all adc, tdc, np, hti and pass back to ev.py
	#In case data is not stored, set response as default
	
	if (adc):
	   adcr = []
	   [adcr.append(adc.GetCh(pmtMapping[i])) for i in range(16)]
	else:
	   adcr = self.pmt_response_default()

	
        if (tdc):
           tdcr = []
           [tdcr.append(tdc.GetCh(pmtMapping[i])) for i in range(16)]
        else:
           tdcr = self.pmt_response_default()

        if (np):
           npr = []
           [npr.append(np.GetCh(pmtMapping[i])) for i in range(16)]
        else:
           npr = self.pmt_response_default()

        if (hti):
           htir = []
           [htir.append(hti.GetCh(pmtMapping[i])) for i in range(16)]
        else:
           htir = self.pmt_response_default()


	return [adcr, tdcr, npr, htir]
    
    def tracker_hits(self, random=None):
        hp = None
        if random:
            hp = ROOT.AbtMtHitPattern()
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

    def fitted_track(self, zs ):
        """
            Provide list of [[x1,y1,z1], [x2,y2,z2], ... ]
            for the [z1,z2,...] arguments 
        """
        if not(self.trg):return [] 
        ft = self.trg.GetTrack()
        if not(ft):return [] 
        return [[ft.X().At((z+233.7)*10),ft.Y().At((z+233.7)*10),z] for z in zs]

    def vertex_position(self):
    	"""
		Provides the location of the center vertex
	"""
	if not(self.trg):return []
	vp = self.trg.GetVertex().GetCenter()
	if not(vp):return [] 
	return [vp.X(),vp.Y(),vp.Z()-295.0,vp.GetNPhoton()]
        
if __name__=='__main__':
    edm = EvDataModel()

    from aberdeen.DataModel.tests.evs import Evs
    evs = Evs()
    #edm.update_ndr_summary(evs[0])
    #print edm.ndr_summary()
    edm.run = evs.ri
    print edm.run_summary()



