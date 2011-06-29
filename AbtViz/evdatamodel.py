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
	self.en = None
        
        self.ndrhtml = self.prepare_ndr_summary()
        self.runhtml = self.prepare_run_summary()
	self.trkhtml = self.prepare_trk_summary()
	self.vrthtml = self.prepare_vrt_summary()

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
        tab = self.ndrhtml.GetTable(0)

	#Reset table for every event
	for iq in range(4):	
	    for iv in range(21):
		tab.SetValue(iv,iq,0.0)

        if not (ev):
            #print "update_ndr_summary null evt "
	    return   

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

    def prepare_trk_summary(self):
        smry = ROOT.HtmlSummary("trkhtml")
        row = 6
        tab = smry.AddTable( "Muon Tracker" , 0 , row , kTRUE, kTRUE, kTRUE, kFALSE )
        #tab.Dump()
        tab.SetSideLabel(0, "Theta" )
        tab.SetSideLabel(1, "Phi" )
        tab.SetSideLabel(2, "Fitness" )
        tab.SetSideLabel(3, "Chi Square" )
	tab.SetSideLabel(4, "N Layers" )
	tab.SetSideLabel(5, "N Tracks" )
        return smry

    def update_trk_summary(self, allft):
	tab = self.trkhtml.GetTable(0)
	self.clear_summary(tab, 6)
        if (allft): 
	    #Get best fitted track
	    ft = allft.Get(0) 
            tab.SetSideValue(0,"%.3f &plusmn %.3f"%(ft.Theta(),ft.ThetaError()))
	    tab.SetSideValue(1,"%.3f &plusmn %.3f"%(ft.Phi(),ft.PhiError()))
	    tab.SetSideValue(2,"%.4f"%ft.GetFitness())
	    tab.SetSideValue(3,"%.4f"%ft.GetChisquare())
	    tab.SetSideValue(4,"%i"%ft.GetNFitLayer())
	    tab.SetSideValue(5,"%i"%allft.GetNTrack())

    def prepare_vrt_summary(self):
        smry = ROOT.HtmlSummary("vrthtml")
        row = 5
        tab = smry.AddTable( "Neutron Detector" , 0 , row , kTRUE, kTRUE, kTRUE, kFALSE )
        #tab.Dump()
        tab.SetSideLabel(0, "Position (mm)" )
        tab.SetSideLabel(1, "N Photon" )
	tab.SetSideLabel(2, "Energy (MeV)" )
	tab.SetSideLabel(3, "Mult. Thres." )
	tab.SetSideLabel(4, "Multiplicity" )
        return smry

    def update_vrt_summary(self, vrt, Threshold):
	tab = self.vrthtml.GetTable(0)
	self.clear_summary(tab, 5)
	if (vrt):
            tab.SetSideValue(0,"[%.0f, %.0f, %.0f]"%(vrt[0],vrt[1],vrt[2]))
	    tab.SetSideValue(1,"%.1f"%vrt[3])
	    tab.SetSideValue(2,"%.2f"%self.en.GetEnergy(self.trg.GetVertex().GetCenter()))
	tab.SetSideValue(3,"%.1f Photons"%Threshold)
	if (self.trg.GetNPhoton()): tab.SetSideValue(4,"%i"%int(self.trg.GetNPhoton().GetMultiplicity(Threshold)))

    def clear_summary(self, tab, row):
	for i in range(row):
            tab.SetSideValue(i," ")

    def branch_addresses(self, tree):
        from ROOT import AbtEvent
        evt = AbtEvent() 
        self.trg = evt   ## formerly AbtTriggerEvent
	self.run = tree.GetUserInfo().At(0)
	self.en = tree.GetUserInfo().FindObject("NdEnCalib")
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

    #def evt_summary(self):
        #return [ self.trg.__class__.__name__ , repr(self.trg) ]
    #def run_summary_old(self):
        #return [ self.run.__class__.__name__ , repr(self.run) ]
    
    def run_summary(self):
        self.update_run_summary(self.run)
        self.runhtml.Build()
        return self.runhtml.Html().Data()
     
    def trk_summary(self):
        self.update_trk_summary(self.fitted_track())
        self.trkhtml.Build()
        return self.trkhtml.Html().Data()

    def vrt_summary(self, MultiThres ):
        self.update_vrt_summary(self.vertex(), MultiThres)
        self.vrthtml.Build()
        return self.vrthtml.Html().Data()

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

    def fitted_track(self):
        """
            Provide list of [[x1,y1,z1], [x2,y2,z2], ... ]
            for the [z1,z2,...] arguments 
        """
        if not(self.trg):return [] 

        ft = self.trg.GetTrack()
        if not(ft):return [] 
        #return [[ft.X().At((z+233.7)*10),ft.Y().At((z+233.7)*10),z] for z in zs]
	return ft

    def vertex(self):
    	"""
		Provides the location of the center vertex
	"""
	if not(self.trg):return [] 
	if not(self.trg.GetVertex()):return []
	vp = self.trg.GetVertex().GetCenter()
	if not(vp):return [] 
	return [vp.X(),vp.Y(),vp.Z(),vp.GetNPhoton()]

    def condition(self,TrkCriteria,NDCriteria):
    	"""
		Check to see if conditions are met
	"""

        if not (self.trg): return 0 #FAIL if nothing present

	if (TrkCriteria[0]): #Require Tracker Event
	    if not (self.trg.GetTrackerHit()) and not (self.trg.GetTrack()): return 0 #FAIL if no Fitted Track or Tracker Hits present

	if (TrkCriteria[1]): #Require FitTrack condition
	    if not (self.trg.GetTrack()): return 0
	    if not (self.trg.GetTrack().Get(0)): return 0	
	    if (TrkCriteria[1] > self.trg.GetTrack().Get(0).GetFitness()):return 0

	if (TrkCriteria[2]): #Require Chi Square condition
	    if not (self.trg.GetTrack()): return 0
	    if not (self.trg.GetTrack().Get(0)): return 0	
	    if (TrkCriteria[2] < self.trg.GetTrack().Get(0).GetChisquare()):return 0

	if (TrkCriteria[3]): #Require N Layers condition
	    if not (self.trg.GetTrackerHit()): return 0
	    if not (self.trg.GetTrackerHit().IsNFoldHit(int(TrkCriteria[3]))): return 0	

	if (NDCriteria[0]): #Require Neutron Event
	    if not (self.trg.GetAdc()) and not (self.trg.GetNPhoton): return 0 #FAIL if no ADC data or NPhoton data present

	if (NDCriteria[1]): #Require ADC sum condition
	    if not (self.trg.GetAdc()): return 0	
	    if (NDCriteria[1] > self.trg.GetAdc().GetSum()):return 0 

	if (NDCriteria[2]): #Require Energy condition
	    if not (self.trg.GetVertex()): return 0	
	    if not (self.trg.GetVertex().GetCenter): return 0	
	    if (NDCriteria[2] > self.en.GetEnergy(self.trg.GetVertex().GetCenter())):return 0 

	if (NDCriteria[4]): #Require Multiplicity condition
	    if not (self.trg.GetNPhoton()): return 0		
	    if (NDCriteria[4] > self.trg.GetNPhoton().GetMultiplicity(NDCriteria[3])):return 0

	return 1 #SUCCESS!
        
if __name__=='__main__':
    edm = EvDataModel()

    from aberdeen.DataModel.tests.evs import Evs
    evs = Evs()
    #edm.update_ndr_summary(evs[0])
    #print edm.ndr_summary()
    edm.run = evs.ri
    print edm.run_summary()



