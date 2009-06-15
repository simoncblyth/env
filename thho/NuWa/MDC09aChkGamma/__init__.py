#!/usr/bin/env python
#
# to check the quality of the gamma MC and to produce some basic distributions.
# --- Wei S. Wang, Apr 2009

__all__ = ['plotGammaBasics']
from GaudiPython.GaudiAlgs import GaudiAlgo
from GaudiPython import SUCCESS, FAILURE
from GaudiPython import gbl

from DybPython.Util import irange
from GaudiKernel import SystemOfUnits as units
import ROOT

TH1F = gbl.TH1F
TH1D = gbl.TH1D
TH2F = gbl.TH2F
TH2D = gbl.TH2D

class plotGammaBasics(GaudiAlgo):

    # Debug.
    instance = None

    def __init__(self,name):
        GaudiAlgo.__init__(self,name)
        self.hist = {}
        # Debug.
        plotGammaBasics.instance = self
        return
    
    def initialize(self):
        print "Initializing the gamma basic ploter", self.name()
        status = GaudiAlgo.initialize(self)
        if status.isFailure(): return status
        self.target_de_name = '/dd/Structure/AD/db-ade1/db-sst1/db-oil1'
#        self.target_de_name = '/dd/Structure/AD/db-oil1'

        # What services do you need?
        self.coorSvc = self.svc('ICoordSysSvc', 'CoordSysSvc')
        if not self.coorSvc:
            print 'Failed to get CoordSysSvc'
            return FAILURE

        self.histSvc = self.svc('ITHistSvc', 'THistSvc')

        self.hist["genRZ"] = TH2D("genRZ", "Generation Vertex R-Z", \
                                      100, 0.0, 6.1504, 100, -2.48, 2.48)
        status = self.histSvc.regHist('/file1/basics/genRZ', \
                                          self.hist["genRZ"])
        if status.isFailure(): return status
        
        self.hist["genXY"] = TH2D("genXY", "Generation Vertex X-Y", \
                                      100, -2.48, 2.48, 100, -2.48, 2.48)
        status = self.histSvc.regHist('/file1/basics/genXY', \
                                          self.hist["genXY"])
        if status.isFailure(): return status
        
        self.hist["HitTime"] = TH1F("HitTime", "Hit Time",
                                    100, 0.0, 100)
        status = self.histSvc.regHist('/file1/basics/HitTime',
                                      self.hist["HitTime"])
        if status.isFailure(): return status
        
        self.hist["peGen_GdLS"] = TH1F("peGen_GdLS", "pe of a gamma (in GdLS)",
                                       500, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/peGen_GdLS', 
                                      self.hist["peGen_GdLS"])
        if status.isFailure(): return status
        
        self.hist["peGen_LS"] = TH1F("peGen_LS",
                                     "pe of a gamma(in LS)",
                                     500, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/peGen_LS', 
                                      self.hist["peGen_LS"])
        if status.isFailure(): return status
        
        self.hist["peGen_inLS"] = TH1F("peGen_inLS", 
                                       "pe of a gamma (within LS)", 
                                       500, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/peGen_inLS', 
                                      self.hist["peGen_inLS"])
        if status.isFailure(): return status
        
        self.hist["peGen_MO"] = TH1F("peGen_MO", 
                                     "pe of a gamma (in MO)", 
                                     500, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/peGen_MO', 
                                      self.hist["peGen_MO"])
        if status.isFailure(): return status
        
        self.hist["peGen_all"] = TH1F("peGen_all",
                                      "pe of a gamma (in AD)", 
                                      500, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/peGen_all', 
                                      self.hist["peGen_all"])
        if status.isFailure(): return status
        
        self.hist["peCap_GdLS"] = TH1F("peCap_GdLS", 
                                       "pe of a gamma stop in GdLS", 
                                       500, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/peCap_GdLS', 
                                      self.hist["peCap_GdLS"])
        if status.isFailure(): return status
        
        self.hist["peCap_LS"] = TH1F("peCap_LS", 
                                     "pe of a gamma stop in LS", 
                                     500, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/peCap_LS', 
                                      self.hist["peCap_LS"])
        if status.isFailure(): return status
        
        self.hist["peCap_MO"] = TH1F("peCap_MO",
                                     "pe of a gamma stop in MO",
                                     500, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/peCap_MO', 
                                      self.hist["peCap_MO"])
        if status.isFailure(): return status
        
        self.hist["peGenCap_GdLS"] = TH1F("peGenCap_GdLS", 
                                          "pe of a gamma in AD",
                                          500, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/peGenCap_GdLS', 
                                      self.hist["peGenCap_GdLS"])
        if status.isFailure(): return status
        
        self.hist["eDepInGdLS"] = TH1D("eDepInGdLS", "Deposited Energy [MeV]",
                                       70, 0, 7)
        status = self.histSvc.regHist('/file1/basics/eDepInGdLS', 
                                      self.hist["eDepInGdLS"])
        if status.isFailure(): return status
        
        self.hist["eDepInLS"] = TH1D("eDepInLS", "Deposited Energy [MeV]", 
                                     70, 0, 7)
        status = self.histSvc.regHist('/file1/basics/eDepInLS',
                                      self.hist["eDepInLS"])
        if status.isFailure(): return status
        
        self.hist["eDepInAD"] = TH1D("eDepInAD", "Deposited Energy [MeV]", 
                                     700, 0, 7)
        status = self.histSvc.regHist('/file1/basics/eDepInAD',
                                      self.hist["eDepInAD"])
        if status.isFailure(): return status
        
        self.hist["eInitial"] = TH1D("eInitial", "Intial Energy [MeV]", 
                                     70, 0, 7)
        status = self.histSvc.regHist('/file1/basics/eInitial',
                                      self.hist["eInitial"])
        if status.isFailure(): return status
        
        self.hist["drift_Gamma"] = TH1F("drift_Gamma",
                                       "Gamma Drift Distance [cm]",
                                        250, 0, 250)
        status = self.histSvc.regHist('/file1/basics/drift_Gamma', 
                                      self.hist["drift_Gamma"])
        if status.isFailure(): return status
        
        self.hist["drift_GdLS"] = TH1F("drift_GdLS",
                                       "Gamma Drift Distance in GdLS [cm]",
                                       250, 0, 250)
        status = self.histSvc.regHist('/file1/basics/drift_GdLS', 
                                      self.hist["drift_GdLS"])
        if status.isFailure(): return status
        
        self.hist["drift_LS"] = TH1F("drift_LS",
                                     "Gamma Drift Distance in LS [cm]",
                                     250, 0, 250)
        status = self.histSvc.regHist('/file1/basics/drift_LS', 
                                      self.hist["drift_LS"])
        if status.isFailure(): return status
        
        self.hist["time_GdLS"] = TH1F("time_GdLS",
                                      "Gamma drift time in GdLS [ns]",
                                      40, 0, 20)
        status = self.histSvc.regHist('/file1/basics/time_GdLS', 
                                      self.hist["time_GdLS"])
        if status.isFailure(): return status

        self.hist["time_LS"] = TH1F("time_LS",
                                    "Gamma Capture Time in LS [ns]",
                                    40, 0, 20)
        status = self.histSvc.regHist('/file1/basics/time_LS', 
                                      self.hist["time_LS"])
        if status.isFailure(): return status

        return SUCCESS

    def execute(self):
        print "Executing plotGammaBasics", self.name()
        evt = self.evtSvc()
        simhdr = evt['/Event/Sim/SimHeader']

        #det = self.detSvc(self.target_de_name)

        # Unobservables
        statshdr = simhdr.unobservableStatistics()
        stats = statshdr.stats()
        tGen = stats["t_Trk1"].sum()
        xGen = stats["x_Trk1"].sum()
        yGen = stats["y_Trk1"].sum()
        zGen = stats["z_Trk1"].sum()
        
        tCap = stats["tEnd_Trk1"].sum()
        xCap = stats["xEnd_Trk1"].sum()
        yCap = stats["yEnd_Trk1"].sum()
        zCap = stats["zEnd_Trk1"].sum()
        
        # Get underlying DE object
        de = self.getDet(self.target_de_name)
        if not de:
            print 'Failed to get DE',self.target_de_name
            return FAILURE
        
        # Get the AD coordinates of the vertexes
        import PyCintex
        Gaudi = PyCintex.makeNamespace('Gaudi')
        genGlbPoint = Gaudi.XYZPoint(xGen, yGen, zGen)
        capGlbPoint = Gaudi.XYZPoint(xCap, yCap, zCap)
#        point = de.geometry().toGlobal(point)
        genLclPoint = de.geometry().toLocal(genGlbPoint)
        capLclPoint = de.geometry().toLocal(capGlbPoint)
#        print 'Current point is [',point.x(),point.y(),point.z(),']'
#        print 'In global coordinate [',gpoint.x(),gpoint.y(),gpoint.z(),']'

        ndrift = ROOT.TVector3(xCap-xGen, yCap-yGen, zCap-zGen)
        
        capTime = tCap - tGen
        capDis = ndrift.Mag()

        print 'Generation locations', \
            '[', genGlbPoint.x(), genGlbPoint.y(), genGlbPoint.z(),']', \
            '[', genLclPoint.x()/units.cm, genLclPoint.y()/units.cm, genLclPoint.z()/units.cm,']'
        
        self.hist["genRZ"].Fill(genLclPoint.x()/units.meter * genLclPoint.x()/units.meter + genLclPoint.y()/units.meter * genLclPoint.y()/units.meter, genLclPoint.z()/units.meter)

        self.hist["genXY"].Fill(genLclPoint.x()/units.meter,genLclPoint.y()/units.meter)

        self.hist["drift_Gamma"].Fill(capDis/units.cm)
        
        # Find the interesting volumes
        genDE = self.coorSvc.coordSysDE(genGlbPoint)
        capDE = self.coorSvc.coordSysDE(capGlbPoint)
        if not genDE:
            print 'Failed to find coordinate system DE for generation', \
                '[', genGlbPoint.x(), genGlbPoint.y(), genGlbPoint.z(),']', \
                '[', genLclPoint.x()/units.mm, genLclPoint.y()/units.mm, genLclPoint.z()/units.mm,']'
            return FAILURE
        else:
            gendmvol = genDE.geometry().belongsToPath(genGlbPoint,-1)

        if not capDE:
            print 'Failed to find coordinate system DE for capture'\
                '[',capGlbPoint.x(),capGlbPoint.y(),capGlbPoint.z(),']'
            return FAILURE
        else:
            capdmvol = capDE.geometry().belongsToPath(capGlbPoint,-1)

        print "gendmvol is ", gendmvol
        print "capdmvol is ", capdmvol


        import re
        genDM = re.split('/', gendmvol).pop()
        capDM = re.split('/', capdmvol).pop()
        print "Generated in ", genDM
        print "Captured in ", capDM
        
        pmtHits = 0
        simhits = simhdr.hits()
        for detector in simhits.hitDetectors():
            hitCollection = simhits.hitsByDetector(detector)
            if hitCollection == None:
                print "No hits in ", detector
            hits = hitCollection.collection()
            for hit in hits:
#                print " PMT", hit.sensDetId(), "hit @ time: ", \
#                    hit.hitTime()/units.nanosecond
                self.hist["HitTime"].Fill(hit.hitTime()/units.nanosecond)
                pmtHits += 1

        # Unobservables
        PID_trk1 = stats["pdgId_Trk1"].sum()

        if PID_trk1 != 22:
            print "PID of track 1 is", PID_trk1
            print "Not an gamma event."
            return FAILURE

        self.hist["peGen_all"].Fill(pmtHits)

        if genDM ==  'db-gds1':
            self.hist["peGen_GdLS"].Fill(pmtHits)

        if genDM ==  'db-lso1':
            self.hist["peGen_LS"].Fill(pmtHits)

        if re.search('db-lso1',gendmvol):
            self.hist["peGen_inLS"].Fill(pmtHits)

        if genDM ==  'db-oil1':
            self.hist["peGen_MO"].Fill(pmtHits)

        if capDM == 'db-lso1':
            self.hist["peCap_LS"].Fill(pmtHits)

        if capDM == 'db-oil1':
            self.hist["peCap_MO"].Fill(pmtHits)

        if capDM == 'db-gds1':
            self.hist["peCap_GdLS"].Fill(pmtHits)

        if genDM == 'db-gds1' and capDM == 'db-gds1':
            self.hist["peGenCap_GdLS"].Fill(pmtHits)
            self.hist["time_GdLS"].Fill(capTime/units.nanosecond)
            self.hist["drift_GdLS"].Fill(capDis/units.cm)

        if genDM == 'db-lso1' and capDM == 'db-lso1':
            self.hist["time_LS"].Fill(capTime/units.nanosecond)
            self.hist["drift_LS"].Fill(capDis/units.cm)

        eDepInGdLS = stats["EDepInGdLS"].sum()
        eDepInLS = stats["EDepInLS"].sum()
        self.hist["eDepInGdLS"].Fill(eDepInGdLS/units.MeV)
        self.hist["eDepInLS"].Fill(eDepInLS/units.MeV)

        self.hist["eDepInAD"].Fill((eDepInLS+eDepInGdLS)/units.MeV)

	if eDepInLS+eDepInGdLS > 6: print "Accumulative: ", str(eDepInLS+eDepInGdLS)

        eInitial = stats["e_Trk1"].sum()
        self.hist["eInitial"].Fill(eInitial/units.MeV)

        return SUCCESS
    
    def finalize(self):
        print "Finalizing ", self.name()
        status = GaudiAlgo.finalize(self)
        return status

def configure():
    from DetHelpers.DetHelpersConf import CoordSysSvc
    from GaudiSvc.GaudiSvcConf import THistSvc
#    from GaudiSvc.GaudiSvcConf import DetectorDataSvc

    histsvc = THistSvc()
    histsvc.Output = ["file1 DATAFILE='GammaBasicPlots.root' OPT='RECREATE' TYP='ROOT' "]

    coorSvc = CoordSysSvc()
    coorSvc.OutputLevel = 1

    from Gaudi.Configuration import ApplicationMgr
    theApp = ApplicationMgr()
    theApp.ExtSvc.append(coorSvc)

    return

def run(app):
    '''Configure and add this algorithm to job'''
#    from Gaudi.Configuration import ApplicationMgr
#    app = ApplicationMgr()
    app.ExtSvc += ["THistSvc"]
    plotBasics = plotGammaBasics("myBasics")
    app.addAlgorithm(plotBasics)
    pass
