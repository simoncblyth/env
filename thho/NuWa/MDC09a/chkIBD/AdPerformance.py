#!/usr/bin/env python
#
# to check the quality of the IBD15 MC and to produce some basic distributions.
# --- Wei, Apr 2009

__all__ = ['plotIbdBasics']
from GaudiPython.GaudiAlgs import GaudiAlgo
from GaudiPython import SUCCESS, FAILURE
from GaudiPython import gbl

from DybPython.Util import irange
from GaudiKernel import SystemOfUnits as units
import PyCintex
import re
import ROOT

TH1F = gbl.TH1F
TH1D = gbl.TH1D
TH2F = gbl.TH2F
TH2D = gbl.TH2D
TH3F = gbl.TH3F
TH3D = gbl.TH3D

class plotIbdBasics(GaudiAlgo):

    def __init__(self,name):
        GaudiAlgo.__init__(self,name)

        return
    
    def initialize(self):
        print "Initializing the IBD basic ploter", self.name()
        status = GaudiAlgo.initialize(self)
        if status.isFailure(): return status

        self.target_de_name = '/dd/Structure/AD/db-ade1/db-sst1/db-oil1'
        self.gds_de_name = '/dd/Structure/AD/db-gds1'
        self.lso_de_name = '/dd/Structure/AD/db-lso1'

        # What services do you need?
        self.coorSvc = self.svc('ICoordSysSvc', 'CoordSysSvc')
        if not self.coorSvc:
            print 'Failed to get CoordSysSvc'
            return FAILURE
        
        self.histSvc = self.svc('ITHistSvc', 'THistSvc')
        
        self.hist = {}
        
        # output file
        self.outputstat = open('IbdStat.txt', 'w')
        
        # Counters
        self.nInGdLS = 0
        self.nInIAV = 0
        self.nInLSO = 0
        self.nInOAV = 0
        self.nInMO = 0
        
        self.nCap = 0
        self.nGen = 0
        
        self.SpillIn_onGd = 0
        self.IavSpillIn_onGd = 0
        self.SpillOut = 0        
        self.SpillIn_onH = 0
        self.OavSpillIn_onH = 0
        self.SpillOut_onH = 0
        
        self.onGdCap = 0           # total on-Gd cap
        self.onGdGdsCap = 0        # total on-Gd cap in GDS
        
        self.onGdCapPassCut = 0    # total on-Gd cap pass Gd cut
        self.onGdGdsCapPassCut = 0 # total on-Gd cap in GDS pass Gd cut
        self.nPassGdCut = 0        # total pass Gd cut

        self.onGdCapCutUncertain = 0     
        self.nGdCutUncertain = 0
        self.onGdGdsCapCutUncertain = 0
        
        self.nonGdPassGdCut = 0

        self.onHCap = 0
        self.onHCapWithinLS = 0
        self.onHCapWithinLSPassCut = 0
        self.onHCapWithinLSCutUncertainL = 0
        self.onHCapWithinLSCutUncertainU = 0

        self.nonHCapWithinLSPassHCut = 0
        self.nonHCapWithinLSHCutUncertainL = 0
        self.nonHCapWithinLSHCutUncertainU = 0

        self.onHCapBeyondLS = 0
        self.onHCapBeyondLSPassCut = 0
        self.onHCapBeyondLSCutUncertainL = 0
        self.onHCapBeyondLSCutUncertainU = 0

        self.nonHCapBeyondLSPassHCut = 0
        self.nonHCapBeyondLSHCutUncertainL = 0
        self.nonHCapBeyondLSHCutUncertainU = 0

        self.onHCapOavMoPassCut = 0
        self.onHCapOavMoCutUncertainL = 0
        self.onHCapOavMoCutUncertainU = 0
        self.onHCapLS = 0
        self.onHCapMO = 0
        self.onHCapOAV = 0
        
        self.onHCapPassCut = 0
        self.onHCapCutUncertainL = 0
        self.onHCapCutUncertainU = 0
        
        self.nPassHCut = 0
        self.nHCutUncertainL = 0
        self.nHCutUncertainU = 0

        self.nGdPassHCut = 0
        self.nGdHCutUncertainL = 0
        self.nGdHCutUncertainU = 0

        self.hist["genRZ"] = TH2F("genRZ", "Generation Vertex R-Z",
                                  100, 0.0, 6.190144, 100, -2.48, 2.48)
        status = self.histSvc.regHist('/file1/basics/genRZ', 
                                      self.hist["genRZ"])
        if status.isFailure(): return status
        
        self.hist["genXY"] = TH2F("genXY", "Generation Vertex X-Y", 
                                  100, -2.48, 2.48, 100, -2.48, 2.48)
        status = self.histSvc.regHist('/file1/basics/genXY', 
                                      self.hist["genXY"])
        if status.isFailure(): return status
        
        self.hist["HitTime"] = TH1F("HitTime", "Hit Time [#mus]",
                                    500, 0.0, 2000)
        status = self.histSvc.regHist('/file1/basics/HitTime', 
                                      self.hist["HitTime"])
        if status.isFailure(): return status

        self.hist["GdCapHitTime"] = TH1F("GdCapHitTime",
                                         "Gd Capture Hit Time [#mus]",
                                         500, 0.0, 2000)
        status = self.histSvc.regHist('/file1/basics/GdCapHitTime', 
                                      self.hist["GdCapHitTime"])
        if status.isFailure(): return status
        
        self.hist["HCapHitTime"] = TH1F("HCapHitTime",
                                        "H Capture Hit Time [#mus]",
                                        500, 0.0, 2000)
        status = self.histSvc.regHist('/file1/basics/HCapHitTime', 
                                      self.hist["HCapHitTime"])
        if status.isFailure(): return status
        
        self.hist["pe_E"] = TH2F("pe_E", "PE vs visible E (e+ within GdLS)",
                                 100, 0, 10, 700, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/pe_E', 
                                      self.hist["pe_E"])
        if status.isFailure(): return status
        
        self.hist["pe_E_inLS"] = TH2F("pe_E_inLS",
                                      "PE vs visible E (e+ within LS)", 
                                      100, 0, 10, 700, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/pe_E_inLS', 
                                      self.hist["pe_E_inLS"])
        if status.isFailure(): return status

        self.hist["pe_E_LS"] = TH2F("pe_E_LS",
                                    "PE vs visible E (e+ in LS)",
                                    100, 0, 10, 700, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/pe_E_LS', 
                                      self.hist["pe_E_LS"])
        if status.isFailure(): return status

        self.hist["pe_E_all"] = TH2F("pe_E_all", "PE vs visible E (all e+)",
                                     100, 0, 10, 700, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/pe_E_all', 
                                      self.hist["pe_E_all"])
        if status.isFailure(): return status
        
        self.hist["nHits_LS"] = TH1F("nHits_LS", 
                                     "nCap Hits in LS",
                                     100, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/nHits_LS', 
                                      self.hist["nHits_LS"])
        if status.isFailure(): return status
        
        self.hist["nHits_MO"] = TH1F("nHits_MO", 
                                     "nCap Hits in MO",
                                     100, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/nHits_MO', 
                                      self.hist["nHits_MO"])
        if status.isFailure(): return status
        
        self.hist["nHits_onGd"] = TH1F("nHits_onGd",
                                       "on-Gd nCap Hits",
                                       100, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/nHits_onGd', 
                                      self.hist["nHits_onGd"])
        if status.isFailure(): return status
        
        self.hist["nHits_onH"] = TH1F("nHits_onH", 
                                      "on-H nCap Hits", 
                                      100, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/nHits_onH', 
                                      self.hist["nHits_onH"])
        if status.isFailure(): return status
        
        self.hist["nHits"] = TH1F("nHits", "Neutron Capture Hits in GdLS",
                                  100, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/nHits', 
                                      self.hist["nHits"])
        if status.isFailure(): return status
        
        self.hist["nHits_all"] = TH1F("nHits_all", "Neutron Capture Hits",
                                  100, 0, 1400)
        status = self.histSvc.regHist('/file1/basics/nHits_all', 
                                      self.hist["nHits_all"])
        if status.isFailure(): return status

        self.hist["nGdCapGenPos"] = TH1F("nGdCapGenPos",
                                         "Neutron Gd Capture Generation Position",
                                         310, 0, 6.190144)
        status = self.histSvc.regHist('/file1/basics/nGdCapGenPos',
                                      self.hist["nGdCapGenPos"])
        if status.isFailure(): return status
        
        self.hist["nGdGdsCapGenPos"] = TH1F("nGdGdsCapGenPos",
                                            "Neutron Gd Capture in Gds Generation Position",
                                            310, 0, 6.190144)
        status = self.histSvc.regHist('/file1/basics/nGdGdsCapGenPos',
                                      self.hist["nGdGdsCapGenPos"])
        if status.isFailure(): return status
        
        self.hist["nGdCapPos"] = TH3F("nGdCapPos",
                                      "Neutron Gd Capture Position",
                                      100, -2.5, 2.5,
                                      100, -2.5, 2.5,
                                      100, -2.5, 2.5)
        status = self.histSvc.regHist('/file1/basics/nGdCapPos',
                                      self.hist["nGdCapPos"])
        if status.isFailure(): return status

        self.hist["nGdCapOilPos"] = TH3F("nGdCapOilPos",
                                         "Neutron Gd Capture beyond GdLS Position",
                                         100, -2.5, 2.5,
                                         100, -2.5, 2.5,
                                         100, -2.5, 2.5)
        status = self.histSvc.regHist('/file1/basics/nGdCapOilPos',
                                      self.hist["nGdCapOilPos"])
        if status.isFailure(): return status

        self.hist["nGdCapOilPos_RZ"] = TH2F("nGdCapOilPos_RZ",
                                            "Gd Cap beyond GdLS R-Z",
                                            100, 0.0, 6.25,
                                            100, -2.5, 2.5)
        status = self.histSvc.regHist('/file1/basics/nGdCapOilPos_RZ',
                                      self.hist["nGdCapOilPos_RZ"])
        if status.isFailure(): return status


        self.hist["nHCapGenPos"] = TH1F("nHCapGenPos",
                                        "Neutron H Capture Generation Position",
                                        310, 0, 6.190144)
        status = self.histSvc.regHist('/file1/basics/nHCapGenPos',
                                      self.hist["nHCapGenPos"])
        if status.isFailure(): return status


        self.hist["nHOilCapGenPos"] = TH1F("nHOilCapGenPos",
                                           "Neutron H Capture Generation Position",
                                           310, 0, 6.190144)
        status = self.histSvc.regHist('/file1/basics/nHOilCapGenPos',
                                      self.hist["nHOilCapGenPos"])
        if status.isFailure(): return status

        self.hist["nHLsoCapGenPos"] = TH1F("nHLsoCapGenPos",
                                           "Neutron H Capture Generation Position",
                                           310, 0, 6.190144)
        status = self.histSvc.regHist('/file1/basics/nHLsoCapGenPos',
                                      self.hist["nHLsoCapGenPos"])
        if status.isFailure(): return status


        self.hist["eDepInGdLS"] = TH1F("eDepInGdLS", "Deposited Energy [MeV]",
                                       20, 0, 20)
        status = self.histSvc.regHist('/file1/basics/eDepInGdLS', 
                                      self.hist["eDepInGdLS"])
        if status.isFailure(): return status

        self.hist["eDepInLS"] = TH1F("eDepInLS", "Deposited Energy [MeV]",
                                       20, 0, 20)
        status = self.histSvc.regHist('/file1/basics/eDepInLS', 
                                      self.hist["eDepInLS"])
        if status.isFailure(): return status

        self.hist["drift_GdLS"] = TH1F("drift_GdLS",
                                      "Neutron Drift Distance in GdLS [cm]",
                                       200, 0, 50)
        status = self.histSvc.regHist('/file1/basics/drift_GdLS', 
                                      self.hist["drift_GdLS"])
        if status.isFailure(): return status

        self.hist["drift_LS"] = TH1F("drift_LS",
                                      "Neutron Drift Distance in LS [cm]",
                                       200, 0, 50)
        status = self.histSvc.regHist('/file1/basics/drift_LS', 
                                      self.hist["drift_LS"])
        if status.isFailure(): return status
        
        self.hist["time_GdLS"] = TH1F("time_GdLS",
                                      "Neutron Capture time in GdLS [#mus]",
                                       400, 0, 400)
        status = self.histSvc.regHist('/file1/basics/time_GdLS', 
                                      self.hist["time_GdLS"])
        if status.isFailure(): return status

        self.hist["time_LS"] = TH1F("time_LS",
                                      "Neutron Capture Time in LS [#mus]",
                                       400, 0, 2000)
        status = self.histSvc.regHist('/file1/basics/time_LS', 
                                      self.hist["time_LS"])
        if status.isFailure(): return status

        return SUCCESS

    def execute(self):
        print "Executing plotIbdBasics", self.name()
        evt = self.evtSvc()
        simhdr = evt['/Event/Sim/SimHeader']
#        print "SimHeader: ", simhdr
        if simhdr == None:
            print "No SimHeader in this ReadOut. Skip."
            return SUCCESS

#        det = self.detSvc(self.target_de_name)
#        det_gds = self.detSvc(self.gds_de_name)
#        det_lso = self.detSvc(self.lso_de_name)

        # Unobservables
        statshdr = simhdr.unobservableStatistics()
        stats = statshdr.stats()

        PID_trk1 = stats["pdgId_Trk1"].sum()
        PID_trk2 = stats["pdgId_Trk2"].sum()

        if PID_trk1 != -11 or PID_trk2 != 2112:
            print "PID of track 1 is", PID_trk1
            print "PID of track 2 is", PID_trk2
            print "Not an IBD event."
            return SUCCESS

        tGen = stats["t_Trk2"].sum()
        xGen = stats["x_Trk2"].sum()
        yGen = stats["y_Trk2"].sum()
        zGen = stats["z_Trk2"].sum()
        
        tCap = stats["tEnd_Trk2"].sum()
        xCap = stats["xEnd_Trk2"].sum()
        yCap = stats["yEnd_Trk2"].sum()
        zCap = stats["zEnd_Trk2"].sum()
        
        # Get underlying DE object
        de = self.getDet(self.target_de_name)
        if not de:
            print 'Failed to get DE',self.target_de_name
            return FAILURE
        
#        de_lso = self.getDet(self.lso_de_name)
#        de_gds = self.getDet(self.gds_de_name)
#        if not de_lso:
#            print 'Failed to get DE',self.lso_de_name
#            return FAILURE        
#        if not de_gds:
#            print 'Failed to get DE',self.gds_de_name
#            return FAILURE
        
        # Get the AD coordinates of the vertexes
        Gaudi = PyCintex.makeNamespace('Gaudi')
        genGlbPoint = Gaudi.XYZPoint(xGen, yGen, zGen)
        capGlbPoint = Gaudi.XYZPoint(xCap, yCap, zCap)
#        point = de.geometry().toGlobal(point)
        genLclPoint = de.geometry().toLocal(genGlbPoint)
        capLclPoint = de.geometry().toLocal(capGlbPoint)
#        genLclPointLso = de_lso.geometry().toLocal(genGlbPoint)
#        capLclPointLso = de_lso.geometry().toLocal(capGlbPoint)
#        genLclPointGds = de_gds.geometry().toLocal(genGlbPoint)
#        capLclPointGds = de_gds.geometry().toLocal(capGlbPoint)
#        print 'In global coordinate [',gpoint.x(),gpoint.y(),gpoint.z(),']'

        ndrift = ROOT.TVector3(xCap-xGen, yCap-yGen, zCap-zGen)
        
        capTime = tCap - tGen
        capDis = ndrift.Mag()

        R2 = genLclPoint.x()/units.meter * genLclPoint.x()/units.meter + \
             genLclPoint.y()/units.meter * genLclPoint.y()/units.meter
        
        R2Cap = capLclPoint.x()/units.meter * capLclPoint.x()/units.meter + \
                capLclPoint.y()/units.meter * capLclPoint.y()/units.meter

        self.hist["genRZ"].Fill(R2, genLclPoint.z()/units.meter)

        self.hist["genXY"].Fill(genLclPoint.x()/units.meter,genLclPoint.y()/units.meter)

        # Find the interesting volumes
        genDE = self.coorSvc.coordSysDE(genGlbPoint)
        capDE = self.coorSvc.coordSysDE(capGlbPoint)
        if not genDE:
            print 'Failed to find coordinate system DE for generation'\
                '[',genGlbPoint.x(),genGlbPoint.y(),genGlbPoint.z(),']'
            print 'Local: [',genLclPoint.x(),genLclPoint.y(),genLclPoint.z(),']'
            return FAILURE
        else:
            self.nGen += 1
            gendmvol = genDE.geometry().belongsToPath(genGlbPoint,-1)

        if not capDE:
            print 'Failed to find coordinate system DE for capture'\
                '[',capGlbPoint.x(),capGlbPoint.y(),capGlbPoint.z(),']'
            return FAILURE
        else:
            self.nCap += 1
            capdmvol = capDE.geometry().belongsToPath(capGlbPoint,-1)

        genDM = re.split('/', gendmvol).pop()
        capDM = re.split('/', capdmvol).pop()
        print "Generated in ", genDM
        print "Captured in ", capDM
        
        positronHits = 0
        neutronHits = 0
        positronTimeCut = 500.

        positronCut = 126.5  # 1 MeV
        positronCutL = 124.0 # 1 MeV
        positronCutR = 129.0 # 1 MeV

        onHLowerCut = 194.   # 1.5 MeV
        onHLowerCutL = 190.1 # 194 (1 - 2%)
        onHLowerCutR = 197.9 # 194 (1 + 2%)
                
        onHUpperCut = 453.  # 3.5 MeV
        onHUpperCutL = 443.9 # 453 (1 - 2%)
        onHUpperCutR = 462.1 # 453 (1 + 2%)

        onGdCut = 811.9  # 6MeV
        onGdCutL = 803.8 # 811.9 (1 - 1%)
        onGdCutR = 820.0 # 811.9 (1 + 1%)

        # Visible energy
        vis_trk1 = 1.022 + stats['ke_Trk1'].sum()/units.MeV
        self.hist["pe_E_all"].Fill(vis_trk1, positronHits)

        # Capture target
        capTarget = stats["capTarget"].sum()
        print "The capture target is ", capTarget

        # Deposit energy
        eDepInGdLS = stats["EDepInGdLS"].sum()
        self.hist["eDepInGdLS"].Fill(eDepInGdLS/units.MeV)
        eDepInLS = stats["EDepInLS"].sum()
        self.hist["eDepInLS"].Fill(eDepInLS/units.MeV)
        
        simhits = simhdr.hits()
        for detector in simhits.hitDetectors():
            hitCollection = simhits.hitsByDetector(detector)
            if hitCollection == None:
                print "No hits in ", detector
            hits = hitCollection.collection()
            for hit in hits:
#                print " PMT", hit.sensDetId(), "hit @ time: ", hit.hitTime()
                self.hist["HitTime"].Fill(hit.hitTime()/units.microsecond)
                if capTarget == 1:
                    self.hist["HCapHitTime"].Fill(hit.hitTime()/units.microsecond)
                if capTarget == 64:
                    self.hist["GdCapHitTime"].Fill(hit.hitTime()/units.microsecond)
                if hit.hitTime()/units.nanosecond<positronTimeCut and \
                        hit.hitTime()/units.nanosecond<capTime/units.nanosecond:
                    positronHits += 1
                else:
                    neutronHits += 1

        self.hist["nHits_all"].Fill(neutronHits)

        if genDM == 'db-gds1':
            self.nInGdLS += 1
            self.hist["pe_E"].Fill(vis_trk1, positronHits)
            if capDM == 'db-gds1':
                self.hist["nHits"].Fill(neutronHits)
                self.hist["time_GdLS"].Fill(capTime/units.microsecond)
                self.hist["drift_GdLS"].Fill(capDis/units.cm)
            else:
                self.SpillOut += 1
                
        if genDM == 'db-iav1':
            self.nInIAV += 1
            if capTarget == 64:
                self.IavSpillIn_onGd += 1

        if genDM == 'db-lso1':
            self.nInLSO += 1
            self.hist["pe_E_LS"].Fill(vis_trk1, positronHits)
            if capDM == 'db-lso1':
                self.hist["time_LS"].Fill(capTime/units.microsecond)
                self.hist["drift_LS"].Fill(capDis/units.cm)
            if capDM == 'db-oil1' or capDM == 'db-oav1':
                self.SpillOut_onH += 1
            if capTarget == 64:
                self.SpillIn_onGd += 1

        if genDM == 'db-oav1':
            self.nInOAV += 1
            if re.search('db-lso1', capdmvol) and capTarget == 1:
                self.OavSpillIn_onH += 1
                if neutronHits > onHLowerCut and neutronHits < onHUpperCut:
                    self.onHCapOavMoPassCut += 1
                if neutronHits > onHLowerCutL and neutronHits < onHLowerCutR:
                    self.onHCapOavMoCutUncertainL += 1
                if neutronHits > onHUpperCutL and neutronHits < onHUpperCutR:
                    self.onHCapOavMoCutUncertainU += 1

        if genDM == 'db-oil1':
            self.nInMO  += 1
            if re.search('db-lso1', capdmvol) and capTarget == 1:
                self.SpillIn_onH += 1
                if neutronHits > onHLowerCut and neutronHits < onHUpperCut:
                    self.onHCapOavMoPassCut += 1
                if neutronHits > onHLowerCutL and neutronHits < onHLowerCutR:
                    self.onHCapOavMoCutUncertainL += 1
                if neutronHits > onHUpperCutL and neutronHits < onHUpperCutR:
                    self.onHCapOavMoCutUncertainU += 1

        if re.search('db-lso1',gendmvol):
            self.hist["pe_E_inLS"].Fill(vis_trk1, positronHits)

        if re.search('db-lso1',capdmvol):
            if capTarget == 1:
                self.onHCapWithinLS += 1
                if neutronHits > onHLowerCut and neutronHits < onHUpperCut:
                    self.onHCapWithinLSPassCut += 1
                if neutronHits > onHLowerCutL and neutronHits < onHLowerCutR:
                    self.onHCapWithinLSCutUncertainL += 1
                if neutronHits > onHUpperCutL and neutronHits < onHUpperCutR:
                    self.onHCapWithinLSCutUncertainU += 1
            else:
                if neutronHits > onHLowerCut and neutronHits < onHUpperCut:
                    self.nonHCapWithinLSPassHCut += 1
                if neutronHits > onHLowerCutL and neutronHits < onHLowerCutR:
                    self.nonHCapWithinLSHCutUncertainL += 1
                if neutronHits > onHUpperCutL and neutronHits < onHUpperCutR:
                    self.nonHCapWithinLSHCutUncertainU += 1
                
        else:
            if capTarget == 1:
                self.onHCapBeyondLS += 1
                if neutronHits > onHLowerCut and neutronHits < onHUpperCut:
                    self.onHCapBeyondLSPassCut += 1
                if neutronHits > onHLowerCutL and neutronHits < onHLowerCutR:
                    self.onHCapBeyondLSCutUncertainL += 1
                if neutronHits > onHUpperCutL and neutronHits < onHUpperCutR:
                    self.onHCapBeyondLSCutUncertainU += 1
            else:
                if neutronHits > onHLowerCut and neutronHits < onHUpperCut:
                    self.nonHCapBeyondLSPassHCut += 1
                if neutronHits > onHLowerCutL and neutronHits < onHLowerCutR:
                    self.nonHCapBeyondLSHCutUncertainL += 1
                if neutronHits > onHUpperCutL and neutronHits < onHUpperCutR:
                    self.nonHCapBeyondLSHCutUncertainU += 1
            
        # Passing on-Gd neutron cuts
        if neutronHits > onGdCut:
            self.nPassGdCut += 1
            if capTarget != 64:
                print "non-Gd capture pass Gd cut: ", capTarget
                self.nonGdPassGdCut += 1
                value = ('Non-Gd capTarget: ', capTarget)
                theline = str(value)
                self.outputstat.write(theline)
                self.outputstat.write('\n')
                
        if neutronHits > onGdCutL and neutronHits < onGdCutR:
            self.nGdCutUncertain += 1

        # Passing on-H neutron cuts
        if neutronHits > onHLowerCut and neutronHits < onHUpperCut:
            self.nPassHCut += 1
            if capTarget == 64:
                self.nGdPassHCut += 1
        if neutronHits > onHLowerCutL and neutronHits < onHLowerCutR:
            self.nHCutUncertainL += 1
            if capTarget == 64:
                self.nGdHCutUncertainL += 1                
        if neutronHits > onHUpperCutL and neutronHits < onHUpperCutR:
            self.nHCutUncertainU += 1
            if capTarget == 64:
                self.nGdHCutUncertainU += 1
        
        if capTarget == 64:
            self.onGdCap += 1
            self.hist["nHits_onGd"].Fill(neutronHits)
            self.hist["nGdCapPos"].Fill(capLclPoint.x()/units.meter,capLclPoint.y()/units.meter,capLclPoint.z()/units.meter)            
            if capDM == 'db-gds1':
                self.onGdGdsCap += 1
                if neutronHits > onGdCut:
                    self.onGdGdsCapPassCut += 1
                if neutronHits > onGdCutL and neutronHits < onGdCutR:
                    self.onGdGdsCapCutUncertain += 1
            else:
                self.hist["nGdCapOilPos"].Fill(capLclPoint.x()/units.meter,capLclPoint.y()/units.meter,capLclPoint.z()/units.meter)
                self.hist["nGdCapOilPos_RZ"].Fill(R2Cap, capLclPoint.z()/units.meter)

            if genLclPoint.z()/units.m < 1. and genLclPoint.z()/units.m > -1.:
                self.hist["nGdCapGenPos"].Fill(R2)
                if capDM == 'db-gds1':
                    self.hist["nGdGdsCapGenPos"].Fill(R2)
                    
            if neutronHits > onGdCut:
                self.onGdCapPassCut += 1
            if neutronHits > onGdCutL and neutronHits < onGdCutR:
                self.onGdCapCutUncertain += 1
            
        if capTarget == 1:
            self.onHCap += 1
            self.hist["nHits_onH"].Fill(neutronHits)
            
            if genLclPoint.z()/units.m < 1. and genLclPoint.z()/units.m > -1.:
                self.hist["nHCapGenPos"].Fill(R2)
                if capDM == 'db-lso1':
                    self.hist["nHLsoCapGenPos"].Fill(R2)
                if capDM == 'db-oil1' or capDM == 'db-oav1':
                    self.hist["nHOilCapGenPos"].Fill(R2)
                    
            if neutronHits > onHLowerCut and neutronHits < onHUpperCut:
                self.onHCapPassCut += 1
            if neutronHits > onHLowerCutL and neutronHits < onHLowerCutR:
                self.onHCapCutUncertainL += 1
            if neutronHits > onHUpperCutL and neutronHits < onHUpperCutR:
                self.onHCapCutUncertainU += 1

        if capDM == 'db-lso1':
            self.hist["nHits_LS"].Fill(neutronHits)
            if capTarget == 1:
                self.onHCapLS += 1
                
        if capDM == 'db-oav1':
            if capTarget == 1:
                self.onHCapOAV += 1
                
        if capDM == 'db-oil1':
            self.hist["nHits_MO"].Fill(neutronHits)
            if capTarget == 1:
                self.onHCapMO += 1

        return SUCCESS
    
    def finalize(self):
        self.outputstat.close()

        print "Total n generation: ", self.nGen
        print "Total n gen in GdLS: ", self.nInGdLS
        print "Total n gen in IAV: ", self.nInIAV
        print "Total n gen in LSO: ", self.nInLSO
        print "Total n gen in OAV: ", self.nInOAV
        print "Total n gen in MO: ", self.nInMO
        print "Total n capture: ", self.nCap
        print ""
        
        print "Total n-capture on Gd: ", self.onGdCap
        print "Total n-capture on Gd in Gds: ", self.onGdGdsCap
        print "Spill-In on Gd n from LS: ", self.SpillIn_onGd
        print "Spill-In on Gd from IAV: ", self.IavSpillIn_onGd
        print "Spill-Out of n from Gd-LS: ", self.SpillOut
        print ""
        
        print "On Gd capture pass on-Gd cut: ", self.onGdCapPassCut
        print "On Gd capture on-Gd cut uncertainty: ", self.onGdCapCutUncertain
        print "On Gd capture in GDS pass on-Gd cut: ", self.onGdGdsCapPassCut
        print "On Gd capture in GDS on-Gd cut uncertainty: ", self.onGdGdsCapCutUncertain
        print "Total non-Gd capture pass Gd cut: ", self.nonGdPassGdCut
        print "Total pass Gd cut: ", self.nPassGdCut
        print "Overall Gd cut uncertainty: ", self.nGdCutUncertain
        print ""
        
        print "Total n-capture on H: ", self.onHCap
        print "Total n-capture on H in LS: ", self.onHCapLS
        print "Total n-capture on H within LS: ", self.onHCapWithinLS
        print ""
        
        print "Spill-In of n from OIL to LSO: ", self.SpillIn_onH
        print "Spill-In of n from OAV to LSO: ", self.OavSpillIn_onH
        print "Spill-Out of n from LSO to OAV or OIL: ", self.SpillOut_onH
        print ""
        
        print "n capture pass on-H cut: ", self.nPassHCut
        print "n capture pass lower on-H cut uncertainty: ", self.nHCutUncertainL
        print "n capture pass upper on-H cut uncertainty: ", self.nHCutUncertainU
        print ""

        print "On H capture pass on-H cut: ", self.onHCapPassCut
        print "On H capture lower on-H cut uncertainty: ", self.onHCapCutUncertainL
        print "On H capture upper on-H cut uncertainty: ", self.onHCapCutUncertainU
        print ""
        
        print "on H nCap within LS pass H cut: ", self.onHCapWithinLSPassCut
        print "on H nCap within LS pass lower H cut uncertainty: ", self.onHCapWithinLSCutUncertainL
        print "on H nCap within LS pass upper H cut uncertainty: ", self.onHCapWithinLSCutUncertainU
        print ""

        print "non H nCap within LS pass H cut: ", self.nonHCapWithinLSPassHCut
        print "non H nCap within LS pass lower H cut uncertainty: ", self.nonHCapWithinLSHCutUncertainL
        print "non H nCap within LS pass upper H cut uncertainty: ", self.nonHCapWithinLSHCutUncertainU
        print ""

        print "on H nCap beyond LS pass on-H cut: ", self.onHCapBeyondLSPassCut
        print "on H nCap beyond LS pass lower on-H cut uncertainty: ", self.onHCapBeyondLSCutUncertainL
        print "on H nCap bdyond LS pass upper on-H cut uncertainty: ", self.onHCapBeyondLSCutUncertainU
        print ""

        print "non H nCap beyond LS pass H cut: ", self.nonHCapBeyondLSPassHCut
        print "non H nCap beyond LS pass lower H cut uncertainty: ", self.nonHCapBeyondLSHCutUncertainL
        print "non H nCap beyond LS pass upper H cut uncertainty: ", self.nonHCapBeyondLSHCutUncertainU
        print ""

        print "On Gd capture pass on-H cut: ", self.nGdPassHCut
        print "On Gd capture pass lower on-H cut uncertainty: ", self.nGdHCutUncertainL
        print "On Gd capture pass upper on-H cut uncertainty: ", self.nGdHCutUncertainU
        print ""

        print "Total n-capture on H in MO: ", self.onHCapMO
        print "Total n-capture on H in OAV: ", self.onHCapOAV
        print "on H nCap in OAV&MO pass on-H cut: ", self.onHCapOavMoPassCut
        print "on H nCap in OAV&MO pass lower on-H cut uncertainty: ", self.onHCapOavMoCutUncertainL
        print "on H nCap in OAV&MO pass upper on-H cut uncertainty: ", self.onHCapOavMoCutUncertainU
        print ""

        print "Finalizing ", self.name()
        status = GaudiAlgo.finalize(self)
        return status

def configure():
    from DetHelpers.DetHelpersConf import CoordSysSvc
    from GaudiSvc.GaudiSvcConf import THistSvc
#    from GaudiSvc.GaudiSvcConf import DetectorDataSvc

    histsvc = THistSvc()
    histsvc.Output = ["file1 DATAFILE='AdPlots.root' OPT='RECREATE' TYP='ROOT' "]

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
    plotBasics = plotIbdBasics("myBasics")
    app.addAlgorithm(plotBasics)
    pass
