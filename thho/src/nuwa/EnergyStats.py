#!/usr/bin/env python
#
# Example module to make some energy histograms
#
#  Usage:
#   nuwa.py -n -1 UnderstandingEnergy.EnergyStats reconAndSimData.root

# Load DybPython
from DybPython.DybPythonAlg import DybPythonAlg
from GaudiPython import SUCCESS, FAILURE
from GaudiPython import gbl, loaddict
from DybPython.Util import irange

# Make shortcuts to any ROOT classes you want to use
TH1F = gbl.TH1F
TH2F = gbl.TH2F

# Make shortcuts to any ROOT classes you want to use

loaddict("libCLHEPRflx")
loaddict("libHepMCRflx")
Detector = gbl.DayaBay.Detector
AdPmtSensor = gbl.DayaBay.AdPmtSensor
ServiceMode = gbl.ServiceMode
ReconStatus = gbl.ReconStatus

# Make your algorithm
class EnergyStatsAlg(DybPythonAlg):
    "Algorithm to make Energy Statistics file"
    def __init__(self,name):
        DybPythonAlg.__init__(self,name)
        return

    def initialize(self):
        status = DybPythonAlg.initialize(self)
        if status.isFailure(): return status
        self.info("initializing")

        self.cableSvc = self.svc('ICableSvc','StaticCableSvc')
        if self.cableSvc == None:
            self.error("Failed to get StaticCableSvc")
            return FAILURE

        # Make energy histograms
        hist = TH1F("genEnergy","Generated Particle Energy",100,937.0,940.0)
        hist.GetXaxis().SetTitle("Particle Energy [MeV]")
        hist.GetYaxis().SetTitle("Generated Particles")
        hist.SetLineColor(4)
        self.stats["/file0/energy/genEnergy"] = hist

         
        hist = TH1F("genKineticEnergy","Generated Particle Kinetic Energy",
                    100,0.0,10.0)
        hist.GetXaxis().SetTitle("Particle Kinetic Energy [MeV]")
        hist.GetYaxis().SetTitle("Generated Particles")
        hist.SetLineColor(4)
        self.stats["/file0/energy/genKineticEnergy"] = hist

        hist = TH1F("simScintEnergy","Energy deposited in Scintillator",
                    500,0.0,10.0)
        hist.GetXaxis().SetTitle("Ionization Energy [MeV]")
        hist.GetYaxis().SetTitle("Simulated Events")
        hist.SetLineColor(4)
        self.stats["/file0/energy/simScintEnergy"] = hist
        
        hist = TH1F("simQuenchedEnergy","Quenched Energy in Scintillator",
                    500,0.0,10.0)
        hist.GetXaxis().SetTitle("Quenched Ionization Energy [MeV]")
        hist.GetYaxis().SetTitle("Simulated Events")
        hist.SetLineColor(4)
        self.stats["/file0/energy/simQuenchedEnergy"] = hist
        
        hist = TH2F("simQuenching",
                    "Quenching vs. Total Energy in Scintillator",
                    300,0.0,10.0,
                    300,0.0,1.5)
        hist.GetXaxis().SetTitle("Ionization Energy [MeV]")
        hist.GetYaxis().SetTitle("Quenched Energy / Ionization Energy")
        self.stats["/file0/energy/simQuenching"] = hist

        hist = TH1F("simHits","Number of Simulated Hits on PMTs",
                    500,0.0,2000)
        hist.GetXaxis().SetTitle("Simulated Number of Photoelectrons [NPE]")
        hist.GetYaxis().SetTitle("Simulated Events")
        hist.SetLineColor(4)
        self.stats["/file0/energy/simHits"] = hist

        hist = TH2F("simHitsVsQE",
                    "Number of Simulated Hits vs. Quenched Energy",
                    200,0.0,10.0,
                    200,0.0,200)
        hist.GetXaxis().SetTitle("Quenched Energy [MeV]")
        hist.GetYaxis().SetTitle("Photelectrons / Quenched Energy")
        self.stats["/file0/energy/simHitsVsQE"] = hist

        hist = TH2F("adcSumVsSimHits",
                    "Sum of Raw ADC vs. Number of Simulated PMT Hits",
                    200,0.0,2000,
                    200,0.0,1000.0)
        hist.GetXaxis().SetTitle("Simulated Number of Photoelectrons [NPE]")
        hist.GetYaxis().SetTitle("Raw ADC Sum / Photoelectrons")
        self.stats["/file0/energy/adcSumVsSimHits"] = hist
        
        hist = TH1F("calibAdcSum","Sum of Calibrated ADC",500,0.0,2000.)
        hist.GetXaxis().SetTitle("Sum of calibrated ADC values in AD [ADC_SPE]")
        hist.GetYaxis().SetTitle("Simulated Triggered Readouts")
        hist.SetLineColor(4)
        self.stats["/file0/energy/calibAdcSum"] = hist
        
        hist = TH2F("calibAdcSumVsSimHits",
                    "Sum of Calibrated ADC vs. Number of Simulated PMT Hits",
                    200,0.0,2000.0,
                    200,0.0,2.)
        hist.GetXaxis().SetTitle("Simulated Number of Photoelectrons [NPE]")
        hist.GetYaxis().SetTitle("Calibrated ADC Sum / Photoelectrons")
        self.stats["/file0/energy/calibAdcSumVsSimHits"] = hist
        
        hist = TH1F("reconEnergy","Reconstructed Energy",500,0.0,10.0)
        hist.GetXaxis().SetTitle("Reconstructed Visible Energy [E_rec]")
        hist.GetYaxis().SetTitle("Triggered Readouts")
        hist.SetLineColor(4)
        self.stats["/file0/energy/reconEnergy"] = hist

        hist = TH2F("calibAdcSumVsReconEnergy",
                    "Sum of Calibrated ADC vs. Reconstructed Energy",
                    200,0.0,10.0,
                    200,0.0,150.)
        hist.GetXaxis().SetTitle("Reconstructed Visible Energy [E_rec]")
        hist.GetYaxis().SetTitle("Calibrated ADC Sum / Reconstructed Energy")
        self.stats["/file0/energy/calibAdcSumVsReconEnergy"] = hist
        
        hist = TH2F("reconEnergyVsQE","Reconstructed vs. Quenched Energy",
                    200,0.0,10.0,
                    200,0.0,1.5)
        hist.GetXaxis().SetTitle("Quenched Energy [MeV]")
        hist.GetYaxis().SetTitle("Reconstructed Energy / Quenched Energy")
        self.stats["/file0/energy/reconEnergyVsQE"] = hist

        return SUCCESS

    def execute(self):
        self.info("executing")
        
        evt = self.evtSvc()

        # Generated Particle Data
        genHdr = evt["/Event/Gen/GenHeader"]
        if genHdr == None:
            self.error("Failed to get GenHeader")
            return FAILURE
        # Fill particle histograms
        totalGenEnergy = 0
        totalGenKineticEnergy = 0
        genEvt = genHdr.event()
        for vertex in irange(genEvt.vertices_begin(),
                             genEvt.vertices_end()):
            for particle in irange(vertex.particles_out_const_begin(),
                                   vertex.particles_out_const_end()):
                totalGenEnergy += particle.momentum().e()
                totalGenKineticEnergy += (particle.momentum().e()
                                          - particle.momentum().m())
        self.stats["/file0/energy/genEnergy"].Fill(totalGenEnergy)
        self.stats["/file0/energy/genKineticEnergy"].Fill(totalGenKineticEnergy)

        # Simulated Particle Data
        simHdr = evt["/Event/Sim/SimHeader"]
        if simHdr == None:
            self.error("Failed to get SimHeader")
            return FAILURE
        statsHdr = simHdr.unobservableStatistics()
        simScintEnergy = 0
        simQuenchedEnergy = 0
        if statsHdr == None:
            self.warning("No SimStatistics for this event")
        else:    
            simStats = statsHdr.stats()
            simScintEnergy = (simStats["EDepInGdLS"].sum()
                              + simStats["EDepInLS"].sum())
            simQuenchedEnergy = (simStats["QEDepInGdLS"].sum()
                                 + simStats["QEDepInLS"].sum())
            self.stats["/file0/energy/simScintEnergy"].Fill( simScintEnergy )
            self.stats["/file0/energy/simQuenchedEnergy"].Fill(
                                                          simQuenchedEnergy )
            if simScintEnergy > 0:
                self.stats["/file0/energy/simQuenching"].Fill( simScintEnergy,
                                           simQuenchedEnergy/simScintEnergy )
        # Simulated Hits    
        hitCollectionMap = simHdr.hits().hitCollection()
        detector = Detector("DayaBayAD1")
        hitCollection = hitCollectionMap[detector.siteDetPackedData()]
        if hitCollection == None:
            self.info("No Hit Collection for "+detector.detName())
            return SUCCESS
        hits = hitCollectionMap[detector.siteDetPackedData()].collection()
        nSimHits = 0
        for hit in hits:
            pmtId = AdPmtSensor( hit.sensDetId() )
            if pmtId.ring() < 1: continue  # Skip calibration PMTs
            nSimHits += 1
        self.stats["/file0/energy/simHits"].Fill( nSimHits )
        if simQuenchedEnergy > 0:
            self.stats["/file0/energy/simHitsVsQE"].Fill( simQuenchedEnergy,
                                                    nSimHits/simQuenchedEnergy)
                                                       
        # Raw Readout Data
        readoutHdr = evt["/Event/Readout/ReadoutHeader"]
        if readoutHdr == None:
            self.error("Failed to get ReadoutHeader")
            return FAILURE
        readout = readoutHdr.readout()
        if readout == None:
            self.info("No Triggered Readout for this event")
            return SUCCESS
        adcSum = 0
        svcMode = ServiceMode( readoutHdr.context(), 0 )
        for channelPair in readout.channelReadout():
            channel = channelPair.second
            chanId = channel.channelId()
            pmtId = self.cableSvc.adPmtSensor( chanId, svcMode )
            if pmtId.ring() < 1: continue  # Skip calibration PMTs
            #adcSum += channel.peakAdc()
            #adcSum += channel.peakAdc()
            for adcIdx in range( channel.size() ):
                adc = channel.adc( adcIdx )
                pedestal = 0
                if channel.pedestal().size()>0:
                    pedestal = channel.pedestal( adcIdx )
                adcClock = channel.adcCycle( adcIdx )
                adcGain = channel.adcRange( adcIdx )
                self.info("ADC value: "+str(adc)
                          + " (pedestal: "+str( pedestal )+","
                          + " peak cycle: "+str( adcClock )+","
                          + " gain: "+str( adcGain )+")")
                # Add to total ADC sum for this trigger
                if adcGain == 1:
                    adcSum += (adc-pedestal)
                elif adcGain == 2:
                    # Adjust low gain adc to high gain scale
                    adcSum += (adc-pedestal) * 20

            #print "THHHHHHHHHHHHHO"
        if nSimHits > 0:
            self.stats["/file0/energy/adcSumVsSimHits"].Fill( nSimHits,
                                                          adcSum/nSimHits )

        # Calibrated Readout Data
        calibReadoutHdr = evt["/Event/CalibReadout/CalibReadoutHeader"]
        if calibReadoutHdr == None:
            self.error("Failed to get CalibReadoutHeader")
            return FAILURE
        calibReadout = calibReadoutHdr.calibReadout()
        if calibReadout == None:
            self.info("No Calibrate Readout for this event")
            return SUCCESS
        calibAdcSum = 0
        svcMode = ServiceMode( calibReadoutHdr.context(), 0 )
        for channelPair in calibReadout.channelReadout():
            channel = channelPair.second
            #chanId = channel.channelId()
            #pmtId = self.cableSvc.adPmtSensor( chanId, svcMode )
            sensorId = channel.pmtSensorId()
            pmtId = AdPmtSensor( sensorId.fullPackedData() )
            if pmtId.ring() < 1: continue  # Skip calibration PMTs
            #calibAdcSum += channel.peakAdc()
            # Calibrated hit data for this channel
            for hitIdx in range( channel.size() ):
                # Hit time is in units of ns, and is relative to trigger time
                hitTime = channel.time( hitIdx )
                self.info("Hit Time: "+str( hitTime ))
                # Hit charge is in units of photoelectrons
                hitCharge = channel.charge( hitIdx )
                self.info("Hit Charge: "+str( hitCharge ))
                calibAdcSum += hitCharge
        self.stats["/file0/energy/calibAdcSum"].Fill( calibAdcSum )
        if nSimHits > 0:
            self.stats["/file0/energy/calibAdcSumVsSimHits"].Fill(
                                                          nSimHits,
                                                          calibAdcSum/nSimHits
                                                          )

        recHdr = evt["/Event/Rec/RecHeader"]
        if recHdr == None:
            self.error("Failed to get RecHeader")
            return FAILURE
        recResults = recHdr.recResults()
        recTrigger = recResults["AdSimple"]
        if recTrigger.energyStatus() == ReconStatus.kGood:
            reconEnergy = recTrigger.energy()
            self.stats["/file0/energy/reconEnergy"].Fill(reconEnergy)
            self.stats["/file0/energy/calibAdcSumVsReconEnergy"].Fill(
                                                       reconEnergy,
                                                       calibAdcSum/reconEnergy )
            if simQuenchedEnergy > 0:
                self.stats["/file0/energy/reconEnergyVsQE"].Fill(
                                                simQuenchedEnergy,
                                                reconEnergy/simQuenchedEnergy )
            
        return SUCCESS
        
    def finalize(self):
        self.info("finalizing")
        status = DybPythonAlg.finalize(self)
        return status


#####  Job Configuration for nuwa.py ########################################

def configure():
    from StatisticsSvc.StatisticsSvcConf import StatisticsSvc
    statsSvc = StatisticsSvc()
    statsSvc.Output ={"file0":"energyStats.root"}
    import DataSvc
    DataSvc.Configure()
    return

def run(app):
    '''
    Configure and add an algorithm to job
    '''
    app.ExtSvc += ["StaticCableSvc", "StatisticsSvc"]
    energyStatsAlg = EnergyStatsAlg("MyEnergyStats")
    app.addAlgorithm(energyStatsAlg)
    pass

