#!/usr/bin/env python
#
# Example module to make some energy histograms
#
#  Usage:
#   nuwa.py -n -1 thhoGenHist GeneratedFile.root

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


        hist = TH1F("genEachEnergy","Generated Particles Kinetic Energy, Each",
                    100,0.0,10.0)
        hist.GetXaxis().SetTitle("Particle Kinetic Energy [MeV]")
        hist.GetYaxis().SetTitle("Generated Particles")
        hist.SetLineColor(4)
        self.stats["/file0/energy/genEachEnergy"] = hist


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
        #print "genEvt.vertices_begin() is ", genEvt.vertices_begin()
        #print "genEvt.vertices_end() is ", genEvt.vertices_end()
        #thhoN = 0
        for vertex in irange(genEvt.vertices_begin(),
                             genEvt.vertices_end()):
            #thhoN +=1
            #print "thhoN is ", thhoN
            for particle in irange(vertex.particles_out_const_begin(),
                                   vertex.particles_out_const_end()):
                #print "vertex.particles_out_const_begin() is", vertex.particles_out_const_begin()
                #print "vertex.particles_out_const_end() is", vertex.particles_out_const_end()
                totalGenEnergy += particle.momentum().e()
                totalGenKineticEnergy += (particle.momentum().e()
                                          - particle.momentum().m())
                #print "totalGenEnergy is ", totalGenEnergy
                #print "totalGenKineticEnergy is ", totalGenKineticEnergy 
        #self.stats["/file0/energy/genEnergy"].Fill(totalGenEnergy)
        #self.stats["/file0/energy/genKineticEnergy"].Fill(totalGenKineticEnergy)
                self.stats["/file0/energy/genEachEnergy"].Fill(particle.momentum().e())

        return SUCCESS
       
    def finalize(self):
        self.info("finalizing")
        status = DybPythonAlg.finalize(self)
        return status


#####  Job Configuration for nuwa.py ########################################

def configure():
    from StatisticsSvc.StatisticsSvcConf import StatisticsSvc
    statsSvc = StatisticsSvc()
    statsSvc.Output ={"file0":"thhoGenHists.root"}
    import DataSvc
    DataSvc.Configure()
    return

def run(app):
    '''
    Configure and add an algorithm to job
    '''
    #app.ExtSvc += ["StaticCableSvc", "StatisticsSvc"]
    app.ExtSvc += ["StatisticsSvc"]
    energyStatsAlg = EnergyStatsAlg("MyEnergyStats")
    app.addAlgorithm(energyStatsAlg)
    pass

