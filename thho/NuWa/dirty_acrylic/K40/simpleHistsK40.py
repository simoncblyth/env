#!/usr/bin/env python
#
# Example module to make some energy histograms
#
#  Usage:
#   nuwa.py -n -1 hist GeneratedFile.root

# Load DybPython
from DybPython.DybPythonAlg import DybPythonAlg
from GaudiPython import SUCCESS, FAILURE
from GaudiPython import gbl, loaddict
from DybPython.Util import irange

import ROOT
from GaudiKernel import SystemOfUnits as units


# Make shortcuts to any ROOT classes you want to use
TH1F = gbl.TH1F
TH2F = gbl.TH2F
TH2D = gbl.TH2D

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

        self.target_de_name = '/dd/Structure/AD/db-ade1/db-sst1/db-oil1'

        hist = TH1F("genEachEnergy","Generated Particles Total Energy, Each",
                    100,0.0,2.0)
        hist.GetXaxis().SetTitle("Particle Total Energy [MeV]")
        hist.GetYaxis().SetTitle("Generated Particles")
        hist.SetLineColor(4)
        self.stats["/file0/energy/genEachTotalEnergy"] = hist


        hist = TH1F("genEachKineticEnergy","Generated Particles Kinetic Energy, Each",
                    100,0.0,2.0)
        hist.GetXaxis().SetTitle("Particle Kinetic Energy [MeV]")
        hist.GetYaxis().SetTitle("Generated Particles")
        hist.SetLineColor(4)
        self.stats["/file0/energy/genEachKineticEnergy"] = hist




        hist = TH1F("simScintEnergy","Energy deposited in Scintillator",
                    500,0.0,10.0)
        hist.GetXaxis().SetTitle("Ionization Energy [MeV]")
        hist.GetYaxis().SetTitle("Simulated Events")
        hist.SetLineColor(4)
        self.stats["/file0/energy/simScintEnergy"] = hist




        hist = TH2D("genRZ", "Generation Vertex R-Z", 100, 0.0, 6.1504, 100, -2.48, 2.48)
        self.stats["/file0/position/genRZ"] = hist
        if status.isFailure(): return status


        hist = TH2D("genXY", "Generation Vertex X-Y", 100, -2.48, 2.48, 100, -2.48, 2.48)
        self.stats["/file0/position/genXY"] = hist
        if status.isFailure(): return status







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
            pos = vertex.position()
            print "thho vertex position is", pos.x(), pos.y(), pos.z()
            for particle in irange(vertex.particles_out_const_begin(),
                                   vertex.particles_out_const_end()):
                particlePDGID = particle.pdg_id()
                print "thho Particle is ", particlePDGID
                if (particlePDGID != 1000200400) and (particlePDGID != 1000180400) and (particlePDGID != 11):
                    print "thho sssss particle is ", particlePDGID
                #print "thho particle is ", particle
                #print "vertex.particles_out_const_begin() is", vertex.particles_out_const_begin()
                #print "vertex.particles_out_const_end() is", vertex.particles_out_const_end()
                genEachTotalEnergy = particle.momentum().e()
                genEachKineticEnergy = (particle.momentum().e() - particle.momentum().m())
                #print "totalGenEnergy is ", totalGenEnergy
                #print "totalGenKineticEnergy is ", totalGenKineticEnergy 
        #self.stats["/file0/energy/genEnergy"].Fill(totalGenEnergy)
        #self.stats["/file0/energy/genKineticEnergy"].Fill(totalGenKineticEnergy)
                self.stats["/file0/energy/genEachTotalEnergy"].Fill(genEachTotalEnergy)
                self.stats["/file0/energy/genEachKineticEnergy"].Fill(genEachKineticEnergy)



        print "Executing sim hist algorithm......"
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
            self.stats["/file0/energy/simScintEnergy"].Fill( simScintEnergy )
        print " sim hist algorithm looping......"



        tGen = simStats["t_Trk1"].sum()
        xGen = simStats["x_Trk1"].sum()
        yGen = simStats["y_Trk1"].sum()
        zGen = simStats["z_Trk1"].sum()

        # Get underlying DE object
        de = self.getDet(self.target_de_name)
        if not de:
            print 'Failed to get DE',self.target_de_name
            return FAILURE

        # Get the AD coordinates of the vertexes
        import PyCintex
        Gaudi = PyCintex.makeNamespace('Gaudi')
        genGlbPoint = Gaudi.XYZPoint(xGen, yGen, zGen)
        #point = de.geometry().toGlobal(point)
        genLclPoint = de.geometry().toLocal(genGlbPoint)
        #print 'Current point is [',point.x(),point.y(),point.z(),']'
        #print 'In global coordinate [',gpoint.x(),gpoint.y(),gpoint.z(),']'



        print 'Generation locations', \
            '[', genGlbPoint.x(), genGlbPoint.y(), genGlbPoint.z(),']', \
            '[', genLclPoint.x()/units.cm, genLclPoint.y()/units.cm, genLclPoint.z()/units.cm,']'

        self.stats["/file0/position/genRZ"].Fill(genLclPoint.x()/units.meter * genLclPoint.x()/units.meter + genLclPoint.y()/units.meter * genLclPoint.y()/units.meter, genLclPoint.z()/units.meter)

        self.stats["/file0/position/genXY"].Fill(genLclPoint.x()/units.meter,genLclPoint.y()/units.meter)






        return SUCCESS
       
    def finalize(self):
        self.info("finalizing")
        status = DybPythonAlg.finalize(self)
        return status


#####  Job Configuration for nuwa.py ########################################

def configure():
    from StatisticsSvc.StatisticsSvcConf import StatisticsSvc
    statsSvc = StatisticsSvc()
    statsSvc.Output ={"file0":"simpleHistsK40.root"}
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

