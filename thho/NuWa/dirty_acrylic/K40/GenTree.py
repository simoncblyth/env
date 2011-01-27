#!/usr/bin/env python
'''
Modified from people/zhang/jom/CheckSim/GenTree.py

Save generator info into root tree
Usage:
    nuwa.py --no-history -A 1 -n -1 -m "GenTree out.root" data_file
'''



from DybPython.DybPythonAlg import DybPythonAlg
from GaudiPython import SUCCESS, FAILURE
from GaudiPython import gbl, loaddict
from DybPython.Util import irange
import GaudiKernel.SystemOfUnits as units
import PyCintex
 
from ROOT import gStyle, TH1F, TH2F, TTree
from array import array
import math

# Make shortcuts
loaddict("libCLHEPRflx")
loaddict("libHepMCRflx")

ServiceMode = gbl.ServiceMode
Detector = gbl.DayaBay.Detector
AdPmtSensor = gbl.DayaBay.AdPmtSensor

#change default ROOT style
gStyle.SetHistLineColor(4)
gStyle.SetHistLineWidth(2)
gStyle.SetMarkerColor(4)
gStyle.SetMarkerStyle(8)
gStyle.SetPalette(1)


# Make your algorithm
class GenTree(DybPythonAlg):
    '''Save genHeader info into root tree '''
   
    def __init__(self,name):
        DybPythonAlg.__init__(self,name)
        return

    # ===========================================
    def initialize(self):
        status = DybPythonAlg.initialize(self)
        if status.isFailure(): return status
       
        self.cableSvc = self.svc('ICableSvc','StaticCableSvc')
        if self.cableSvc == None:
            self.error("Failed to get StaticCableSvc")
            return FAILURE
       
        self.info("initializing")
        self.nEvents = 0
        #self.target_de_name = '/dd/Structure/AD/db-ade1/db-sst1/db-oil1'
        self.target_de_name = '/dd/Structure/AD/db-lso1'
        self.genTree = TTree("genTree","genTree")
       
        self.initTree()
       
        self.stats['/file1/tree/genTree'] = self.genTree
       
        return SUCCESS

    # ===========================================
    def initTree(self):
         
        # genHeader info
       
        self.genType = array('i', [0])
        self.genX = array('d', [0])    # postion of each generator
        self.genY = array('d', [0])
        self.genZ = array('d', [0])
        self.nVtx = array('i', [0])    # number of vertices of each primary particle
        self.maxVtx = 10
        self.genParentPDG = array('i', self.maxVtx*[0])
        self.genParentKE = array('d', self.maxVtx*[0])
        self.genParentPx = array('d', self.maxVtx*[0])
        self.genParentPy = array('d', self.maxVtx*[0])
        self.genParentPz = array('d', self.maxVtx*[0])
        self.genParentE = array('d',  self.maxVtx*[0])       
        self.genParentT = array('d',  self.maxVtx*[0])       
        self.nPDG = array('i', [0])    # number of particles associated with this primary particle
        self.maxPDG = 40    # number of particles from primary particle
        self.genPDG = array('i', self.maxPDG*[0])
        self.genKE = array('d', self.maxPDG*[0])
        self.genPx = array('d', self.maxPDG*[0])
        self.genPy = array('d', self.maxPDG*[0])
        self.genPz = array('d', self.maxPDG*[0])
        self.genE = array('d', self.maxPDG*[0])
       
        self.genTree.Branch("genType", self.genType, "genType/I")
        self.genTree.Branch("genX", self.genX, "genX/D")
        self.genTree.Branch("genY", self.genY, "genY/D")
        self.genTree.Branch("genZ", self.genZ, "genZ/D")
        self.genTree.Branch("nVtx", self.nVtx, "nVtx/I")
        self.genTree.Branch("genParentPDG", self.genParentPDG, "genParentPDG[nVtx]/I")
        self.genTree.Branch("genParentKE", self.genParentKE, "genParentKE[nVtx]/D")
        self.genTree.Branch("genParentPx", self.genParentPx, "genParentPx[nVtx]/D")
        self.genTree.Branch("genParentPy", self.genParentPy, "genParentPy[nVtx]/D")
        self.genTree.Branch("genParentPz", self.genParentPz, "genParentPz[nVtx]/D")
        self.genTree.Branch("genParentE", self.genParentE, "genParentE[nVtx]/D")
        self.genTree.Branch("genParentT", self.genParentT, "genParentT[nVtx]/D")
        self.genTree.Branch("nPDG", self.nPDG, "nPDG/I")
        self.genTree.Branch("genPDG", self.genPDG, "genPDG[nPDG]/I")
        self.genTree.Branch("genKE", self.genKE, "genKE[nPDG]/D")
        self.genTree.Branch("genPx", self.genPx, "genPx[nPDG]/D")
        self.genTree.Branch("genPy", self.genPy, "genPy[nPDG]/D")
        self.genTree.Branch("genPz", self.genPz, "genPz[nPDG]/D")
        self.genTree.Branch("genE", self.genE, "genE[nPDG]/D")
       
        self.genTypes = {
            "IBD_gds":11, "IBD_lso":12, "IBD_oil":13, "IBD_acrylic":14,           
            "U238_gds":21, "U238_lso":22, "U238_PMT":23, "U238_sst":24, "U238_iav":25,
            "Th232_gds":31, "Th232_lso":32, "Th232_PMT":33, "Th232_sst":34, "Th235_iav":35,
            "K40_gds":41, "K40_lso":42, "K40_PMT":43, "K40_sst":44, "K40_iav":45,
            "Co60_sst":54,
        }
        self.pdgTypes = {
            "alpha" : 1000020040,
            "electron"  : 11, "positron" : -11,
            "gamma" : 22,
            "neutron" : 2112,
        }
       
    # ===========================================
    def reset(self):
        self.nPDG[0] = 0
        self.nVtx[0] = 0
        self.genType[0] = 0
        self.genX[0] = 0
        self.genY[0] = 0
        self.genZ[0] = 0   

    # ===========================================   
    def processGenHeader(self, genHdr):
        '''Process a list of genHeader (from pull simulation)'''
   
        if genHdr == None:
            self.error("Failed to get GenHeader")
            return SUCCESS
       
        nPDG = 0
        nVtx = 0
        genName = genHdr.generatorName()
        self.genType[0] = self.genTypes.get(genName, 0)
        genEvt = genHdr.event()
        for vtx in irange(genEvt.vertices_begin(), genEvt.vertices_end()):         
            if nVtx == 0:
                x, y, z, t = vtx.position().x(), vtx.position().y(), \
                             vtx.position().z(), vtx.position().t()
           
            for particle in irange(vtx.particles_in_const_begin(),
                                   vtx.particles_in_const_end()):
                self.genParentPDG[nVtx]  = particle.pdg_id()
                self.genParentKE[nVtx] = (particle.momentum().e() 
                                         - particle.momentum().m())/units.MeV
                self.genParentPx[nVtx] = particle.momentum().px()/units.MeV
                self.genParentPy[nVtx] = particle.momentum().py()/units.MeV
                self.genParentPz[nVtx] = particle.momentum().pz()/units.MeV
                self.genParentE[nVtx] = particle.momentum().e()/units.MeV
                self.genParentT[nVtx] = vtx.position().t()/units.ns
               
            for particle in irange(vtx.particles_out_const_begin(),
                                   vtx.particles_out_const_end()):
                self.genPDG[nPDG]  = particle.pdg_id()
                self.genKE[nPDG] = (particle.momentum().e() 
                                 - particle.momentum().m())/units.MeV
                self.genPx[nPDG] = particle.momentum().px()/units.MeV
                self.genPy[nPDG] = particle.momentum().py()/units.MeV
                self.genPz[nPDG] = particle.momentum().pz()/units.MeV
                self.genE[nPDG]  = particle.momentum().e()/units.MeV
                nPDG += 1
            nVtx += 1
           
        self.nPDG[0] = nPDG             
        self.nVtx[0] = nVtx    # nVtx > 1 for most Radioact generators
       
        # get vertex position
        de = self.getDet(self.target_de_name)
        if not de:
            self.info('Failed to get DE' + self.target_de_name)
            return FAILURE
        Gaudi = PyCintex.makeNamespace('Gaudi')
        genGlbPoint = Gaudi.XYZPoint(x, y, z)
        genLclPoint = de.geometry().toLocal(genGlbPoint)
        self.genX[0] = genLclPoint.x()/units.mm
        self.genY[0] = genLclPoint.y()/units.mm
        self.genZ[0] = genLclPoint.z()/units.mm
                 
    # ===========================================
    def execute(self):
        evt = self.evtSvc()
        self.nEvents += 1
        self.info("executing #" + str(self.nEvents))
        self.reset()
       
        genHdr = evt["/Event/Gen/GenHeader"]
        self.processGenHeader(genHdr)
       
       
        # Fill Tree
        self.genTree.Fill()
       
        return SUCCESS
     
    # ===========================================   
    def finalize(self):
        self.info("finalizing")
       
        status = DybPythonAlg.finalize(self)
        return status

#####  Job Configuration for nuwa.py ########################################

def configure(argv=[]):
    """Configuration"""
   
    # Setup root file for output histograms
    from StatisticsSvc.StatisticsSvcConf import StatisticsSvc
    statsSvc = StatisticsSvc()
    try:
        outfile = argv[0]
    except:
        outfile = 'out.root'
    statsSvc.Output = {"file1" : outfile}
    return

def run(app):
    """Add Algorithm"""
    app.ExtSvc += ["StatisticsSvc", "StaticCableSvc"]
    myAlg = GenTree("Gen Tree")
    app.addAlgorithm(myAlg)
    pass
