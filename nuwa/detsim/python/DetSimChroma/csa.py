#!/usr/bin/env python
"""
usage example:

   nuwa.py --history=off -A none -n 100 -l 4 -R 100 -o muon.root -m "csa"


Adapted from $LOCAL_BASE/env/muon_simulation/optical_photon_weighting/OPW/fmcpmuon.py 

Some notes 22sept11
 change defaults to recommendations of doc6923
 maxTimeForWeighting = 300ns
 enable Dynamic Weighting with wsLimit = 10000, adLimits = 5000
 (with Dynamic Weighting enabled wsWeight and adWeights are unused)
  
"""

DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S'

import IPython as IP
import os, math, logging
import sys, getopt
from time import localtime, gmtime, mktime, strftime, strptime, timezone
from pprint import pformat

log = logging.getLogger(__name__)

from DybPython.Constants import *
import GaudiKernel.SystemOfUnits as units

cmg = None

def parse_args( argv ):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-e","--step",default="DetSim",type="str",
                      help="Step of simulation. DetSim, All, Digitization allowed. [default %default]")
    parser.add_option("-m","--mode",default="Full",type="str",
                      help="Simulation mode. Full or Fast. [default %default]")
    parser.add_option("-w","--start-time",default="0",type="str", 
                      help="Start time. [default %default]")

    parser.add_option("","--use-pregenerated-muons",action="store_true",default=False,
                      help="Use PREGENERATED muon file [%default]")

    parser.add_option("","--use-basic-physics",action="store_true",default=False,
                      help="Use BASIC physics list only [%default]")

    parser.add_option("","--Enable-Debug",action="store_true",default=False,
                      help="Set OutputLevel=DEBUG for DsFastMuonStackAction, DsPmtSensDet, DsPhysConsOptical [default %default]")

    parser.add_option("-s","--site",default="DayaBay",type="str",
                      help="site. DayaBay, Lingao or Far. [default %default]")

    parser.add_option("-l","--level",default="INFO",type="str",
                      help="INFO/DEBUG/WARN/... [default %default]")

    parser.add_option("-c","--chroma",action="store_true",
                      help="Send OPs off to remote Chroma for perusal [default %default]")

    parser.add_option(     "--chroma-flags",default="",type="str",
                      help="Chroma flags string [default %default]")

    parser.add_option("","--chroma-disabled",action="store_true",
                      help="Disable Chroma propagation but retain Chroma instrumentaion [default %default]")

    parser.add_option(  "--max-photon",default=5e6, type="int",
                      help="StackAction parameter, Maximum number of Optical Photons  [default %default]")

    parser.add_option(  "--modulo-photon",default=100, type="int",
                      help="StackAction parameter, Optical Photon prescale for unrealistic but faster testing [default %default]")

    parser.add_option("-t","--test",action="store_true",
                      help="Configure Gun for testing  [default %default]")

    parser.add_option(  "--machinerytest",action="store_true",
                      help="Machinery testing  [default %default]")

    parser.add_option("-D","--disable-op-weighting",action="store_true",
                      help="DISABLE optical photon weighting")

    parser.add_option("","--wsLimit",default=10000,type="int",
                      help="Threshold for optical photon generation in water before weighting kicks in. [default %default]")
    parser.add_option("","--wsWeight",default=10,type="int",
                      help="Weight per optical photon in water. Negative weight implies no optical photon weighting in water [default %default], but ignored if dynamic weighting is enabled.")
    
    parser.add_option("","--adVolumes",default="['oil','lso','gds']",
                      help="List of volume(s) in the AD for which a threshold and weight are specified. [default %default]")
    parser.add_option("","--adLimits",default="[5000,5000,5000]",
                      help="List of threshold(s) for optical photon generation in AD volume(s) before weighting kicks in. [default %default]")
    parser.add_option("","--adWeights",default="[100,100,100]",
                      help="List of weight(s) per optical photon in AD volume(s). [default %default], but ignored if dynamic weighting enabled")

    parser.add_option("","--useDynamicWeighting",action="store_true",default=True,
                      help="Use dynamic weighting once the # of OP is greater than the threshold.LIST OF WEIGHTS IS UNUSED IF DYNAMIC WEIGHTING ENABLED. [default %default]")
    parser.add_option("","--maxTimeForWeighting",default="300*nanosecond",
                      help="Maximum global time of photon that can be weighted. If time(OP) > maxTime, then OP will not be considered for weighting. [default maxTime %default]")

    
    (options, args) = parser.parse_args(args=argv)

    logging.basicConfig(level=getattr(logging, options.level.upper()))
    log.info(pformat(vars(options)))
    return options, args 


class ConfigMuonGeneration(object):
    """ script for muon generation. test optical photon weighting """
    sites = { 
       'DayaBay':("/dd/Structure/Pool/db-ows","/db-","DYB"),  # volume, volpref,musicsite 
       'Lingao':("/dd/Structure/Pool/la-ows","/la-", "LA"),
          'Far':("/dd/Structure/Pool/far-ows","/far-","Far"),
            }

    volume  = property(lambda self:self.sites[self.site][0])
    volpref = property(lambda self:self.sites[self.site][1])
    musicsite = property(lambda self:self.sites[self.site][2])
    allowedAdVolumes = ['oil', 'oav', 'lso', 'iav', 'gds']
 
    def __init__(self, argv ):

        from DybPython.Control import nuwa
        self.evtmax = nuwa.opts.executions
        self.seed = str(nuwa.opts.run)   # use NuWa's random seed
        self.Nevt = max(1000, self.evtmax*10)

        opts, args = parse_args( argv )

        assert opts.site in self.sites
        self.site = opts.site

        self.step = opts.step 
        self.mode = opts.mode
        self.t0   = opts.start_time
        self.test = opts.test
        self.machinerytest = opts.machinerytest
        self.opts = opts  

        self.check_argument_consistency(opts.adVolumes,opts.adLimits,opts.adWeights)
        self.check_muondatapath(os.getenv('MuonDataPath'))

    def banner(self):
        print "======================================================"
        print "Begin JOB TIME = ", strftime(DATETIME_FORMAT, gmtime())
        print "Random seed: ", self.seed
        print "Number of Muon from Muon.exe: ", self.Nevt
        print "Target volume: ", self.volume
        print "======================================================"

    def check_argument_consistency(self,vols,lims,wgts):
        """
        #  lists for prescaling in AD volumes must be same size
        """
        log.info("check_argument_consistency")
        vols, lims, wgts = map(eval, [vols,lims,wgts])
        arglis = [vols,lims,wgts]
        arglen = map(len,arglis)
        assert len(set(arglen)) == 1, ("inconsistent argument lists", arglis, arglen)

        self.adVolumes = vols
        self.adLimits = lims
        self.adWeights = wgts

        for adV in self.adVolumes:
            assert adV in self.allowedAdVolumes, ("volume %s not in allowed list %s " % (adV, repr(self.adVolumes)))

    def check_muondatapath(self, MuonDataPath):
        log.info("check_muondatapath %s " % MuonDataPath )
        if MuonDataPath is None:
            print "Muon data path ($MuonDataPath) is not defined."
            print "You may also need to get the data file from http://dayabay.ihep.ac.cn/svn/dybsvn/data/trunk/NewMuonGenerator/data/"
            sys.exit()
        else:
            print "Read muon data from ", MuonDataPath
        pass
        self.MuonDataPath = MuonDataPath 

    def _get_wallTime(self):
        t0 = self.opts.start_time 
        if -1 != t0.find('T'):
            wallTime = int(mktime(strptime(t0,  DATETIME_FORMAT)) - timezone)
        else:
            wallTime = int(t0)
        return wallTime
    wallTime = property(_get_wallTime) 

    def configure_muon_gun(self):
        log.info("configure_muon_gun")
        import GenTools
        from GenTools.Helpers import Gun
        mygun = Gun()
        mygun.gun.ParticleName = 'mu-'
        mygun.gun.Momentum = 30.0*units.GeV
        mygun.setVolume(self.volume)
        mygun.positioner.Position = [3*units.cm, 0*units.cm, 0*units.cm]
        mygun.timerator.LifeTime = 1*units.second
        pass
        gtc = GenTools.Configure()
        gtc.register(mygun)

    def configure_muon_generator(self):
        log.info("configure_muon_generator")
        source = "Muon.exe -n %s -s %s -seed %s -r Yes -v RPC -music_dir %s |"%(str(self.Nevt), self.musicsite, self.seed, self.MuonDataPath)
    
        if self.opts.use_pregenerated_muons: 
            print " Not using data from",self.MuonDataPath,"! USING PREGENERATED MUONS "

            pgfilename = 'tenthousandmuons'
            pgfilename = os.path.join(os.path.dirname(os.path.abspath(__file__)), pgfilename)
            #source = "cat " + pgfilename+ " |"
            source = pgfilename
            assert os.path.isfile(pgfilename), "FILE DOES NOT EXIST %s " % pgfilename
            print " muon source ", source
            print " musicSite: %s, and RandomSeed: %s"%(self.musicsite,self.seed)
        pass
            
        from GenTools.Helpers import HepEVT
    
        hepevt = HepEVT(source)
        hepevt.positioner.Volume = self.volume
        hepevt.positioner.Mode = "Relative"
        hepevt.positioner.Position = [0,0,0]
        hepevt.timerator.LifeTime = 1*units.second
        hepevt.transformer.Volume = self.volume
        hepevt.transformer.Offset = [0., 0., (0.042)*units.meter]
    
        import GenTools
        gt = GenTools.Configure(helper=hepevt)
        gt.generator.TimeStamp = int(self.wallTime)
        gt.generator.GenName = "Muon"

    def configure_run_manager(self):
        """
        tone down verbosity of G4RunManager, 
        skipping the particle table dump for Verbosity less than 3
        https://wiki.bnl.gov/dayabay/index.php?title=FAQ:How_to_turn_off_long_particle_listing%3F
        """
        log.info("configure_run_manager")
        from GiGa.GiGaConf import GiGa, GiGaRunManager
        giga = GiGa("GiGa")
        gigarm = GiGaRunManager("GiGa.GiGaMgr")
        gigarm.Verbosity = 2

    def configure_geometry_export(self):
        log.info("configure_geometry_export")
        from GaussTools.GaussToolsConf import GiGaRunActionCommand
        grac = GiGaRunActionCommand("GiGa.GiGaRunActionCommand")
        grac.BeginOfRunCommands = [
             "/vis/open VRML2FILE",
             "/vis/drawVolume",
             "/vis/viewer/flush"
        ]    
        from GiGa.GiGaConf import GiGa
        giga = GiGa()
        giga.RunAction = grac    
        giga.VisManager = "GiGaVisManager/GiGaVis"


    def configure_detsim(self):
        """
        See NuWa-trunk/dybgaudi/Simulation/DetSim/python/DetSim/Default.py
        """
        log.info("configure_detsim for site %s " %  self.site)

        import DetSim
        if self.opts.use_basic_physics :
            physlist=DetSim.physics_list_basic 
        else:
            physlist=DetSim.physics_list_basic + DetSim.physics_list_nuclear
        pass

        if self.opts.chroma:
            pco = physlist.index("DsPhysConsOptical")
            physlist[pco] = "DsChromaPhysConsOptical" 
        pass

        log.info("using physlist %s " % repr(physlist))
        detsim = DetSim.Configure(physlist=physlist,site=self.site)

        detsim.OutputLevel = 1

        self.detsim = detsim 


    def configure_physconsoptical(self):
        log.info("configure_physconsoptical")

        if self.machinerytest:
            log.info("configure_phyconsoptical : skipping due to --machinerytest")
            return  

        if self.opts.chroma:
            optical = self.configure_dschromaoptical()
        else:
            optical = self.configure_dsoptical()
        pass
        #optical.CerenPhotonScaleWeight = 3.0
        #optical.ScintPhotonScaleWeight = 3.0
        optical.UseScintillation = True
        fastMuEnergyCut = True
        if self.mode == "Full":
            optical.UseCerenkov = True
            optical.UseFastMu300nsTrick = False
            fastMuEnergyCut = False
        else:
            optical.UseCerenkov = False
            optical.UseFastMu300nsTrick = True
        pass
        self.fastMuEnergyCut = fastMuEnergyCut
        self.optical = optical


    def configure_dschromaoptical(self):
        log.info("configure_dschromaoptical")
        import DetSimChroma
        from DetSimChroma.DetSimChromaConf import DsChromaPhysConsOptical
        optical = DsChromaPhysConsOptical("GiGa.GiGaPhysListModular.DsChromaPhysConsOptical")
        return optical 

    def configure_dsoptical(self):
        log.info("configure_dsoptical")
        from DetSim.DetSimConf import DsPhysConsOptical
        optical = DsPhysConsOptical("GiGa.GiGaPhysListModular.DsPhysConsOptical")
        return optical 


    def configure_runaction(self):
        """
        """
        log.info("configure_runaction")
        from GiGa.GiGaConf import GiGa
        giga = GiGa("GiGa")

        if self.opts.chroma:
            action = self.configure_chromarunaction()
        else:
            action = None
        pass

        if not action is None:
            log.info("setting RunAction %s " % action )
            giga.RunAction = action
        else:
            log.warn("NOT setting RunAction")
        pass

    def configure_eventaction(self):
        log.info("configure_eventaction")
        from GiGa.GiGaConf import GiGa
        giga = GiGa("GiGa")

        if self.opts.chroma:
            action = self.configure_chromaeventaction()
        else:
            action = None
        pass

        if not action is None:
            log.info("setting EventAction %s " % action )
            giga.EventAction = action
        else:
            log.warn("NOT setting EventAction")
        pass


    def configure_stackaction(self):
        """
        # need this bit to use the stacking action defined above (taken from r11023 gonchar/muonSim.py)
        """
        log.info("configure_stackaction")
        from GiGa.GiGaConf import GiGa
        giga = GiGa("GiGa")

        DFMSA = False
        saction = None
        if self.opts.chroma:
            saction = self.configure_chromastackaction()
        elif self.opts.disable_op_weighting:
            print "DISABLE volume-based weighting of optical photons using DsFastMuonStackAction"
        else:
            print "ENABLE volume-based weighting of optical photons using DsFastMuonStackAction"
            saction = self.configure_fastmuonstackaction()
            DFMSA = True
        pass
        if not saction is None:
            giga.StackingAction = saction
            self.DFMSA = DFMSA
        else:
            log.warn("NOT SETTING giga.StackingAction as saction is None")
        pass

        if self.opts.Enable_Debug :
            from DetSim.DetSimConf import DsPmtSensDet
            procs = [self.optical]
            procs.append( DsPmtSensDet('GiGaGeo.DsPmtSensDet') )
            if self.DFMSA : procs.append( saction )
            for proc in procs :
                proc.OutputLevel = DEBUG
                print "Setting",proc.getName(),"OutputLevel to",proc.OutputLevel

    def configure_chromastackaction(self): 
        log.info("configure_chromastackaction")
        import DetSimChroma
        from DetSimChroma.DetSimChromaConf import DsChromaStackAction 
        action = DsChromaStackAction("GiGa.DsChromaStackAction")
        action.NeutronParent = False
        action.PhotonKill = False        # kill OP after collection
        action.MaxPhoton = self.opts.max_photon         
        action.ModuloPhoton = self.opts.modulo_photon    
        return action

    def configure_chromarunaction(self): 
        log.info("configure_chromarunaction")
        import DetSimChroma
        from DetSimChroma.DetSimChromaConf import DsChromaRunAction 

        action = DsChromaRunAction("GiGa.DsChromaRunAction")
        action.databasekey = "G4DAECHROMA_DATABASE_PATH" 
        action.transport = "G4DAECHROMA_CLIENT_CONFIG" 
        action.cachekey = "G4DAECHROMA_CACHE_DIR"     
        action.sensdet = "DsPmtSensDet" 
        action.TouchableToDetelem = "TH2DE"
        action.PackedIdPropertyName = "PmtID"
        action.EnableChroma = not self.opts.chroma_disabled
        action.ChromaFlags = self.opts.chroma_flags
        return action

    def configure_chromaeventaction(self): 
        log.info("configure_chromaeventaction")
        import DetSimChroma
        from DetSimChroma.DetSimChromaConf import DsChromaEventAction 
        action = DsChromaEventAction("GiGa.DsChromaEventAction")
        return action
 
    def configure_fastmuonstackaction(self):
        """
        #reminder of detector paths
        #saction.Detectors = [
        #    '/dd/Structure/DayaBay/db-rock/db-ows',
        #    '/dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws',
        #    '/dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1',
        #    '/dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade2/db-sst2/db-oil2' ]
        #saction.Limits = [ 10000, 10000, 10000, 10000 ]
        #saction.Weights = [ 10, 10, 10, 10 ]
        # /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade2/db-sst2/db-oil2/db-oav2/db-lso2/db-iav2/db-gds2

        # config for OP weighting, originally taken from doc6713-v4
        # See https://wiki.bnl.gov/dayabay/index.php?title=Weighted_optical_photons , also
        # next bit from doc6713-v4

        """
        log.info("configure_fastmuonstackaction")
        from DetSim.DetSimConf import DsFastMuonStackAction
        saction = DsFastMuonStackAction("GiGa.DsFastMuonStackAction")
        saction.Detectors = [ ]
        saction.Limits    = [ ]
        saction.Weights   = [ ]
        saction.UseDynamicWeighting = self.opts.useDynamicWeighting
        from DybPython.Tools import unitify
        saction.MaxTimeForWeighting = unitify( self.opts.maxTimeForWeighting )

        # modifications to set volumes, limits, weights based on inputs
        # assume common limit,weight for all water volumes
        wsLimit = self.opts.wsLimit
        wsWeight= self.opts.wsWeight
        OWS = self.volume.replace("Pool",self.site + self.volpref + "rock")
        IWS = OWS + self.volpref + 'curtain' + self.volpref + 'iws'
        if wsWeight > 0. : 
            saction.Detectors.append(OWS)
            saction.Limits.append(wsLimit)   
            saction.Weights.append(wsWeight) 
            saction.Detectors.append(IWS)
            saction.Limits.append(wsLimit)   
            saction.Weights.append(wsWeight) 

        # allow different limit,weights for concentric AD volumes
        totAD = 2
        if self.musicsite == 'Far' : totAD = 4
        for iAD in range(totAD):
            sAD = str(iAD+1)
            AD = IWS + self.volpref + 'ade' + sAD + self.volpref + 'sst' + sAD 
            for mat in self.allowedAdVolumes: 
                AD = AD + self.volpref + mat + sAD
                if mat in adVolumes:
                    saction.Detectors.append(AD)
                    ix = self.adVolumes.index(mat)
                    saction.Limits.append( self.adLimits[ix] )
                    saction.Weights.append( self.adWeights[ix] )
                pass
            pass             
        pass
        if len(saction.Detectors) > 0 : 
            return saction  
        pass
        print ' *** No volume-based weighting of optical photons will be applied because no volumes were specified or no non-negative weights were specified ***'
        return None



    def configure_historian(self):
        """
        historian and unobservable
        """ 
        log.info("configure_historian")
        if self.machinerytest:
            log.info("configure_historian : skipping due to --machinerytest")
            return  

        self.detsim.historian(
                trackSelection= "(pdg == 2112)",
                vertexSelection="( ((pdg == 13 or pdg == -13) or (pdg == 2112)) and (IsStopping == 1) )",
                useFastMuEnergyCut=self.fastMuEnergyCut
                )
        params = {
                'inLso1': "DetElem in '/dd/Structure/AD/db-lso1'",
                'inLso2': "DetElem in '/dd/Structure/AD/db-lso2'",
                'ingds1': "DetElem in '/dd/Structure/AD/db-gds1'",
                'ingds2': "DetElem in '/dd/Structure/AD/db-gds2'",
                'AD1':    "AD ==1",
                'AD2':    "AD ==2",
                'GD':    "MaterialName == '/dd/Materials/GdDopedLS'",
                'LS':    "MaterialName == '/dd/Materials/LiquidScintillator'",
                'MO':    "MaterialName == '/dd/Materials/MineralOil'",
                'IWS': "MaterialName == '/dd/Materials/IwsWater'",
                'OWS': "MaterialName == '/dd/Materials/OwsWater'",
                'lastvtx': "IsStopping == 1",
                'firstvtx': "IsStarting == 1",
                'NeutronTrk': "pdg == 2112",
                'NeutronMom': "creator == 2112",
                'NCap': "ProcessName == 'nCapture'",
                'Gamma': "pdg == 22",
                'Muon': "(pdg == 13 or pdg == -13)",
                'TimeCut': "time <= 300"
                 }

        self.detsim.unobserver(stats=[
                
                ["QEDepInLS1",        "qdE",   "(%(LS)s and %(AD1)s) and %(TimeCut)s"%params],
                ["QEDepInLS2",        "qdE",   "(%(LS)s and %(AD2)s) and %(TimeCut)s"%params],
                ["QEDepInGdLS1",      "qdE",   "(%(GD)s and %(AD1)s) and %(TimeCut)s"%params],
                ["QEDepInGdLS2",      "qdE",   "(%(GD)s and %(AD2)s) and %(TimeCut)s"%params],
                ["EDepInMO1",         "dE",    "(%(MO)s and %(AD1)s) and %(TimeCut)s"%params],
                ["EDepInMO2",         "dE",    "(%(MO)s and %(AD2)s) and %(TimeCut)s"%params],

                ["AllEDepInLS1",     "dE",   "%(LS)s and %(AD1)s"%params],
                ["AllEDepInLS2",     "dE",   "%(LS)s and %(AD2)s"%params],
                ["AllEDepInGdLS1",   "dE",   "%(GD)s and %(AD1)s"%params],
                ["AllEDepInGdLS2",   "dE",   "%(GD)s and %(AD2)s"%params],
                ["AllEDepInMO1",     "dE",   "%(MO)s and %(AD1)s"%params],
                ["AllEDepInMO2",     "dE",   "%(MO)s and %(AD2)s"%params],
                ["AllEDepInIWS",     "dE",   "%(IWS)s"%params],
                ["AllEDepInOWS",     "dE",   "%(OWS)s"%params],
                
                ["AllQEDepInLS1",     "qdE", "%(LS)s and %(AD1)s"%params],
                ["AllQEDepInLS2",     "qdE", "%(LS)s and %(AD2)s"%params],
                ["AllQEDepInGdLS1",   "qdE", "%(GD)s and %(AD1)s"%params],
                ["AllQEDepInGdLS2",   "qdE", "%(GD)s and %(AD2)s"%params],
                
                ["MuTrackInLS1",   "dx", "%(Muon)s and (%(LS)s and %(AD1)s)"%params],
                ["MuTrackInLS2",   "dx", "%(Muon)s and (%(LS)s and %(AD2)s)"%params],
                ["MuTrackInGdLS1", "dx", "%(Muon)s and (%(GD)s and %(AD1)s)"%params],
                ["MuTrackInGdLS2", "dx", "%(Muon)s and (%(GD)s and %(AD2)s)"%params],
                ["MuTrackInMO1",   "dx", "%(Muon)s and (%(MO)s and %(AD1)s)"%params],
                ["MuTrackInMO2",   "dx", "%(Muon)s and (%(MO)s and %(AD2)s)"%params],
                ["MuTrackInIWS",   "dx", "%(Muon)s and %(IWS)s"%params],
                ["MuTrackInOWS",   "dx", "%(Muon)s and %(OWS)s"%params]
                
                ])


    def configure_muonhitsim(self):
        log.info("configure_muonhitsim")
        if self.machinerytest:
            log.info("configure_muonhitsim : skipping due to --machinerytest")
            return  
        import MuonHitSim
        muonhitsim = MuonHitSim.Configure()

    def configure_digitization(self):
        """
        explicitly set suppression of very long time hits
        """
        log.info("configure_digitization")
        if self.machinerytest:
            log.info("configure_digitization : skipping due to --machinerytest")
            return  

        from ElecSim.ElecSimConf import EsPmtEffectPulseTool
        epept = EsPmtEffectPulseTool()
        epept.EnableVeryLongTimeSuppression = True
        epept.VeryLongTime = 1e7*units.ns

        import DigitizeAlg
        DigitizeAlg.Configure()
 
    def __call__(self):
        """
        Steering 
        """
        if self.step in ("All","DetSim"):
            if self.test:
                self.configure_muon_gun() 
            else:
                self.configure_muon_generator() 
            pass
            self.configure_run_manager()
            #self.configure_geometry_export()
            self.configure_detsim()
            self.configure_physconsoptical()

            # Geant4 actions 
            self.configure_runaction() 
            self.configure_stackaction() 
            self.configure_eventaction() 

            self.configure_historian()
        pass

        if self.mode == "Fast":
            self.configure_muonhitsim()

        if self.step in ("All","Digitization"):
             self.configure_digitization()
                    


from DybPython.DybPythonAlg import DybPythonAlg
from GaudiPython import SUCCESS, FAILURE

class InspectorAlg(DybPythonAlg):
    def __init__(self,name):
        DybPythonAlg.__init__(self,name)
        return

    def initialize(self):
        status = DybPythonAlg.initialize(self)
        self.count = 0
        if status.isFailure(): return status
        log.info("initializing")
        return SUCCESS

    def first(self):
        log.info("first")
        det = self.detSvc("/dd")
        log.info("det %s " % repr(det))
        #IP.embed()   causes resource not available tailspin 
        log.info("detSvc dump done")

    def execute(self):
        log.info("executing")
        if self.count == 0:
            self.first()
        pass
        self.count += 1
        evt = self.evtSvc()
        return SUCCESS

    def finalize(self):
        log.info("finalizing")
        status = DybPythonAlg.finalize(self)
        return status


def configure(argv=[]):
    global cmg 
    cmg = ConfigMuonGeneration(argv)
    cmg()
    cmg.banner()

  
def run(app):
    pass
    log.info("run app %s " % repr(app))
    inspectorAlg = InspectorAlg("MyInspectorAlg")
    app.addAlgorithm(inspectorAlg)



