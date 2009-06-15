#!/usr/bin/env python

'''
usage example:

  nuwa.py -A -n 2 -o output.root -m "MDC09a.runGamma -k 1.0"
  Photon is generated uniformly in the GdLS
  "-k argument specify energy of photon in MeV"
  
'''

DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S'

import os, math

def configure(argv = []):
    """Configure this module with gamma energy"""
    
    import sys, getopt
    from time import localtime, gmtime, mktime, strftime, strptime, timezone
    opts,args = getopt.getopt(argv,"p:w:k:")
    wallTime = 0
    gammaE = 1.0
    pmtDataPath = None
    for opt,arg in opts:
        if opt == "-p":
            pmtDataPath = arg
        if opt == "-w":
            if -1 != arg.find('T'):
                wallTime = int(mktime(strptime(arg,
                                               DATETIME_FORMAT)) - timezone)
            else:
                wallTime = int(arg)
 
        if opt == "-k":
            gammaE = float(arg)
            print "======================================================"
            print "Photon energy = ", gammaE, " MeV"
            print "======================================================"


    print "wallTime is ", wallTime
 
    print "======================================================"
    print "Begin JOB TIME = ", strftime(DATETIME_FORMAT,
                                              gmtime())
    print "======================================================"

    # Generator
    import GaudiKernel.SystemOfUnits as units
    from GenTools.Helpers import Gun
    from GenTools.GenToolsConf import GtGunGenTool, GtPositionerTool
    volume = "/dd/Structure/AD/db-oil1"
    gun = Gun(volume,
              gun = GtGunGenTool("gun",
                                 ParticlesPerEvent = 1,
                                 ParticleName = "gamma",
                                 Momentum = gammaE*units.MeV,
                                 MomentumMode = "Fixed",
                                 MomentumSpread = 0.*units.MeV,
                                 DirectionMode = "Uniform",
                                 Direction = [ 1, 0, 0 ],
                                 DirectionSpread = 3),
              positioner = GtPositionerTool("pos",
                                            Strategy = "FullVolume",
                                            Mode = "Uniform",
                                            Spread = 2.5*units.meter,
                                            Position = [0,0,0*units.meter])
              )
    gun.timerator.LifeTime = 0.020*units.second
    import GenTools
    GenTools.Configure().register(gun)
    
    # Detector Simulation
    import DetSim
    detsim = DetSim.Configure(site="dayabay", \
                              physlist = DetSim.physics_list_basic)
    detsim.historian(trackSelection="(pdg == 22)",\
                     vertexSelection="(pdg == 22)")
    params = {
        'start' :"(start > 0)",
        'track1':"(id==1)",
        'track2':"(id==2)",
        'GD':    "MaterialName == '/dd/Materials/GdDopedLS'",
        'LS':    "MaterialName == '/dd/Materials/LiquidScintillator'",
        'MO':   "MaterialName == '/dd/Materials/MineralOil'",
        'IAV':   "DetectorElementName == 'db-iav1'",
        'OAV':   "DetectorElementName == 'db-oav1'",
        'IWS': "MaterialName == '/dd/Materials/IwsWater'",
        'OWS': "MaterialName == '/dd/Materials/OwsWater'",
        'lastvtx': "IsStopping == 1",
        'firstvtx': "IsStarting == 1",
        'Neutron': "pdg == 2112",
        'NeutronMom': "creator == 2112",
        'Gamma': "pdg == 22",
        'Muon': "(pdg == 13 or pdg == -13)"
        }

    detsim.unobserver(stats=[
            ["EDepInGdLS", "dE", "%(GD)s"%params],
            ["EDepInLS", "dE", "%(LS)s"%params],
            ["EDepInIAV", "dE", "%(IAV)s"%params],
            ["EDepInOAV", "dE", "%(OAV)s"%params],
            ["EDepInOIL", "dE", "%(MO)s"%params],

            ["QEDepInGdLS", "qdE", "%(GD)s"%params],
            ["QEDepInLS", "qdE", "%(LS)s"%params],
            ["QEDepInIAV", "qdE", "%(IAV)s"%params],
            ["QEDepInOAV", "qdE", "%(OAV)s"%params],
            ["QEDepInOIL", "qdE", "%(MO)s"%params],

            ["tQESumGdLS", "qEt", "%(GD)s"%params],
            ["xQESumGdLS", "qEx", "%(GD)s"%params],
            ["yQESumGdLS", "qEy", "%(GD)s"%params],
            ["zQESumGdLS", "qEz", "%(GD)s"%params],

            ["tQESumLS", "qEt", "%(LS)s"%params],
            ["xQESumLS", "qEx", "%(LS)s"%params],
            ["yQESumLS", "qEy", "%(LS)s"%params],
            ["zQESumLS", "qEz", "%(LS)s"%params],

            ["tQESumMO", "qEt", "%(MO)s"%params],
            ["xQESumMO", "qEx", "%(MO)s"%params],
            ["yQESumMO", "qEy", "%(MO)s"%params],
            ["zQESumMO", "qEz", "%(MO)s"%params],
            
            # track 1
            ["pdgId_Trk1","pdg","%(track1)s and %(start)s"%params],
            ["t_Trk1",    "t" , "%(track1)s and %(start)s"%params],
            ["x_Trk1",    "x", "%(track1)s and %(start)s"%params],
            ["y_Trk1",    "y", "%(track1)s and %(start)s"%params],
            ["z_Trk1",    "z", "%(track1)s and %(start)s"%params],
            ["tEnd_Trk1",    "t" , "%(track1)s and %(lastvtx)s"%params],
            ["xEnd_Trk1",    "x", "%(track1)s and %(lastvtx)s"%params],
            ["yEnd_Trk1",    "y", "%(track1)s and %(lastvtx)s"%params],
            ["zEnd_Trk1",    "z", "%(track1)s and %(lastvtx)s"%params],
            ["e_Trk1",    "E",  "%(track1)s and %(start)s"%params],
            ["p_Trk1",    "p",  "%(track1)s and %(start)s"%params],
            ["ke_Trk1",   "KE", "%(track1)s and %(start)s"%params],
            ["vx_Trk1",   "lvx","%(track1)s and %(start)s"%params],
            ["vy_Trk1",   "lvy","%(track1)s and %(start)s"%params],
            ["vz_Trk1",   "lvz","%(track1)s and %(start)s"%params],
            ["TrkLength_GD_Trk1",  "dx","%(track1)s and %(GD)s"%params],
            ["TrkLength_iAV_Trk1", "dx","%(track1)s and %(IAV)s"%params],
            ["TrkLength_LS_Trk1",  "dx","%(track1)s and %(LS)s"%params],
            ["TrkLength_oAV_Trk1", "dx","%(track1)s and %(OAV)s"%params],
            ["TrkLength_Oil_Trk1", "dx","%(track1)s and %(MO)s"%params]
            ])
            
    import ElecSim
    elecsim = ElecSim.Configure()
    if pmtDataPath != None:
        # change PMT properties
        elecsim.dataSvc.setPmtSimData( pmtDataPath )
    
    import TrigSim
    trigsim = TrigSim.Configure()

    import ReadoutSim
    rosim = ReadoutSim.Configure()
    import ReadoutSim.ReadoutSimConf as ROsConf
    ROsConf.ROsReadoutAlg().RoTools=["ROsFecReadoutTool","ROsFeePeakOnlyTool"]

def run(app):
    pass

