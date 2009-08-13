#!/usr/bin/env python

'''
usage example:

  nuwa.py -A -n 2 -o output.root -m "MDC09a.runIBD"
  Generating full inverse beta decay events
  
'''

DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S'

import os, math

def configure(argv = []):

    import sys, getopt
    from time import localtime, gmtime, mktime, strftime, strptime, timezone
    opts,args = getopt.getopt(argv,"p:w:s:n:v:")
    wallTime = 0
    gammaE = 1.0
    seed = "42"
    nevts = "10000"
    pmtDataPath = None
    volume = "/dd/Structure/Sites/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1"
    for opt,arg in opts:
        if opt == "-s":
            seed = arg
        if opt == "-n":
            nevts = arg
        if opt == "-v":
            volume = arg
        if opt == "-p":
            pmtDataPath = arg
        if opt == "-w":
            if -1 != arg.find('T'):
                wallTime = int(mktime(strptime(arg,
                                               DATETIME_FORMAT)) - timezone)
            else:
                wallTime = int(arg)
  
    print "======================================================"
    print "Begin JOB TIME = ", strftime(DATETIME_FORMAT,
                                              gmtime())
    print "IBD random seed: ", seed
    print "Number of IBD events: ", nevts
    print "Target volume: ", volume
    print "======================================================"
            
    import GenTools
    import GaudiKernel.SystemOfUnits as units
   
    #GenTools
    #ibd = "InverseBeta.exe -seed " + seed + " -n " + nevts + " |"
    print "InverseBeta.exe -n " + nevts + " -seed " + seed + " |"
    ibd = "InverseBeta.exe -n " + nevts + " -seed " + seed + " |"
    from GenTools.Helpers import HepEVT
    he = HepEVT(hepEvtDataSource = ibd)
    he.positioner.Volume = volume
    he.positioner.Strategy = "FullVolume"
    he.positioner.Mode = "Uniform"
#    he.positioner.Position = [0,0,2.5*units.m]
    he.positioner.Position = [0,0,0]
    he.positioner.Spread = 2.6*units.meter
    he.transformer.Volume = volume

    import GenTools
    gtc = GenTools.Configure()
    gtc.generator.TimeStamp = int(wallTime)
    gtc.register(he)

    # Then DetSim, with smaller than default configuration:
    import DetSim
    detsim = DetSim.Configure(site="dayabay")
    detsim.historian(trackSelection="(pdg == 2112)",vertexSelection="(pdg == 2112)")
    params = {
        'start' :"(start > 0)",
        'track1':"(id==1)",
        'track2':"(id==2)",
        'inGdLS':"DetectorElementName == 'db-gds1'",
        'inLS':  "DetectorElementName == 'db-lso1'",
        'GD':    "MaterialName == '/dd/Materials/GdDopedLS'",
        'LS':    "MaterialName == '/dd/Materials/LiquidScintillator'",
        'MO':   "MaterialName == '/dd/Materials/MineralOil'",
        'IAV':   "DetectorElementName == 'db-iav1'",
        'OAV':   "DetectorElementName == 'db-oav1'",
        'IWS': "MaterialName == '/dd/Materials/IwsWater'",
        'OWS': "MaterialName == '/dd/Materials/OwsWater'",
        'lastvtx': "IsStopping == 1",
        'firstvtx': "IsStarting == 1",
        'NeutronTrk': "pdg == 2112",
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

            ["tGen",   "t","%(NeutronTrk)s and %(firstvtx)s"%params],
            ["xGen",   "x","%(NeutronTrk)s and %(firstvtx)s"%params],
            ["yGen",   "y","%(NeutronTrk)s and %(firstvtx)s"%params],
            ["zGen",   "z","%(NeutronTrk)s and %(firstvtx)s"%params],

            ["tCap",   "t","%(NeutronTrk)s and %(lastvtx)s"%params],
            ["xCap",   "x","%(NeutronTrk)s and %(lastvtx)s"%params],
            ["yCap",   "y","%(NeutronTrk)s and %(lastvtx)s"%params],
            ["zCap",   "z","%(NeutronTrk)s and %(lastvtx)s"%params],

            ["capTarget", "capTargetZ","%(track2)s and %(lastvtx)s"%params],

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
            ["TrkLength_Oil_Trk1", "dx","%(track1)s and %(MO)s"%params],
            # track 2
            ["pdgId_Trk2","pdg","%(track2)s and %(start)s"%params],
            ["t_Trk2",    "t" , "%(track2)s and %(start)s"%params],
            ["x_Trk2",    "x", "%(track2)s and %(start)s"%params],
            ["y_Trk2",    "y", "%(track2)s and %(start)s"%params],
            ["z_Trk2",    "z", "%(track2)s and %(start)s"%params],
            ["tEnd_Trk2",    "t" , "%(track2)s and %(lastvtx)s"%params],
            ["xEnd_Trk2",    "x", "%(track2)s and %(lastvtx)s"%params],
            ["yEnd_Trk2",    "y", "%(track2)s and %(lastvtx)s"%params],
            ["zEnd_Trk2",    "z", "%(track2)s and %(lastvtx)s"%params],
            ["e_Trk2",    "E",  "%(track2)s and %(start)s"%params],
            ["p_Trk2",    "p",  "%(track2)s and %(start)s"%params],
            ["ke_Trk2",   "KE", "%(track2)s and %(start)s"%params],
            ["vx_Trk2",   "lvx","%(track2)s and %(start)s"%params],
            ["vy_Trk2",   "lvy","%(track2)s and %(start)s"%params],
            ["vz_Trk2",   "lvz","%(track2)s and %(start)s"%params],
            ["TrkLength_GD_Trk2",  "dx","%(track2)s and %(GD)s"%params],
            ["TrkLength_iAV_Trk2", "dx","%(track2)s and %(IAV)s"%params],
            ["TrkLength_LS_Trk2",  "dx","%(track2)s and %(LS)s"%params],
            ["TrkLength_oAV_Trk2", "dx","%(track2)s and %(OAV)s"%params],
            ["TrkLength_Oil_Trk2", "dx","%(track2)s and %(MO)s"%params]
            ])

#    import ElecSim
#    elecsim = ElecSim.Configure()
#    if pmtDataPath != None:
        # change PMT properties
#        elecsim.dataSvc.setPmtSimData( pmtDataPath )
    
#    import TrigSim
#    trigsim = TrigSim.Configure()

#    import ReadoutSim
#    rosim = ReadoutSim.Configure()

def run(app):
    pass

