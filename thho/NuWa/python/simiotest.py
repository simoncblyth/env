#!/usr/bin/env python

import sys
try:
    io = sys.argv[1]
except IndexError:
    io = "output"
input = False
if "input" == io: input = True

output = not input
use_aes = True
gen_location = "/Event/Gen/GenHeader"
sim_location = "/Event/Sim/SimHeader"

import GaudiPython as gp

if use_aes:

    iapp = gp.iService("ApplicationMgr")
    iapp.SvcMapping = [
        'EvtDataSvc/EventDataArchiveSvc',
        'DybDataSvc/EventDataSvc', 
        #"EvtDataSvc/EventDataSvc",
        "DetDataSvc/DetectorDataSvc",
        "HistogramSvc/HistogramDataSvc",
        "HbookCnv::PersSvc/HbookHistSvc",
        "RootHistCnv::PersSvc/RootHistSvc",
        "EvtPersistencySvc/EventPersistencySvc",
        "DetPersistencySvc/DetectorPersistencySvc",
        "HistogramPersistencySvc/HistogramPersistencySvc",
        ]


if not input:
    import xmldetdesc
    xmldetdesc.config()

app = gp.AppMgr(outputlevel=3)

msg = app.service("MessageSvc")
#msg.Format = "\--%-F%40W%S%7W%R%T %23W%t\n \-> %0W%M"
msg.Format = "% F%25W%S%7W%R%T %0W%M"
msg.useColors = False
msg.fatalColorCode=['red','white']
msg.errorColorCode=['red']
msg.warningColorCode=['yellow']
msg.debugColorCode=['blue']
msg.verboseColorCode=['cyan']


app.EvtMax = 3
if input:
    app.EvtSel =""              # yeah, I know, it's weird.
else:
    app.EvtSel = "NONE"

if input:
    #app.ExtSvc += [ "RootIOEvtSelector/EventSelector" ]
    app.ExtSvc += [ "DybEvtSelector/EventSelector" ]
if output:
    app.ExtSvc += [ "DybStorageSvc" ]
    dss = app.service("DybStorageSvc")
    dss.OutputLevel = 1
    
app.ExtSvc += [ "RootIOCnvSvc" ]

per = app.service("EventPersistencySvc")
per.CnvServices = [ "RootIOCnvSvc" ];

eds = app.service("EventDataService")
eds.OutputLevel = 1

rio = app.property("RootIOCnvSvc")
rio.OutputLevel = 1

iomap = { "default": "simiotest.root" };

if input:
    rio.InputStreams = iomap
if output:
    rio.OutputStreams = iomap

app.TopAlg = [ ]

# set up gentools
if not input:
    import xmldetdesc
    xmldetdesc.config()

    volume = "/dd/Structure/AD/far-oil2"

    # time
    import GaudiKernel.SystemOfUnits as units
    tim = app.property("ToolSvc.GtTimeratorTool")
    tim.LifeTime = int(100*units.second)

    # position
    poser = app.property("ToolSvc.GtPositionerTool")
    poser.Volume = volume
    poser.Mode = "Fixed"
    poser.Position = [0,0,0]

    # momentum
    gun = app.property("ToolSvc.GtGunGenTool")
    gun.ParticlesPerEvent = 1
    gun.ParticleName = "e+"
    gun.Momentum = 3*units.MeV
    gun.MomentumMode = "Fixed"
    gun.DirectionMode = "Fixed"
    gun.Direction = [ 1,0,0 ]

    # translate from local volume to global
    trans = app.property("ToolSvc.GtTransformTool")
    trans.Volume = volume
    
    app.TopAlg += [ 'GtGenerator/gen' ]
    gen = app.algorithm("gen")
    gen.GenTools = [ "GtGunGenTool", "GtPositionerTool", "GtTimeratorTool", "GtTransformTool" ]
    gen.GenName = "Bang Bang"
    gen.Location = gen_location
    pass

# set up detsim
if not input:
    modularPL = app.property("GiGa.GiGaPhysListModular")
    #modularPL.OutputLevel = 1
    modularPL.CutForElectron = 100*units.micrometer
    modularPL.CutForPositron = 100*units.micrometer
    modularPL.CutForGamma = 1*units.millimeter
    modularPL.PhysicsConstructors = [ 
        "DsPhysConsGeneral", 
        "DsPhysConsOptical",
        "DsPhysConsEM",
        #"DsPhysConsElectroNu",
        #"DsPhysConsHadron",
        #"DsPhysConsIon"
        ]
    optical = app.property("GiGa.GiGaPhysListModular.DsPhysConsOptical")
    #optical.UseCerenkov = False
    #optical.UseScintillation = False

    giga = app.service("GiGa")
    #giga.OutputLevel = 1
    giga.PhysicsList = "GiGaPhysListModular"
    giga.SteppingAction = "GiGaStepActionSequence"
    stepseq = app.property("GiGa.GiGaStepActionSequence")
    stepseq.Members = ["HistorianStepAction","UnObserverStepAction"]

    params = {
        'start' :"(start > 0)",
        'track1':"(id==1 and ProcessType==1)",
        'track2':"(id==2 and ProcessType==1)",
        'GD':    "MaterialName == '/dd/Materials/GdDopedLS'",
        'LS':    "MaterialName == '/dd/Materials/LiquidScintillator'",
        'oil':   "MaterialName == '/dd/Materials/MineralOil'",
        'iAV':   "MaterialName == '/dd/Materials/Acrylic'",
        'oAV':   "MaterialName == '/dd/Materials/Acrylic'"
        }
    unobs = app.property("GiGa.GiGaStepActionSequence.UnObserverStepAction")
    unobs.Stats=[
        ["pdgId_Trk1","pdg","%(track1)s and %(start)s"%params],
        ["t_Trk1",    "t" , "%(track1)s and %(start)s"%params],
        ["x_Trk1",    "lx", "%(track1)s and %(start)s"%params],
        ["y_Trk1",    "ly", "%(track1)s and %(start)s"%params],
        ["z_Trk1",    "lz", "%(track1)s and %(start)s"%params],
        ["e_Trk1",    "E",  "%(track1)s and %(start)s"%params],
        ["p_Trk1",    "p",  "%(track1)s and %(start)s"%params],
        ["ke_Trk1",   "KE", "%(track1)s and %(start)s"%params],
        ["vx_Trk1",   "lvx","%(track1)s and %(start)s"%params],
        ["vy_Trk1",   "lvy","%(track1)s and %(start)s"%params],
        ["vz_Trk1",   "lvz","%(track1)s and %(start)s"%params],
        ["TrkLength_GD_Trk1",  "dx","%(track1)s and %(GD)s"%params],
        ["TrkLength_iAV_Trk1", "dx","%(track1)s and %(iAV)s"%params],
        ["TrkLength_LS_Trk1",  "dx","%(track1)s and %(LS)s"%params],
        ["TrkLength_oAV_Trk1", "dx","%(track1)s and %(oAV)s"%params],
        ["TrkLength_Oil_Trk1", "dx","%(track1)s and %(oil)s"%params],
        # for track 2 now
        ["pdgId_Trk2","pdg","%(track2)s and %(start)s"%params],
        ["t_Trk2",    "t" , "%(track2)s and %(start)s"%params],
        ["x_Trk2",    "lx", "%(track2)s and %(start)s"%params],
        ["y_Trk2",    "ly", "%(track2)s and %(start)s"%params],
        ["z_Trk2",    "lz", "%(track2)s and %(start)s"%params],
        ["e_Trk2",    "E",  "%(track2)s and %(start)s"%params],
        ["p_Trk2",    "p",  "%(track2)s and %(start)s"%params],
        ["ke_Trk2",   "KE", "%(track2)s and %(start)s"%params],
        ["vx_Trk2",   "lvx","%(track2)s and %(start)s"%params],
        ["vy_Trk2",   "lvy","%(track2)s and %(start)s"%params],
        ["vz_Trk2",   "lvz","%(track2)s and %(start)s"%params],
        ["TrkLength_GD_Trk2",  "dx","%(track2)s and %(GD)s"%params],
        ["TrkLength_iAV_Trk2", "dx","%(track2)s and %(iAV)s"%params],
        ["TrkLength_LS_Trk2",  "dx","%(track2)s and %(LS)s"%params],
        ["TrkLength_oAV_Trk2", "dx","%(track2)s and %(oAV)s"%params],
        ["TrkLength_Oil_Trk2", "dx","%(track2)s and %(oil)s"%params]
        ]

    gggeo = app.service("GiGaGeo")
    #gggeo.OutputLevel = 1
    gggeo.XsizeOfWorldVolume = 2.4*units.kilometer
    gggeo.YsizeOfWorldVolume = 2.4*units.kilometer
    gggeo.ZsizeOfWorldVolume = 2.4*units.kilometer
    app.TopAlg += [ "GaudiSequencer/SimSeq" ]
    simseq = app.algorithm("SimSeq")
    simseq.Members = [ "GiGaInputStream/GGInStream" ]

    ggin = app.algorithm("GGInStream")
    #ggin.OutputLevel = 1
    ggin.ExecuteOnce = True
    ggin.ConversionSvcName = "GiGaGeo"
    ggin.DataProviderSvcName = "DetectorDataSvc"
    ggin.StreamItems = [ "/dd/Structure/Sites/far-rock",
                         "/dd/Geometry/AdDetails/AdSurfacesAll",
                         "/dd/Geometry/AdDetails/AdSurfacesFar",
                         "/dd/Geometry/PoolDetails/FarPoolSurfaces",
                         "/dd/Geometry/PoolDetails/PoolSurfacesAll",
                         ]

    simseq.Members += [ "DsPushKine/PushKine", "DsPullEvent/PullEvent" ]
    push = app.algorithm("PushKine")
    push.Converter = "HepMCtoG4"
    push.Location = gen_location
    
    pull = app.algorithm("PullEvent")
    pull.GenLocation = gen_location
    pull.Location = sim_location
    pull.OutputLevel = 2
    
    # Class name to use is set in DetDesc xml's "sensdet" attribute.
    pmtsd = app.property("GiGaGeo.DsPmtSensDet")
    #pmtsd.OutputLevel = 2
    pass


if input:
    rioes = app.service("RootIOEvtSelector")
    rioes.OutputLevel = 1

app.TopAlg += [ 'RegSeqDumpAlg/rsd' ]
rsd = app.algorithm("rsd")
rsd.OutputLevel = 1

if output:
    app.TopAlg += [ 'DybStoreAlg/dsa' ]
    dsa = app.algorithm("dsa")
    dsa.OutputLevel = 1

# Run...
app.initialize()
app.run(app.EvtMax)
