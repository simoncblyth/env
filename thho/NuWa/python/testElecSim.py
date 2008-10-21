#!/usr/bin/env python

from  GaudiPython import AppMgr
from GaudiKernel import SystemOfUnits as units

import os,sys

print "\tBuilding detector\n"
import xmldetdesc
xmldetdesc.config()
print "\tFinish importing detector module\t"

from  GaudiPython import AppMgr
app = AppMgr()
app.TopAlg = []
app.EvtSel = "NONE"

ms = app.service('MessageSvc')
#ms.OutputLevel=1

ms.useColors=True
ms.fatalColorCode=['red','white']
ms.errorColorCode=['red']
ms.warningColorCode=['yellow']
ms.debugColorCode=['blue']
ms.verboseColorCode=['cyan']

volume = "/dd/Structure/AD/far-oil2"

# Set up timerator
tim = app.property("ToolSvc.GtTimeratorTool")
tim.LifeTime = int(1*units.second)

# Set up positioner
poser = app.property("ToolSvc.GtPositionerTool")
#poser.OutputLevel = 1
poser.Strategy = "FullVolume"
poser.Volume = volume
poser.Mode = "Fixed"
#poser.Mode = "Smeared"
#poser.Spread = 1*units.meter
poser.Position = [0,0,0]

# Set up gun
gun = app.property("ToolSvc.GtGunGenTool")
#gun.OutputLevel = 1
gun.ParticlesPerEvent = 1
#gun.ParticleName = "opticalphoton"
gun.ParticleName = "e+"
print 'Particle is ', gun.ParticleName
#gun.Momentum = 6*units.eV   # try to enhance rayleigh scattering
gun.Momentum = 2.5*units.eV
gun.MomentumMode = "Fixed"
#gun.MomentumMode = "Smeared"
gun.MomentumSpread = 1*units.eV
gun.DirectionMode = "Fixed"

#from math import sin, cos, pi
#pmt_column_number = 9
#angle = (2*pmt_column_number - 1)*pi/24.0;
#gun.Direction = [ cos(angle),sin(angle),0 ] # aim for PMT 
gun.Direction = [ 1, 0, 0 ] # aim for a PMT
print 'gun.Direction=',gun.Direction

trans = app.property("ToolSvc.GtTransformTool")
trans.Volume = volume
                

app.TopAlg += [ "GaudiSequencer/GenSeq" ]
genseq = app.algorithm("GenSeq")
genseq.Members = [ "GtGenerator/GenAlg", "GtHepMCDumper/GenDump" ]
print 'app.TopAlg is',app.TopAlg

gen = app.algorithm("GenAlg")
#gen.OutputLevel = 1
gen.GenTools = [ "GtGunGenTool", "GtPositionerTool", "GtTimeratorTool", "GtTransformTool" ]
gen.GenName = "Bang Bang"
#gen.Location = "/Event/Gen/GenHeader" # this is default anyways

#print " GtDumper"
gendump = app.algorithm("GenDump")
#gendump.Location = "/Event/Gen/GenHeader"  # this is default anyways.

app.ExtSvc += ["GiGa"]

modularPL = app.property("GiGa.GiGaPhysListModular")
#modularPL.OutputLevel = 1
modularPL.CutForElectron = 100*units.micrometer
modularPL.CutForPositron = 100*units.micrometer
modularPL.CutForGamma = 1*units.millimeter
modularPL.PhysicsConstructors = [ 
    "DsPhysConsGeneral", 
    "DsPhysConsOptical",
    "DsPhysConsEM"
#    "DsPhysConsElectroNu",
#    "DsPhysConsHadron",
#    "DsPhysConsIon"
    ]

# in DsPhysConsOptical.cc these properties are declared.
optical = app.property("GiGa.GiGaPhysListModular.DsPhysConsOptical")

#optical.UseScintillation = False  # false option will kill all optical photons, should be true all the time
#optical.UseRayleigh = True       # enable rayleigh scattering

giga = app.service("GiGa")
giga.OutputLevel = 3
giga.PhysicsList = "GiGaPhysListModular"

gggeo = app.service("GiGaGeo")
gggeo.OutputLevel = 3
gggeo.XsizeOfWorldVolume = 2.4*units.kilometer
gggeo.YsizeOfWorldVolume = 2.4*units.kilometer
gggeo.ZsizeOfWorldVolume = 2.4*units.kilometer

giga.SteppingAction = "GiGaStepActionSequence"
stepseq = app.property("GiGa.GiGaStepActionSequence")
stepseq.Members = ["HistorianStepAction","UnObserverStepAction"]


TH2DE="TH2DE"       # TH2DE is the fastest
#TH2DE="TouchableToDetectorElementFast"
historian = app.property("GiGa.GiGaStepActionSequence.HistorianStepAction")
historian.TouchableToDetelem = TH2DE

params = {
    'track1':"(id == 1 and ProcessType == 1)",
    'track2':"(id == 2 and ProcessType == 1)",
#    'track1':"(id == 1)",
#    'track2':"(id == 2)",
    'GD':    "MaterialName == '/dd/Materials/GdDopedLS'",
    'LS':    "MaterialName == '/dd/Materials/LiquidScintillator'",
    'oil':   "MaterialName == '/dd/Materials/MineralOil'",
    'iAV':   "MaterialName == '/dd/Materials/Acrylic'",
    'oAV':   "MaterialName == '/dd/Materials/Acrylic'"
    }
unobs = app.property("GiGa.GiGaStepActionSequence.UnObserverStepAction")
unobs.TouchableToDetelem = TH2DE
unobs.Stats=[
    ["pdgId_Trk1","pdg","%(track1)s"%params],
    ["t_Trk1",    "t" , "%(track1)s"%params],
    ["x_Trk1",    "lx", "%(track1)s"%params],
    ["y_Trk1",    "ly", "%(track1)s"%params],
    ["z_Trk1",    "lz", "%(track1)s"%params],
    ["e_Trk1",    "E",  "%(track1)s"%params],
    ["p_Trk1",    "p",  "%(track1)s"%params],
    ["ke_Trk1",   "KE", "%(track1)s"%params],
    ["vx_Trk1",   "lvx","%(track1)s"%params],
    ["vy_Trk1",   "lvy","%(track1)s"%params],
    ["vz_Trk1",   "lvz","%(track1)s"%params],
    ["TrkLength_GD_Trk1",  "dx","%(track1)s and %(GD)s"%params],
    ["TrkLength_iAV_Trk1", "dx","%(track1)s and %(iAV)s"%params],
    ["TrkLength_LS_Trk1",  "dx","%(track1)s and %(LS)s"%params],
    ["TrkLength_oAV_Trk1", "dx","%(track1)s and %(oAV)s"%params],
    ["TrkLength_Oil_Trk1", "dx","%(track1)s and %(oil)s"%params]
    ]
                                                

# Make Geant4 sing!
ggrm = app.property("GiGa.GiGaMgr")
ggrm.Verbosity = 3
event_ac_cmds = app.property("GiGa.GiGaEventActionCommand")
verbosity_cmds = [
    "/control/verbose 0",
    "/run/verbose 0",
    "/event/verbose 2",
    "/tracking/verbose 2",
    "/geometry/navigator/verbose 0"
    ]
silent_cmds = [
    "/control/verbose 0",
    "/run/verbose 0",
    "/event/verbose 0",
    "/tracking/verbose 0",
    "/geometry/navigator/verbose 0"
    ]

event_ac_cmds.BeginOfEventCommands = silent_cmds
giga.EventAction = "GiGaEventActionCommand"


app.TopAlg += [ "GaudiSequencer/SimSeq" ]
simseq = app.algorithm("SimSeq")
simseq.Members = [ "GiGaInputStream/GGInStream" ]

ggin = app.algorithm("GGInStream")
#ggin.OutputLevel = 1
ggin.ExecuteOnce = True
ggin.ConversionSvcName = "GiGaGeo"
ggin.DataProviderSvcName = "DetectorDataSvc"
#ggin.StreamItems = [ "/dd/Structure/DayaBay", ]
ggin.StreamItems = [ "/dd/Structure/Sites/far-rock", ]

simseq.Members = [ "GiGaInputStream/GGInStream", "DsPushKine/PushKine", "DsPullEvent/PullEvent" ]
pull = app.algorithm("PullEvent")
#pull.OutputLevel = 1 
push = app.algorithm("PushKine")
push.Converter = "HepMCtoG4"
#push.Location = "/Event/Gen/GenHeader" # default anyway

# Class name to use is set in DetDesc xml's "sensdet" attribute.
pmtsd = app.property("GiGaGeo.DsPmtSensDet")
pmtsd.OutputLevel = 4

app.TopAlg += [ "EsFrontEndAlg" ]

#feeAlg = app.algorithm("EsFrontEndAlg")
#feeAlg.PmtTool = "EsPmtEffectPulseTool"

from tools import *
trigalg = SimpleTriggerAlg("SimpleTriggerAlg")
fpgaalg = SimpleFPGAAlg("SimpleFPGAAlg")
writealg = FeeReadoutWriter("FeeReadoutWriter")
app.addAlgorithm(trigalg)
app.addAlgorithm(fpgaalg)
app.addAlgorithm(writealg)
writealg.setOutputFile("ElecReadout.root")

histsvc = app.service("THistSvc")
histsvc.Output =["file1 DATAFILE='photon2.5.root' OPT='RECREATE' TYP='ROOT' "]

eds = app.service("EventDataSvc")
eds.ForceLeaves = True

app.initialize()
app.run(1)


# Examine the data in the transient data store
import GaudiPython
evt = app.evtSvc()
genHdr = evt['/Event/Gen/GenHeader']
simHdr = evt['/Event/Sim/SimHeader']
elecHdr = evt['/Event/Elec/ElecHeader']
print "dT = ",elecHdr.latest().GetNanoSec()-elecHdr.earliest().GetNanoSec()

det = GaudiPython.ROOT.DayaBay.Detector(0x04,2) # far, AD2
packedDet = det.siteDetPackedData()
channelId = GaudiPython.ROOT.DayaBay.FeeChannelId(1,3,0x04,2)

simhitHdr= simHdr.hits()
pulseHdr = elecHdr.pulseHeader()
crateHdr = elecHdr.crateHeader() 

hit_map   = simhitHdr.hitCollection()
pulse_map = pulseHdr.pulseCollection()
crate_map = crateHdr.crates()

pulses = pulse_map[det].pulses()
crate = crate_map[det]

for pulse in pulses:
    time = pulse.time()
    print "time",time

channel = crate.channel(channelId)
tdc = channel.tdc()
for i in range(len(tdc)):
    print i,"tdc",tdc[i]

#Lboard = 12
#Lconnectors = 16
#TDCsum = 0
#HitSum = 0
#NhitSum = 0
#for board in range(1,Lboard+1):
#    for conn in range(1,Lconnectors+1):
#        channelId = GaudiPython.ROOT.DayaBay.FeeChannelId(board,conn,0x04,2) 
#        channel = crate.channel(channelId)
#        tdc = channel.tdc()
#        hit = channel.hit()
#        TDCsum += len(tdc)
#        HitSum += sum(hit)
#        print board,conn,len(tdc),sum(hit)
#    channelId = GaudiPython.ROOT.DayaBay.FeeChannelId(board,1,0x04,2) 
#    NhitSum += max(crate.nhitSignal(channelId.boardId()))
#print "TDCsum =",TDCsum
#print "HitSum =",HitSum
#print "NhitSum =",NhitSum

#adcHigh = channel.adcHigh()
#adcHigh.size()
#print "ADC signals for one channel:"
#for adc in adcHigh:
#    print "   ", adc
#print ""

energy = channel.energy()
for i in range(len(energy)):
    if energy[i] > 0.00001:
        print i,"raw signal",energy[i]




print 'app.TopAlg is',app.TopAlg
