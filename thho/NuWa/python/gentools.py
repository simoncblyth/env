#!/usr/bin/env python

volume = "/dd/Structure/Pool/la-iws"

print "\tConfiguring geometry\n"
import xmldetdesc
xmldetdesc.config()

from  GaudiPython import AppMgr
from GaudiKernel import SystemOfUnits as units

app = AppMgr()
app.TopAlg = []

# Set up timerator
import GaudiKernel.SystemOfUnits as units
tim = app.property("ToolSvc.GtTimeratorTool")
tim.LifeTime = int(1*units.second)
tim.OutputLevel = 1

spray_photons = False

# Set up positioner
poser = app.property("ToolSvc.GtPositionerTool")
poser.Volume = volume
poser.Mode = "Fixed"
if spray_photons:
    poser.Position = [0,3*units.meter,1.1*units.meter]
else:
    #poser.Position = [-3*units.meter,3*units.meter,0]
    poser.Position = [0,0,0]
#poser.OutputLevel = 1

# Set up gun
gun = app.property("ToolSvc.GtGunGenTool")
#gun.OutputLevel = 1
if spray_photons:
    gun.ParticlesPerEvent = 1000
    gun.ParticleName = "opticalphoton"
    gun.Momentum = 2.5*units.eV
    gun.DirectionMode = "Smeared"
    gun.DirectionSpread = 10*units.degree;
    gun.Direction = [ 0,1,0 ]
else:
    gun.ParticlesPerEvent = 1
    gun.ParticleName = "mu+"
    gun.Momentum = 10*units.GeV
    gun.DirectionMode = "Fixed"
    gun.Direction = [ 1,0,0 ]
    pass
gun.MomentumMode = "Fixed"

print 'gun.Direction=',gun.Direction

trans = app.property("ToolSvc.GtTransformTool")
trans.Volume = volume

app.TopAlg += [ "GaudiSequencer/GenSeq" ]
genseq = app.algorithm("GenSeq")
genseq.Members = [ "GtGenerator/GenAlg", 
                   #"GtHepMCDumper/GenDump" 
                   ]


gen = app.algorithm("GenAlg")
#gen.OutputLevel = 1
gen.GenTools = [ "GtGunGenTool", "GtPositionerTool", "GtTimeratorTool", "GtTransformTool" ]
gen.GenName = "Bang Bang"
#gen.OutputFilePath = "/file1/gen"
gen.OutputLevel = 1

histsvc = app.service("THistSvc")
histsvc.Output = [ "file1 DATAFILE='gentools.root' OPT='RECREATE' TYP='ROOT'" ]

#print " GtDumper"
gendump = app.algorithm("GenDump")




app.EvtSel = "NONE"

from PyCintex import *
loadDictionary("libBaseEventDict")
loadDictionary("libGenEventDict")
loadDictionary("libHepMCRflx")
loadDictionary("libCLHEPRflx")

app.run(10)
