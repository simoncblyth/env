#!/usr/bin/env python


volume = "/dd/Structure/Pool/la-iws"

print "THHO TESTING"
print "\tConfiguring geometry\n"
import xmldetdesc
xmldetdesc.config()

from  GaudiPython import AppMgr
from GaudiKernel import SystemOfUnits as units

app = AppMgr()

eds = app.service("EventDataSvc")
eds.ForceLeaves = True

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

#histsvc = app.service("THistSvc")
#histsvc.Output = [ "file1 DATAFILE='gentools.root' OPT='RECREATE' TYP='ROOT'" ]

#print " GtDumper"
gendump = app.algorithm("GenDump")
print "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"



app.EvtSel = "NONE"
print "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
from PyCintex import *
loadDictionary("libBaseEventDict")
loadDictionary("libGenEventDict")
loadDictionary("libHepMCRflx")
loadDictionary("libCLHEPRflx")
print "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
#app.run(1)
print "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"



#loc = gen.Location 
#esv = None
#dgh = None
#
#def test_genrepr():
#    import DybPython.Interactive 
#
#def test_evtsvc():
#    global app
#    global esv
#    esv = app.evtsvc()
#    print str(esv)
#    print repr(esv)
#    assert esv.__class__.__name__ == 'iDataSvc'
#
#def test_header_access():
#    global loc
#    global esv
#    global dgh
#    assert loc == '/Event/Gen/GenHeader'
#    assert esv != None
#    dgh = esv[loc]
#    assert dgh.__class__.__name__ == 'DayaBay::GenHeader'
#
#def test_header_repr():
#    global dgh
#    print str(dgh)
#    print repr(dgh)
#
#
#if '__main__'==__name__:
#    test_genrepr()
#    test_evtsvc()
#    test_header_access()
#    test_header_repr()
#
#
#
#
#
