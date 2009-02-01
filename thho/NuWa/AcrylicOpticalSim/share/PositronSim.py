#!/usr/bin/env python

'''
usage example:

  nuwa.py -A -n 10 -o positron_1MeV_center_n10.root DivingIn.PositronSim

'''

def configure():
    import GenTools
    from GenTools.Helpers import Gun
    import GaudiKernel.SystemOfUnits as units
    mygun = Gun()
    mygun.gun.ParticleName = 'e+'
    mygun.gun.Momentum = 1.0*units.eV
    mygun.setVolume("/dd/Structure/AD/db-oil1")
    mygun.positioner.Position = [0.*units.mm, 0.*units.mm, 0.*units.mm]
    mygun.timerator.LifeTime = 0.020*units.second
    gtc = GenTools.Configure()
    gtc.register(mygun)

    import DetSim
    detsim = DetSim.Configure(physlist = DetSim.physics_list_basic)

    import ElecSim
    elecsim = ElecSim.Configure()

    import TrigSim
    trigsim = TrigSim.Configure()

    import ReadoutSim
    rosim = ReadoutSim.Configure()
    import ReadoutSim.ReadoutSimConf as ROsConf
    ROsConf.ROsReadoutAlg().RoTools=["ROsFecReadoutTool","ROsFeePeakOnlyTool"]
    return
