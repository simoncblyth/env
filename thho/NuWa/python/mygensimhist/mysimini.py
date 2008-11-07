#!/usr/bin/env python

#
# Almost the same as the tutorial/Simulation/SimHistsExample/python/__init__.py @ release 1.0.0-rc01
# for initilizing settings of GenTools and DetSim
#

__all__ = ['Configure']

import GaudiKernel.SystemOfUnits as units


class Configure:
    '''
    Do default configuration
    '''

    def __init__(self,do_detsim = True, volume = '/dd/Structure/AD/far-oil1'):
        print "Configuring Geometry..."
        import XmlDetDesc
        XmlDetDesc.Configure()
        print "... geometry done."

        print "Configuring GenTools..."
        import GenTools
        from GenTools.Helpers import Gun
        from GenTools import Dumper
        gtc = GenTools.Configure()
        gunner = Gun()
        gtc.register(gunner)

        # Non-default values for the gen tools:

        gunner.timerator.LifeTime = int(1*units.second)

        gunner.positioner.Strategy = "FullVolume"
        gunner.positioner.Volume = volume
        gunner.positioner.Mode = "Smeared"
        gunner.positioner.Spread = 1*units.meter
        gunner.positioner.Position = [0,0,5.0/16.0*units.meter]

	gunner.gun.ParticlesPerEvent = 100
        gunner.gun.ParticleName = "opticalphoton"
        gunner.gun.Momentum = 2.5*units.eV
        gunner.gun.MomentumMode = "Smeared"
        gunner.gun.MomentumSpread = 1*units.eV
        gunner.gun.DirectionMode = "Fixed"

        # Aim photons to PMT
        from math import sin, cos, pi
        pmt_column_number = 9
        angle = (2*pmt_column_number - 1)*pi/24.0;
        gunner.gun.Direction = [ cos(angle),sin(angle),0 ]

        # Add the dumper to see log output of kinematics
        dump = Dumper()
        print "... GenTools done."

        print 'Configure the THistSvc...'
        from GaudiSvc.GaudiSvcConf import THistSvc
        histsvc = THistSvc()
        histsvc.Output =["file1 DATAFILE='genhists.root' OPT='RECREATE' TYP='ROOT' "]
        print '... THistSvc config done.'

#        Get the app so we can add to TopAlg
#        from Gaudi.Configuration import ApplicationMgr
#        theApp = ApplicationMgr()

#        print 'Configure the GenHists alg...'
#        from SimHistsExample.SimHistsExampleConf import GenHists
#        gh = GenHists()
#        theApp.TopAlg.append(gh)
#        gh.Location = "/Event/Gen/GenHeader"
#        gh.Volume = volume
#        gh.FilePath = "/file1/gen"
#        gh.MaxEnergy = 5
#        gh.EnergyUnits = units.eV
#        print '... GenHists alg configured.'

        if not do_detsim:
            print "Not doing DetSim configuration, only GenTools level."
            return

        print "Configuring DetSim with only basic physics list..."
        import DetSim
        from DetSim.Default import physics_list_basic
        detsim = DetSim.Configure(physlist = physics_list_basic)
        print '... done.'

#        print 'Configure the SimHists alg...'
#        from SimHistsExample.SimHistsExampleConf import SimHists
#        sh = SimHists()
#        theApp.TopAlg.append(sh)
#        sh.Location = "/Event/Sim/SimHeader"
#        sh.FilePath = "/file1/sim"
#        print '... SimHists alg configured.'


        return
    pass
