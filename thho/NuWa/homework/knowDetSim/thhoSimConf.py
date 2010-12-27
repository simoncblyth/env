#!/usr/bin/env python

__all__ = ['Configure']

import GaudiKernel.SystemOfUnits as units


class Configure:
    '''
    Do default configuration for genenrator
    '''

    def __init__(self,do_detsim = True, volume = '/dd/Structure/AD/db-oil1'):

        print "Configuring GenTools..."
        import GenTools
        from GenTools.Helpers import Gun
        gtc = GenTools.Configure()
        gunner = Gun()
        gtc.register(gunner)

        # Non-default values for the gen tools:

        gunner.timerator.LifeTime = int(1*units.microsecond)

        gunner.positioner.Strategy = "FullVolume"
        gunner.positioner.Volume = volume
        gunner.positioner.Mode = "Smeared"
        gunner.positioner.Spread = 1*units.meter
        gunner.positioner.Position = [0,0,0]

        gunner.gun.ParticlesPerEvent = 1
        gunner.gun.ParticleName = "e+"
        gunner.gun.Momentum = 4*units.MeV
        gunner.gun.MomentumMode = "Smeared"
        gunner.gun.MomentumSpread = 1*units.MeV
        gunner.gun.DirectionMode = "Uniform"
        gunner.gun.DirectionSpread = 180*units.degree
        gunner.gun.PolarizeMode = "Random"

        # Add the dumper to see log output of kinematics
        dump = GenTools.Dumper()
        print "... GenTools done."



        print "Configuring DetSim with only basic physics list..."
        import DetSim
        from DetSim.Default import physics_list_basic
        detsim = DetSim.Configure(physlist = physics_list_basic)



        detsim.historian(trackSelection="(pdg == -11)",\
                         vertexSelection="(pdg == -11)")

        params = {
            'GD':    "MaterialName == '/dd/Materials/GdDopedLS'",
            'LS':    "MaterialName == '/dd/Materials/LiquidScintillator'",
            }
    
        detsim.unobserver(stats=[
                ["EDepInGdLS", "dE", "%(GD)s"%params],
                ["EDepInLS", "dE", "%(LS)s"%params],
                ])
    





        print '... done.'


        return
    pass
