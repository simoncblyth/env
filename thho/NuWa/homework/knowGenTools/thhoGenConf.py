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
        return
    pass
