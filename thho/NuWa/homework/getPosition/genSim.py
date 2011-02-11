#!/usr/bin/env python

'''
usage example:

  nuwa.py -n 2 -o output.root genSim 

  positrum is generated in the center of the AD
  The energy deposited in the Gd and GdLS zone is recorded.
  
'''

def configure():
    """Configure this module with neutron energy"""

    # Set run info
    from RunDataSvc.RunDataSvcConf import RunDataSvc
    runDataSvc = RunDataSvc()
    runDataSvc.SimRunType = "Physics"

    # Generator
    import GaudiKernel.SystemOfUnits as units
    from GenTools.Helpers import Gun
    from GenTools.GenToolsConf import GtGunGenTool, GtPositionerTool
    gun = Gun(name = "positrumGun",
              volume = "/dd/Structure/AD/db-oil1",
              gun = GtGunGenTool("gun",
                                 ParticlesPerEvent = 1,
                                 ParticleName = "e+",
                                 Momentum = 4*units.MeV,
                                 MomentumMode = "Fixed",
                                 DirectionMode = "Uniform",
                                 DirectionSpread = 180*units.degree,
                                 PolarizeMode = "Random"),
              positioner = GtPositionerTool("pos",
                                            Mode = "Fixed",
                                            Position = [0,0,0])
              )
    gun.timerator.LifeTime = int(1*units.microsecond)
    import GenTools
    GenTools.Configure().register(gun)
    

    print "Configuring DetSim with only basic physics list..."
    import DetSim
    from DetSim.Default import physics_list_basic
    detsim = DetSim.Configure(physlist = physics_list_basic)



    detsim.historian(trackSelection="(pdg == -11)",\
                     vertexSelection="(pdg == -11)")

    params = {
        'GD':    "MaterialName == '/dd/Materials/GdDopedLS'",
        'LS':    "MaterialName == '/dd/Materials/LiquidScintillator'",
        'track1':"(id==1)",
        'start' :"(start > 0)", 
        'lastvtx': "IsStopping == 1",
        }

    detsim.unobserver(stats=[
            ["EDepInGdLS", "dE", "%(GD)s"%params],
            ["EDepInLS", "dE", "%(LS)s"%params],
            ["pdgId_Trk1","pdg","%(track1)s and %(start)s"%params],
            ["t_Trk1",    "t" , "%(track1)s and %(start)s"%params],
            ["x_Trk1",    "x", "%(track1)s and %(start)s"%params],
            ["y_Trk1",    "y", "%(track1)s and %(start)s"%params],
            ["z_Trk1",    "z", "%(track1)s and %(start)s"%params],
            ["tEnd_Trk1",    "t" , "%(track1)s and %(lastvtx)s"%params],
            ["xEnd_Trk1",    "x", "%(track1)s and %(lastvtx)s"%params],
            ["yEnd_Trk1",    "y", "%(track1)s and %(lastvtx)s"%params],
            ["zEnd_Trk1",    "z", "%(track1)s and %(lastvtx)s"%params],
            ])















def run(app):
    pass

