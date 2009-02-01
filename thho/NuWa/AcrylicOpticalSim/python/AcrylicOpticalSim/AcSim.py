#!/usr/bin/env python

'''
    Acrylic optical study related setup
'''

class IBDPositron:
    '''
        Configure GenTools and DetSim for running positrons only
        from IBD events distributed uniformly in an AD.
    '''

    def __init__(self,histogram_filename = 'IBDpositron.root',volume = "/dd/Structure/AD/far-lso2",seed = "0",nevts = "200"):
        ''' Construct the default configuration.
        '''
        import GaudiKernel.SystemOfUnits as units

        # First GenTools
        ibd = "InverseBeta.exe -seed " + seed + " -n " + nevts + " -eplus_only |"
        from GenTools.Helpers import HepEVT
        he = HepEVT(hepEvtDataSource = ibd)
        he.positioner.Strategy = "FullVolume"
        he.positioner.Volume = volume
        he.positioner.Mode = "Uniform"
        he.positioner.Spread = 2.6*units.meter
        he.positioner.Position = [0,0,2.5*units.meter]
        he.transformer.Volume = volume
        import GenTools
        GenTools.Configure().register(he)
        self.hepevt = he

        self.dumper = GenTools.Dumper()

        # Then DetSim, with smaller than default configuration:
        import DetSim
        detsim = DetSim.Configure(site="far",physlist = DetSim.physics_list_basic)
        params = {
            'start' :"(start > 0)",
            'track1':"(id==1)",
            'track2':"(id==2)",
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
                ["MuonTrkLengthInOws", "dx", "%(Muon)s and %(OWS)s"%params],
                ["MuonTrkLengthInIws", "dx", "%(Muon)s and %(IWS)s"%params],
                ["MuonTrkLengthInLS", "dx", "%(Muon)s and %(LS)s"%params],
                ["MuonTrkLengthInGdLS","dx", "%(Muon)s and %(GD)s"%params],
                ["dEInn","dE", "(pdg!=20022) and %(IWS)s"%params],
                ["dEOut","dE", "(pdg!=20022) and %(OWS)s"%params],
                ["MuonStop", "dx",  "%(Muon)s and %(lastvtx)s"%params],
                
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
                ["xGen",   "lx","%(NeutronTrk)s and %(firstvtx)s"%params],
                ["yGen",   "ly","%(NeutronTrk)s and %(firstvtx)s"%params],
                ["zGen",   "lz","%(NeutronTrk)s and %(firstvtx)s"%params],
        
                ["tCap",   "t","%(NeutronTrk)s and %(lastvtx)s"%params],
                ["xCap",   "lx","%(NeutronTrk)s and %(lastvtx)s"%params],
                ["yCap",   "ly","%(NeutronTrk)s and %(lastvtx)s"%params],
                ["zCap",   "lz","%(NeutronTrk)s and %(lastvtx)s"%params],
        
                ["capTarget", "capTargetZ","%(track1)s and %(lastvtx)s"%params],
                
                # track 1
                ["pdgId_Trk1","pdg","%(track1)s and %(start)s"%params],
                ["t_Trk1",    "t" , "%(track1)s and %(start)s"%params],
                ["x_Trk1",    "lx", "%(track1)s and %(start)s"%params],
                ["y_Trk1",    "ly", "%(track1)s and %(start)s"%params],
                ["z_Trk1",    "lz", "%(track1)s and %(start)s"%params],
                ["tEnd_Trk1",    "t" , "%(track1)s and %(lastvtx)s"%params],
                ["xEnd_Trk1",    "lx", "%(track1)s and %(lastvtx)s"%params],
                ["yEnd_Trk1",    "ly", "%(track1)s and %(lastvtx)s"%params],
                ["zEnd_Trk1",    "lz", "%(track1)s and %(lastvtx)s"%params],
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
                ["x_Trk2",    "lx", "%(track2)s and %(start)s"%params],
                ["y_Trk2",    "ly", "%(track2)s and %(start)s"%params],
                ["z_Trk2",    "lz", "%(track2)s and %(start)s"%params],
                ["tEnd_Trk2",    "t" , "%(track2)s and %(lastvtx)s"%params],
                ["xEnd_Trk2",    "lx", "%(track2)s and %(lastvtx)s"%params],
                ["yEnd_Trk2",    "ly", "%(track2)s and %(lastvtx)s"%params],
                ["zEnd_Trk2",    "lz", "%(track2)s and %(lastvtx)s"%params],
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
        
        self.detsim = detsim

        # Finally, DetSimVali itself
        #from DetSimValidation.DetSimValidationConf import DetSimVali
        #dsv = DetSimVali()
        #dsv.Volume = volume
        #self.detsimvali = dsv
        
        #from Gaudi.Configuration import ApplicationMgr
        #theApp = ApplicationMgr()
        #theApp.TopAlg.append(dsv)

        from GaudiSvc.GaudiSvcConf import THistSvc
        histsvc = THistSvc()
        histsvc.Output =["file1 DATAFILE='%s' OPT='RECREATE' TYP='ROOT' "%histogram_filename]
        return
    pass # end of class IBDPositron

if '__main__' == __name__:
    from Gaudi.Configuration import ApplicationMgr
    theApp = ApplicationMgr()

    from DybPython.Control import main
    nuwa = main()

    import XmlDetDesc
    XmlDetDesc.Configure()
    
    #up = UniformPositron()

    print theApp.TopAlg
    nuwa.run()
