#!/usr/bin/env python
"""
Usage::

   cd ~/e/geant4/geometry/export
   nuwa.py -G $XMLDETDESCROOT/DDDB/dayabay.xml -n1 -m export 

   ## actually that -G option is not needed as that path is the default 
   ## as observed from DybPython.Control.configure_geometry/XmlDetDesc.Configure

   OR

   ~/e/geant4/geometry/export/export.sh


Based on opw/fmcpmuon.py from David Jaffe, 

* :dybsvn:`source:dybgaudi/trunk/Detector/XmlDetDesc/python/XmlDetDesc/dump_geo.py`


"""
import os, logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def configure(argv=None):
    
    #if argv:
    #    path = argv[0] 
    #else:   
    #    path = '/dd/Geometry'
    #
    #from XmlDetDescChecks.XmlDetDescChecksConf import XddDumpAlg
    #da = XddDumpAlg()
    #da.Paths = [path]

    sitevol = dict(DayaBay="/dd/Structure/Pool/db-ows", Lingao="/dd/Structure/Pool/la-ows", Far="/dd/Structure/Pool/far-ows",)
    site = os.environ.get('G4DAE_EXPORT_SITE','DayaBay')
    volume = sitevol[site]
    log.info("export_all.configure site %s volume %s " % (repr(site),repr(volume)))

    import GaudiKernel.SystemOfUnits as units

    from GenTools.Helpers import HepEVT
    source =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'onemuon')
    hepevt = HepEVT(source)
    hepevt.positioner.Volume = volume
    hepevt.positioner.Mode = "Relative"
    hepevt.positioner.Position = [0,0,0]
    hepevt.timerator.LifeTime = 1*units.second
    hepevt.transformer.Volume = volume
    hepevt.transformer.Offset = [0., 0., (0.042)*units.meter]

    import GenTools
    wallTime = 0
    gt = GenTools.Configure(helper=hepevt)
    gt.generator.TimeStamp = int(wallTime)
    gt.generator.GenName = "Muon"


    # --- skip dumping particle properties ---------------------
    #
    from GiGa.GiGaConf import GiGa, GiGaRunManager
    giga = GiGa("GiGa")
    gigarm = GiGaRunManager("GiGa.GiGaMgr")
    gigarm.Verbosity = 2


    # --- WRL + GDML + DAE geometry export ---------------------------------
    from GaussTools.GaussToolsConf import GiGaRunActionExport, GiGaRunActionCommand, GiGaRunActionSequence
    export = GiGaRunActionExport("GiGa.GiGaRunActionExport")

    giga.RunAction = export
    giga.VisManager = "GiGaVisManager/GiGaVis"

    import DetSim 
    DetSim.Configure(physlist=DetSim.physics_list_basic,site=site)


    
    #   NOT WORKING :  RunSeq fails to do the vis : only the GDML+DAE gets exported
    #   so do at C++ level 
    #
    #wrl  = GiGaRunActionCommand("GiGa.GiGaRunActionCommand")
    #wrl.BeginOfRunCommands = [ 
    #         "/vis/open VRML2FILE",
    #         "/vis/viewer/set/culling global false",
    #         "/vis/viewer/set/culling coveredDaughters false",
    #         "/vis/drawVolume",
    #         "/vis/viewer/flush"
    #] 
    #runseq = GiGaRunActionSequence("GiGa.GiGaRunActionSequence")
    #giga.addTool( runseq , name="RunSeq" )
    #giga.RunSeq.Members += ["GiGaRunActionCommand"]
    #giga.RunSeq.Members += ["GiGaRunActionGDML"]
    #giga.RunAction = "GiGaRunActionSequence/RunSeq"     
    # why so many ways to address things ? Duplication is evil  

    #from Gaudi.Configuration import ApplicationMgr
    #app = ApplicationMgr()
    #app.TopAlg.append(da)


def run(app):
    pass

