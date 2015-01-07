DetSim Check Hits
===================

Where do the hits go next ?
---------------------------

::

    [blyth@ntugrid5 src]$ grep -l Hit *.*
    DsPmtSensDet.cc
    DsPmtSensDet.h
    DsPullEvent.cc
    DsRpcSensDet.cc
    DsRpcSensDet.h


DsPullEvent
-------------

* transfer into SimHitCollection

::

    132         DayaBay::SimHitCollection::hit_container hits;
    133         DayaBay::Detector detector;

    134         DayaBay::SimHitCollection* shc =
    135                        new DayaBay::SimHitCollection(hit_header,detector,hits);


::

    [blyth@ntugrid5 Simulation]$ find . -name '*.cc' -exec grep -l SimHitCollection {} \;
    ./ElecSim/src/components/EsFrontEndAlg.cc
    ./ElecSim/src/components/EsDarkDetectorAlg.cc
    ./ElecSim/src/components/EsPmtEffectPulseTool.cc
    ./ElecSim/src/components/EsIdealPulseTool.cc
    ./ElecSim/src/components/EsTestAlg.cc
    ./MuonHitSim/src/MuonHitSim.cc
    ./Historian/src/DrawHistoryAlg.cc
    ./FastMCProduction/DigitizeAlg/src/DigitizeAlg.cc
    ./FastMCProduction/SimHitSplitSvc/src/lib/SimHitSplitAlgorithm.cc
    ./FastMCProduction/SimHitSplitSvc/src/components/SimHitSplitSvc.cc
    ./FastMCProduction/MixInputSvc/src/lib/MixInputAlgorithm.cc
    ./FastMCProduction/MixInputSvc/src/components/MixInputSvc.cc
    ./FastMCProduction/PreElecSimSvc/src/lib/PreElecAlgorithm.cc
    ./FastMCProduction/PreElecSimSvc/src/components/PreElecSimSvc.cc
    ./FastMCProduction/DigitalizeAlg/src/DigitalizeAlg.cc
    ./DetSim/src/DsPullEvent.cc
    ./Fifteen/ElecSimProc/src/ElecSimProc.cc
    ./Fifteen/DetSimProc/src/DetSimProc.cc
    ./Fifteen/Stage/src/components/Sim15.cc
    [blyth@ntugrid5 Simulation]$ 


