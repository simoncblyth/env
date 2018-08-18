
offline-vi(){ vi $BASH_SOURCE ; }
offline-env(){ echo -n ; }
offline-usage(){ cat << EOU

Sensitive Detector Setup
---------------------------


init of PMTManagers finds via G4SDManager the PMTSDMgr m_detector
--------------------------------------------------------------------

offline-cls MCP20inchPMTManager::

    96 void
     97 MCP20inchPMTManager::init() {
     98     G4SDManager* SDman = G4SDManager::GetSDMpointer();
     99     m_detector = SDman->FindSensitiveDetector("PMTSDMgr");
    100     assert(m_detector);
    101     // construct


offline-q SetSensitiveDetector::

    epsilon:env blyth$ offline-q SetSensitiveDetector
    ./Simulation/DetSimV2/PMTSim/src/Hello3inchPMTManager.cc:    body_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/Hello3inchPMTManager.cc:    inner1_log->SetSensitiveDetector(m_detector);
    ./Simulation/DetSimV2/PMTSim/src/dyw_PMT_LogicalVolume.cc:  body_log->SetSensitiveDetector(detector);
    ./Simulation/DetSimV2/PMTSim/src/dyw_PMT_LogicalVolume.cc:  inner1_log->SetSensitiveDetector(detector);
    ...

::

    257 void
    258 MCP20inchPMTManager::helper_make_logical_volume()
    259 {
    260     body_log= new G4LogicalVolume
    261         ( body_solid,
    262           GlassMat,
    263           GetName()+"_body_log" );
    264 
    265     m_logical_pmt = new G4LogicalVolume
    266         ( pmt_solid,
    267           GlassMat,
    268           GetName()+"_log" );
    269 
    270     body_log->SetSensitiveDetector(m_detector);
    271 
    272     inner1_log= new G4LogicalVolume
    273         ( inner1_solid,
    274           PMT_Vacuum,
    275           GetName()+"_inner1_log" );
    276     inner1_log->SetSensitiveDetector(m_detector);
    277 
    278     inner2_log= new G4LogicalVolume
    279         ( inner2_solid,
    280           PMT_Vacuum,
    281           GetName()+"_inner2_log" );
    282 


offline-cls dywSD_PMT_v2::

   30 class dywSD_PMT_v2 : public G4VSensitiveDetector, public IToolForSD_PMT


Only two G4VSensitiveDetector subclasses::

    epsilon:offline blyth$ offline-q "public G4VSensitiveDetector"
    ./Simulation/DetSimV2/PMTSim/include/dywSD_PMT.hh:class dywSD_PMT : public G4VSensitiveDetector, public IToolForSD_PMT
    ./Simulation/DetSimV2/PMTSim/include/dywSD_PMT_v2.hh:class dywSD_PMT_v2 : public G4VSensitiveDetector, public IToolForSD_PMT
    epsilon:offline blyth$ 


offline-q G4SDMan::

   Simulation/DetSimV2/PMTSim/src/dywSD_PMT_v2.cc

offline-cls PMTSDMgr::

     46 G4VSensitiveDetector*
     47 PMTSDMgr::getSD()
     48 {   
     49     G4VSensitiveDetector* ifsd = 0;
     50     if (m_pmt_sd == "dywSD_PMT") {
     51         dywSD_PMT* sd = new dywSD_PMT(objName());
     52 
     53         sd->setMergeFlag(m_merge_flag);
     54         sd->setMergeWindows(m_time_window);
     55         ifsd = sd;
     56     } else if (m_pmt_sd == "dywSD_PMT_v2") {
     57         dywSD_PMT_v2* sd = new dywSD_PMT_v2(objName());
     58         sd->setCEMode(m_ce_mode);
     59         // if flat mode
     60         sd->setCEFlatValue(m_ce_flat_value);
     61         // func mode
     62         sd->setCEFunc(m_ce_func, m_ce_func_params);
     63         sd->setMergeFlag(m_merge_flag);
     64         sd->setMergeWindows(m_time_window);
     65         sd->setMerger(m_pmthitmerger);
     66         sd->setHitType(m_hit_type);
     67         // configure the merger
     68         m_pmthitmerger->setMergeFlag(m_merge_flag);
     69         m_pmthitmerger->setTimeWindow(m_time_window);
     70         ifsd = sd;
     71         if (m_disableSD) {
     72             LogInfo << "dywSD_PMT_v2::ProcessHits is disabled now. " << std::endl;
     73             sd->disableSD();
     74         }
     75     }
     76 
     77     return ifsd;
     78 }




EOU
}


offline-cd(){ cd ~/offline ; }
offline-c(){ offline-cd  ; }

offline-f(){ offline-cd ; find . -not -iwholename '*.svn*' -a -type f  ; }  
offline-q(){ offline-cd ; find . -not -iwholename '*.svn*' -a -type f -exec grep -H "${1:-Sensitive}" {} \; ; }



offline-h-(){
   local name=${1:-Hello3inchPMTManager} 
   find . \
       -name $name.hh -o \
       -name $name.h -o \
       -name $name.hpp -o \
       -name $name.rst
}


offline-cls-(){
   local name=${1:-Hello3inchPMTManager} 
   find . \
       -name $name.cc -o \
       -name $name.icc -o \
       -name $name.hh -o \
       -name $name.h -o \
       -name $name.cpp -o \
       -name $name.hpp -o \
       -name $name.rst
}

offline-h(){ offline-cd  ; local cmd="vi $(offline-h- $*)" ; echo $cmd ; eval $cmd ; }
offline-cls(){ offline-cd  ; local cmd="vi $(offline-cls- $*)" ; echo $cmd ; eval $cmd ; }


# dbi(){  && . $ENV_HOME/offline/dbi.bash ; }
 

