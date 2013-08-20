Verbosity 
============

10k lines of  G4ParticleDefinition
-----------------------------------

Get 10k lines of dump ending::

    10255 --- G4ParticleDefinition ---
    10256  Particle Name : xi_c0
    10257  PDG particle code : 4132 [PDG anti-particle code: -4132]
    10258  Mass [GeV/c2] : 2.471     Width : 5.9e-12
    10259  Lifetime [nsec] : 0.000112
    10260  Charge [e]: 0
    10261  Spin : 1/2
    10262  Parity : 1
    10263  Charge conjugation : 0
    10264  Isospin : (I,Iz): (1/2 , -1/2 )
    10265  GParity : 0
    10266  Quark contents     (d,u,s,c,b,t) : 1, 0, 1, 1, 0, 0
    10267  AntiQuark contents               : 0, 0, 0, 0, 0, 0
    10268  Lepton number : 0 Baryon number : 1
    10269  Particle type : baryon [xi_c]
    10270 Decay Table is not defined !!
    10271 List of instantiated particles ============================================
    10272 GenericIon He3 N(1440)+ N(1440)0 N(1520)+ N(1520)0 N(1535)+ N(1535)0 N(1650)+ N(1650)0
    10273 N(1675)+ N(1675)0 N(1680)+ N(1680)0 N(1700)+ N(1700)0 N(1710)+ N(1710)0 N(1720)+ N(1720)0
    10274 N(1900)+ N(1900)0 N(1990)+ N(1990)0 N(2090)+ N(2090)0 N(2190)+ N(2190)0 N(2220)+ N(2220)0


Presumably verboseLevel is 3 or more::

    218 void G4RunManagerKernel::SetPhysics(G4VUserPhysicsList* uPhys)
    219 {
    220   physicsList = uPhys;
    221   G4ParticleTable::GetParticleTable()->SetReadiness();
    222   physicsList->ConstructParticle();
    223   if(verboseLevel>2) G4ParticleTable::GetParticleTable()->DumpTable();
    224   if(verboseLevel>1)
    225   {
    226     G4cout << "List of instantiated particles ============================================" << G4endl;
    227     G4int nPtcl = G4ParticleTable::GetParticleTable()->entries();
    228     for(G4int i=0;i<nPtcl;i++)
    229     {
    230       G4ParticleDefinition* pd = G4ParticleTable::GetParticleTable()->GetParticle(i);
    231       G4cout << pd->GetParticleName() << " ";
    232       if(i%10==9) G4cout << G4endl;
    233     }
    234     G4cout << G4endl;
    235   }
    236 }


::

    [blyth@cms01 source]$ vi run/src/G4RunManagerKernel.cc
    [blyth@cms01 source]$ 
    [blyth@cms01 source]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source
    [blyth@cms01 source]$ find . -name '*.cc' -exec grep -H instantiated {} \;

    [blyth@cms01 source]$ find . -name 'G4RunManagerKernel.*'
    ./run/src/G4RunManagerKernel.cc
    ./run/include/G4RunManagerKernel.hh

::

     175     inline void SetVerboseLevel(G4int vl)


::

     27 // $Id: G4RunManagerKernel.hh,v 1.9 2007/05/30 00:42:09 asaim Exp $
     28 // GEANT4 tag $Name: geant4-09-02 $
     29 //
     30 // 
     31 
     32 // class description:
     33 //
     34 //     This is a class for mandatory control of GEANT4 kernel. 
     35 //     
     36 //     This class is constructed by G4RunManager. If a user uses his/her own
     37 //     class instead of G4RunManager, this class must be instantiated by
     38 //     him/herself at the very beginning of the application and must be deleted
     39 //     at the very end of the application. Also, following methods must be
     40 //     invoked in the proper order.
     41 //       DefineWorldVolume
     42 //       InitializePhysics
     43 //       RunInitialization
     44 //       RunTermination
     45 // 
     46 //     User must provide his/her own classes derived from the following
     47 //     abstract class and register it to the RunManagerKernel. 
     48 //        G4VUserPhysicsList - Particle types, Processes and Cuts
     49 // 
     50 //     G4RunManagerKernel does not have any eveny loop. Handling of events
     51 //     is managed by G4RunManager.



* source:`dybgaudi/trunk/Simulation/DetSim/python/DetSim/SetG4Verbosity.py`

::

     41         from GaussTools.GaussToolsConf import GiGaEventActionCommand
     42         geac = GiGaEventActionCommand("GiGa.GiGaEventActionCommand")
     43         geac.BeginOfEventCommands = [
     44             "/control/verbose "+str(opts.control),
     45             "/run/verbose  "+str(opts.run),
     46             "/event/verbose  "+str(opts.event),
     47             "/tracking/verbose  "+str(opts.tracking),
     48             "/geometry/navigator/verbose  "+str(opts.geometry),
     49             "/process/verbose "+str(opts.process),
     50             "/process/setVerbose " + str(opts.scint) + " Scintillation",
     51             "/process/setVerbose " + str(opts.allProcesses) + " all"
     52             ]
     53         from GiGa.GiGaConf import GiGa
     54         giga = GiGa()
     55         giga.EventAction = geac



Adding to nuwa.py commandline::

     -m 'DetSim.SetG4Verbosity --run=0' 

causing SEGV and not getting rid of the particle dump.

* http://www.phy.bnl.gov/~bviren/git/lhcb/Sim/GiGa/src/component/GiGa.cpp


GiGa/G4RunManager verbosity control
-------------------------------------

Find the G4RunManager::

    [blyth@cms01 NuWa-trunk]$ find . -name '*.cpp' -exec grep -l G4RunManager {} \;
    ./lhcb/Sim/GiGa/src/component/GiGaRunManagerG4RM.cpp
    ./lhcb/Sim/GiGa/src/component/GiGaRunManager.cpp
    ./lhcb/Sim/GiGa/src/component/GiGaRunManagerInterface.cpp
    ./lhcb/Sim/GaussTools/src/genConf/genConf_forG4UAction.cpp

::

     [blyth@cms01 NuWa-trunk]$ vi ./lhcb/Sim/GiGa/src/component/GiGaRunManagerInterface.cpp

     50 StatusCode GiGaRunManager::initialize ()
     51 {
     52   /// initialise the base class GiGaBase
     53   StatusCode sc = GiGaBase::initialize();
     54   if( sc.isFailure() )
     55   { return Error("Could not initialize the base class " , sc ) ; }
     56   ///
     57   G4RunManager::SetVerboseLevel( m_verbosity );
     58   ///


* https://wiki.bnl.gov/dayabay/index.php?title=FAQ:How_to_turn_off_long_particle_listing%3F

The verbose level is passed onto the kernal::

    [blyth@cms01 source]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source
    [blyth@cms01 source]$ vi run/include/G4RunManager.hh 


DetSim GiGa usage
-------------------
::

    [blyth@cms01 dybgaudi]$ find . -name '*.cc' -exec grep -l GiGa {} \;
    ./Simulation/Fifteen/DetSimProc/src/DetSimProc.cc
    ./Simulation/Historian/src/QueriableStepAction.cc
    ./Simulation/GiGaLegacyDyb/src/GgLegacyDybAlg.cc
    ./Simulation/DetSim/src/DsPhysConsEM.cc
    ./Simulation/DetSim/src/DsPullEvent.cc
    ./Simulation/DetSim/src/DsFastMuonStackAction.cc
    ./Simulation/DetSim/src/DsPhysConsHadron.cc
    ./Simulation/DetSim/src/DsRpcSensDet.cc
    ./Simulation/DetSim/src/DsPhysConsGeneral.cc
    ./Simulation/DetSim/src/DsPhysConsOptical.cc
    ./Simulation/DetSim/src/DsPmtSensDet.cc
    ./Simulation/DetSim/src/DsPhysConsElectroNu.cc
    ./Simulation/DetSim/src/DsPushKine.cc
    ./Simulation/DetSim/src/DsOpStackAction.cc
    ./Simulation/DetSim/src/DsPhysConsIon.cc
    ./Simulation/DetSim/src/DsAlg.cc
    [blyth@cms01 dybgaudi]$ 

::

    [blyth@cms01 dybgaudi]$ vi Simulation/Fifteen/DetSimProc/src/DetSimProc.cc   # MuonProphet muons get special treatment



From python control
---------------------


::

    +
    +        #
    +        # tone down verbosity of G4RunManager, 
    +        # skipping the particle table dump for Verbosity less than 3
    +        # https://wiki.bnl.gov/dayabay/index.php?title=FAQ:How_to_turn_off_long_particle_listing%3F
    +        #
    +        from GiGa.GiGaConf import GiGa, GiGaRunManager
    +        giga = GiGa("GiGa")
    +        gigarm = GiGaRunManager("GiGa.GiGaMgr")
    +        gigarm.Verbosity = 2
    +
         



Debug
------

.. literalinclude:: smoking_gun_particle_dump.txt 
