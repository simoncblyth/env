Profiling Analysis
====================

The examples are from the "base" muon simulation with optical photon 
reweighting optimization switched off.

text
-----

Parse and sort the text columns from googleperftools created with::

    [blyth@belle7 20130820-1318]$ pprof --text $(which python) base.prof > base.prof.txt
    Using local file /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python.
    Using local file base.prof
    Removing _init from all stack traces. 

Meaning of the columns

1. `nsmp` Number of profiling samples in this function
2. `psmp` Percentage of profiling samples in this function
3. `fsmp` Percentage of profiling samples in the functions printed so far
4. `nsub` Number of profiling samples in this function and its callees
5. `psub` Percentage of profiling samples in this function and its callees
6. `name` Function name


psub reverse ordered
~~~~~~~~~~~~~~~~~~~~~~

Reverse ordering by `psub` filtering to exclude boring "global" containers::

    [blyth@belle7 20130820-1318]$ which gproftext.py
    ~/env/bin/gproftext.py
    [blyth@belle7 20130820-1318]$ gproftext.py base.prof.txt -r --filter 'psub > 2 and nsmp > 0' --sort psub --pattern ::
    Total: 721927 samples

    filter : psub > 2 and nsmp > 0      sort : psub    reverse : True

      nsmp       psmp %       fsmp             nsub       psub %         name 
      2.0         0.0 %       99.9             708995.0   98.2 %          DsPullEvent::execute 
      357.0       0.0 %       90.1             705979.0   97.8 %          G4EventManager::DoProcessing 
      1607.0      0.2 %       64.8             705402.0   97.7 %          G4TrackingManager::ProcessOneTrack 
      5959.0      0.8 %       30.3             678848.0   94.0 %          G4SteppingManager::Stepping 
      2027.0      0.3 %       58.7             214554.0   29.7 %          G4SteppingManager::InvokePostStepDoItProcs 
      4888.0      0.7 %       38.4             209391.0   29.0 %          G4SteppingManager::InvokePSDIP 
      3011.0      0.4 %       48.8             205794.0   28.5 %          GiGaStepActionSequence::UserSteppingAction 
      4576.0      0.6 %       39.0             197716.0   27.4 %          G4SteppingManager::DefinePhysicalStepLength 
      14264.0     2.0 %       8.9              162729.0   22.5 %          UnObserverStepAction::UserSteppingAction 
      3746.0      0.5 %       42.4             152794.0   21.2 %          G4Transportation::AlongStepGetPhysicalInteractionLength 
      297.0       0.0 %       92.3             153389.0   21.2 %          G4VProcess::AlongStepGPIL 
      3743.0      0.5 %       42.9             146694.0   20.3 %          G4Navigator::ComputeStep 
      5017.0      0.7 %       36.3             116872.0   16.2 %          G4VoxelNavigation::ComputeStep 
      2995.0      0.4 %       49.2             101018.0   14.0 %          RuleParser::AndRule::select 
      5993.0      0.8 %       29.4             100221.0   13.9 %          RuleParser::EQ_Rule::select 
      8346.0      1.2 %       21.0             85920.0    11.9 %          DsG4OpBoundaryProcess::PostStepDoIt 
      30380.0     4.2 %       4.2              71103.0     9.8 %          std::vector::operator[] 
      3160.0      0.4 %       46.2             54907.0     7.6 %          G4Transportation::PostStepDoIt 
      12168.0     1.7 %       14.3             51787.0     7.2 %          G4VoxelNavigation::LocateNextVoxel 
      545.0       0.1 %       84.9             47110.0     6.5 %          G4Navigator::LocateGlobalPointAndUpdateTouchableHandle 
      7980.0      1.1 %       22.1             37398.0     5.2 %          HistorianStepAction::UserSteppingAction 
      11258.0     1.6 %       15.9             34217.0     4.7 %          std::vector::size 
      1493.0      0.2 %       66.7             32324.0     4.5 %          G4VProcess::PostStepGPIL 
      3583.0      0.5 %       43.4             31541.0     4.4 %          G4Navigator::LocateGlobalPointAndSetup 
      2188.0      0.3 %       56.4             31461.0     4.4 %          G4SteppingManager::InvokeAlongStepDoItProcs 
      259.0       0.0 %       93.5             31815.0     4.4 %          QueriableStepAction::getDetectorId@6e276 
      354.0       0.0 %       90.3             30990.0     4.3 %          TH2DE::GetBestDetectorElement 
      621.0       0.1 %       83.5             29124.0     4.0 %          QueriableStepAction::getDetectorElement 
      1076.0      0.1 %       73.3             27837.0     3.9 %          TH2DE::CheckCache 
      9160.0      1.3 %       18.7             26659.0     3.7 %          G4Step::UpdateTrack 
      19914.0     2.8 %       7.0              23656.0     3.3 %          std::vector::begin 
      5448.0      0.8 %       33.4             23853.0     3.3 %          DsG4Scintillation::PostStepDoIt 
      2617.0      0.4 %       51.1             23982.0     3.3 %          std::map::operator[] 
      2793.0      0.4 %       50.0             22204.0     3.1 %          G4LogicalSkinSurface::GetSurface 
      1779.0      0.2 %       62.7             22660.0     3.1 %          G4VDiscreteProcess::PostStepGetPhysicalInteractionLength 
      2218.0      0.3 %       55.2             20397.0     2.8 %          G4DisplacedSolid::Inside 
      6271.0      0.9 %       28.6             19463.0     2.7 %          GaudiCommon::verbose 
      959.0       0.1 %       75.8             19091.0     2.6 %          RuleParser::OrRule::select 
      496.0       0.1 %       86.4             17776.0     2.5 %          G4Navigator::CreateTouchableHistory 
      7272.0      1.0 %       23.1             16566.0     2.3 %          G4MaterialPropertiesTable::GetProperty 
      3156.0      0.4 %       46.6             16440.0     2.3 %          G4MaterialPropertyVector::GetProperty 
      986.0       0.1 %       74.7             16401.0     2.3 %          G4VoxelNavigation::LevelLocate 
      518.0       0.1 %       85.7             16507.0     2.3 %          std::map::lower_bound 
      4254.0      0.6 %       40.8             15751.0     2.2 %          G4Track::GetVelocity 
      954.0       0.1 %       76.1             15590.0     2.2 %          G4SteppingManager::SetInitialStep 
      13166.0     1.8 %       12.7             14938.0     2.1 %          __gnu_cxx::__normal_iterator::operator+ 
      4364.0      0.6 %       39.6             15493.0     2.1 %          std::_Rb_tree::lower_bound 
      2606.0      0.4 %       51.5             15364.0     2.1 %          G4SubtractionSolid::Inside 
      1343.0      0.2 %       68.8             15318.0     2.1 %          G4TouchableHistory::G4TouchableHistory 


PostStepDoIt to see relevant processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 20130820-1318]$ gproftext.py base.prof.txt -r --pattern 'PostStepDoIt' --filter True
    Total: 721927 samples

    filter : True      sort : psub    reverse : True

      nsmp       psmp %       fsmp             nsub       psub %         name 
      2027.0      0.3 %       58.7             214554.0   29.7 %          G4SteppingManager::InvokePostStepDoItProcs 
      8346.0      1.2 %       21.0             85920.0    11.9 %          DsG4OpBoundaryProcess::PostStepDoIt 
      3160.0      0.4 %       46.2             54907.0     7.6 %          G4Transportation::PostStepDoIt 
      5448.0      0.8 %       33.4             23853.0     3.3 %          DsG4Scintillation::PostStepDoIt 
      191.0       0.0 %       95.3             414.0       0.1 %          G4VDiscreteProcess::PostStepDoIt 
      94.0        0.0 %       97.8             573.0       0.1 %          G4OpAbsorption::PostStepDoIt 
      72.0        0.0 %       98.4             740.0       0.1 %          DsG4Cerenkov::PostStepDoIt 
      184.0       0.0 %       95.5             235.0       0.0 %          G4VRestDiscreteProcess::PostStepDoIt 
      38.0        0.0 %       99.1             135.0       0.0 %          DsG4OpRayleigh::PostStepDoIt 
      8.0         0.0 %       99.8             32.0        0.0 %          G4FastSimulationManagerProcess::PostStepDoIt 
      6.0         0.0 %       99.8             6.0         0.0 %          G4SteppingManager::GetfN2ndariesPostStepDoIt 
      4.0         0.0 %       99.9             87.0        0.0 %          G4VMultipleScattering::PostStepDoIt 
      2.0         0.0 %       99.9             37.0        0.0 %          G4LowEnergyCompton::PostStepDoIt 
      1.0         0.0 %       100.0            9.0         0.0 %          G4LowEnergyPhotoElectric::PostStepDoIt 
      1.0         0.0 %       100.0            9.0         0.0 %          G4LowEnergyRayleigh::PostStepDoIt 
      1.0         0.0 %       100.0            3.0         0.0 %          G4VEnergyLossProcess::PostStepDoIt 
      0.0         0.0 %       100.0            24.0        0.0 %          G4FastSimulationManager::InvokePostStepDoIt 
      0.0         0.0 %       100.0            13.0        0.0 %          G4LowEnergyBremsstrahlung::PostStepDoIt 
      0.0         0.0 %       100.0            46.0        0.0 %          G4LowEnergyIonisation::PostStepDoIt 
    [blyth@belle7 20130820-1318]$ 




SVG node diagram
-------------------

To pan/zoom around the below SVG image open it in a separate tab or window, 
or use :download:`base.prof.svg`


.. image:: base.prof.svg


The SVG was created with the below which incorporates javascript panning code::

    pprof --svg $(which python) base.prof > base.prof.svg


The initial viewport is controlled by setting the transform matrix in the SVG source ~/e/muon_simulation/profiling/base/base.prof.svg::

     246 ]]></script>
     247 <!--g id="viewport" transform="translate(0,0)"-->
         <g id="viewport" transform="matrix(1,0,0,1,0,-3600)">

     248 <g id="viewport" class="graph" transform="scale(1 1) rotate(0) translate(4 5156)">
     249 <title>/data1/env/local/dyb/external/Python/2.7/i686&#45;slc5&#45;gcc41&#45;dbg/bin/python; 721927 samples</title>
     250 <polygon fill="white" stroke="white" points="-4,5 -4,-5156 1596,-5156 1596,5 -4,5"/>




