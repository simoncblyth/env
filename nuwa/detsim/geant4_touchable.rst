Geant4 Touchable
==================

G4TouchableHistory 
-------------------

* :google:`G4TouchableHistory UpdateYourself`

* http://www.ge.infn.it/geant4/training/portland/readout.pdf
* http://www.ge.infn.it/geant4/training/roma2004/geom_basics.pdf  

  * good description from G. Cosmo


`G4TouchableHistory` object is a vector of information 
with an entry at each level of a geometrical hierarchy
containing

* copy number
* transformation/rotation to its mother


DsPmtSensDet::SensDetElem  TH to DE
--------------------------------------

`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPmtSensDet.cc`::

    231 const DetectorElement* DsPmtSensDet::SensDetElem(const G4TouchableHistory& hist)
    232 {
    233     const IDetectorElement* idetelem = 0;
    234     int steps=0;
    ...
    241     StatusCode sc =
    242         m_t2de->GetBestDetectorElement(&hist,m_sensorStructures,idetelem,steps);
    ...   
    257     return dynamic_cast<const DetectorElement*>(idetelem);
    258 }


`NuWa-trunk/dybgaudi/Simulation/G4DataHelpers/src/components/TH2DE.h`::

     12 #include "G4DataHelpers/ITouchableToDetectorElement.h"
     13 #include "GaudiAlg/GaudiTool.h"
     14 
     15 #include <string>
     16 #include <vector>
     17 #include <map>
     18 
     19 class TH2DE : public GaudiTool, virtual ITouchableToDetectorElement
     20 {
     21 public:
     22     TH2DE(const std::string& type,
     23          const std::string& name,
     24          const IInterface* parent);
     25     virtual ~TH2DE();
     26 
     27     virtual StatusCode GetBestDetectorElement(const G4TouchableHistory* inHistory,
     28                                               const std::vector<std::string>& /*ignored*/,
     29                                               const IDetectorElement* &outElement,
     30                                               int& outCompatibility);
     31 
     32     virtual StatusCode G4VolumeToDetDesc(const G4VPhysicalVolume* inVol,
     33                                          const IPVolume* &outVol);
     34 
     35     virtual StatusCode ClearCache();
     36 
     37 private:
     38 
     39     typedef std::pair<std::string,std::string> LvPvPair_t;
     40     typedef std::vector<LvPvPair_t> NameHistory_t;
     41     typedef std::vector<G4VPhysicalVolume*> TouchableHistory_t;
     42     typedef std::map<TouchableHistory_t,const IDetectorElement*> THCache_t;
     43     typedef std::map<const G4VPhysicalVolume*,const IPVolume*> PVCache_t;
     44     THCache_t m_THcache;
     45     PVCache_t m_PVcache;
     46 
     47 
     48     const IDetectorElement* FindChildDE(const IDetectorElement* de, NameHistory_t& name_history);
     49     const IDetectorElement* FindDE(const IDetectorElement* de, NameHistory_t& name_history);
     50     int InHistory(const IDetectorElement* de, const NameHistory_t& name_history);
     51 
     52     const IDetectorElement* CheckCache(const G4TouchableHistory* g4hist);
     53 
     54 };







Where to write idmap ? 
------------------------

* `NuWa-trunk/lhcb/Sim/GaussTools/src/Components/GiGaRunActionExport.cpp`
* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/lhcb/trunk/Sim/GaussTools/src/Components/GiGaRunActionExport.cpp

* dependencies mean cannot sensibly use `DsPmtSensDet` from there, but the 
  relevant functionality is 




  * 

Run test with::

    [blyth@belle7 export]$ export.sh MX DayaBay


G4TouchableHistory
-------------------

Ahha, tis upside down. Normally start from "deepest" 
level of a point and work way back up to world.  


`source/geometry/volumes/include/G4TouchableHistory.hh`::

     53 class G4TouchableHistory : public G4VTouchable
     54 {
     55 
     56  public:  // with description
     57 
     58   G4TouchableHistory( const G4NavigationHistory& history );
     59   G4TouchableHistory();
     60     // The default constructor produces a touchable-history of 
     61     // 'zero-depth', ie an "unphysical" and not very unusable one.
     62     // It is for initialisation only.  
     63 
     64  ~G4TouchableHistory();
     65 
     66   inline G4VPhysicalVolume* GetVolume( G4int depth=0 ) const;
     67   inline G4VSolid* GetSolid( G4int depth=0 ) const;
     68   const G4ThreeVector& GetTranslation( G4int depth=0 ) const;
     69   const G4RotationMatrix* GetRotation( G4int depth=0 ) const;
     70 
     71   inline G4int GetReplicaNumber( G4int depth=0 ) const;
     72   inline G4int GetHistoryDepth()  const;
     73   G4int MoveUpHistory( G4int num_levels = 1 );
     74     // Access methods for touchables with history
     75 
     76   void  UpdateYourself( G4VPhysicalVolume*   pPhysVol,
     77                         const G4NavigationHistory* history=0 );
     78     // Update methods for touchables with history
     ..
     80  public:  // without description
     81 
     82   inline const G4NavigationHistory* GetHistory() const;
     83     // Should this method be "deprecated" ?
     84     // it is used now in G4Navigator::LocateGlobalPointAndSetup
     85 
     86   inline void *operator new(size_t);
     87   inline void operator delete(void *aTH);
     88     // Override "new" and "delete" to use "G4Allocator".
     89 
     90  private:
     91 
     92   inline G4int CalculateHistoryIndex( G4int stackDepth ) const;
     93 
     94   G4RotationMatrix frot;
     95   G4ThreeVector ftlate;
     96   G4NavigationHistory fhistory;
     97 };
     98 
     99 #include "G4TouchableHistory.icc"


`source/geometry/volumes/include/G4TouchableHistory.icc`::

     35 inline
     36 void  G4TouchableHistory::UpdateYourself( G4VPhysicalVolume*   pPhysVol,
     37                                     const G4NavigationHistory* pHistory )
     38 {
     39   fhistory = *pHistory;
     40   G4AffineTransform tf(fhistory.GetTopTransform().Inverse());
     41   if( pPhysVol == 0 )
     42   {
     43     // This means that the track has left the World Volume.
     44     // Since the Navigation History does not already reflect this,
     45     // we must correct this problem here.
     46     //
     47     fhistory.SetFirstEntry(pPhysVol);
     48   }
     49   ftlate = tf.NetTranslation();
     50   frot = tf.NetRotation();
     51 }
     52 
     53 inline
     54 G4int G4TouchableHistory::CalculateHistoryIndex( G4int stackDepth ) const
     55 {
     56   return (fhistory.GetDepth()-stackDepth); // was -1
     57 }
     58 
     59 inline
     60 G4VPhysicalVolume* G4TouchableHistory::GetVolume( G4int depth ) const
     61 {
     62   return fhistory.GetVolume(CalculateHistoryIndex(depth));
     63 }
     64 
     65 inline
     66 G4VSolid* G4TouchableHistory::GetSolid( G4int depth ) const
     67 {
     68   return fhistory.GetVolume(CalculateHistoryIndex(depth))
     69                             ->GetLogicalVolume()->GetSolid();
     70 }
     71 
     72 inline
     73 G4int G4TouchableHistory::GetReplicaNumber( G4int depth ) const
     74 {
     75   return fhistory.GetReplicaNo(CalculateHistoryIndex(depth));
     76 }
     77 
     78 inline
     79 G4int G4TouchableHistory::GetHistoryDepth()  const
     80 {
     81   return  fhistory.GetDepth();
     82 }
     ..
     84 inline
     85 G4int G4TouchableHistory::MoveUpHistory( G4int num_levels )
     86 {
     87   G4int maxLevelsMove = fhistory.GetDepth();
     88   G4int minLevelsMove = 0;              // Cannot redescend today!
     89                                         // Soon it will be possible
     90                                         // by adding a data member here
     91                                         //     fCurrentDepth;
     92   if( num_levels > maxLevelsMove )
     93   {
     94     num_levels = maxLevelsMove;
     95   }
     96   else if( num_levels < minLevelsMove )
     97   {
     98     num_levels = minLevelsMove;
     99   }
     00   fhistory.BackLevel( num_levels );
     01 
     02   return num_levels;
     03 }
     04 
     05 inline
     06 const G4NavigationHistory* G4TouchableHistory::GetHistory() const
     07 {
     08   return &fhistory;
     09 }



G4NavigationHistory
--------------------

Attempt to use NavigationHistory is lacking depth. Need to initiate the levels.

* :google:`G4NavigationHistory NewLevel`

::

    delta:geant4.10.00.p01 blyth$ find . -name '*.cc' -exec grep -H NewLevel {} \;
    ./source/geometry/navigation/src/G4Navigator.cc:              fHistory.NewLevel(fBlockedPhysicalVolume, kNormal,
    ./source/geometry/navigation/src/G4Navigator.cc:              fHistory.NewLevel(fBlockedPhysicalVolume, kReplica,
    ./source/geometry/navigation/src/G4Navigator.cc:                fHistory.NewLevel(fBlockedPhysicalVolume, kParameterised,
    ./source/geometry/navigation/src/G4ParameterisedNavigation.cc:      history.NewLevel(pPhysical, kParameterised, replicaNo);
    ./source/geometry/navigation/src/G4RegularNavigation.cc:  history.NewLevel(pPhysical, kParameterised, replicaNo );
    ./source/processes/electromagnetic/dna/management/src/G4ITNavigator.cc:              fHistory.NewLevel(fBlockedPhysicalVolume, kNormal,
    ./source/processes/electromagnetic/dna/management/src/G4ITNavigator.cc:              fHistory.NewLevel(fBlockedPhysicalVolume, kReplica,
    ./source/processes/electromagnetic/dna/management/src/G4ITNavigator.cc:                fHistory.NewLevel(fBlockedPhysicalVolume, kParameterised,
    ./source/processes/hadronic/models/de_excitation/photon_evaporation/src/G4LevelReader.cc:      if(0 < nLevels) { MakeNewLevel(levels); }
    ./source/processes/hadronic/models/de_excitation/photon_evaporation/src/G4LevelReader.cc:    MakeNewLevel(levels);
    ./source/processes/hadronic/models/de_excitation/photon_evaporation/src/G4LevelReader.cc:void G4LevelReader::MakeNewLevel(std::vector<G4NucLevel*>* levels)
    ./source/processes/scoring/src/G4ScoreSplittingProcess.cc:    ptrNavHistory->NewLevel( curPhysicalVol, kParameterised, newVoxelNum );


`source/processes/electromagnetic/dna/management/src/G4ITNavigator.cc`::

     224         if ( fEntering )
     225         {
     226           switch (VolumeType(fBlockedPhysicalVolume))
     227           {
     228             case kNormal:
     229               fHistory.NewLevel(fBlockedPhysicalVolume, kNormal,
     230                                 fBlockedPhysicalVolume->GetCopyNo());
     231               break;
     232             case kReplica:
     233               freplicaNav.ComputeTransformation(fBlockedReplicaNo,
     234                                                 fBlockedPhysicalVolume);
     235               fHistory.NewLevel(fBlockedPhysicalVolume, kReplica,
     236                                 fBlockedReplicaNo);
     237               fBlockedPhysicalVolume->SetCopyNo(fBlockedReplicaNo);
     238               break;


`source/geometry/volumes/include/G4NavigationHistory.hh`::

     58 class G4NavigationHistory
     59 {
     60 
     61  public:  // with description
     62 
     63   friend std::ostream&
     64   operator << (std::ostream &os, const G4NavigationHistory &h);
     65 
     66   G4NavigationHistory();
     67     // Constructor: sizes history lists & resets histories.
     68 
     69   ~G4NavigationHistory();
     70     // Destructor.
     71 
     72   G4NavigationHistory(const G4NavigationHistory &h);
     73     // Copy constructor.
     74 
     75   G4NavigationHistory& operator=(const G4NavigationHistory &h);
     76     // Assignment operator.
     77 
     78   inline void Reset();
     79     // Resets history. It now does clear most entries.
     80     // Level 0 is preserved.
     81 
     82   inline void Clear();
     83     // Clears entries, zeroing transforms, matrices & negating
     84     // replica history.
     85 
     86   inline void SetFirstEntry(G4VPhysicalVolume* pVol);
     87     // Setup initial entry in stack: copies through volume transform & matrix.
     88     // The volume is assumed to be unrotated.
     89 
     90   inline const G4AffineTransform& GetTopTransform() const;
     91     // Returns topmost transform.
     92 
     93   inline const G4AffineTransform* GetPtrTopTransform() const;
     94     // Returns pointer to topmost transform.
     95 
     96   inline G4int GetTopReplicaNo() const;
     97     // Returns topmost replica no record.
     98 
     99   inline EVolume GetTopVolumeType() const;
     00     // Returns topmost volume type.
     01 
     02   inline G4VPhysicalVolume* GetTopVolume() const;
     03     // Returns topmost physical volume pointer.
     04 
     05   inline G4int GetDepth() const;
     ..
     07 
     08   inline G4int GetMaxDepth() const;
     09     // Returns current maximum size of history.
     10     // Note: MaxDepth of 16 mean history entries [0..15] inclusive.
     11 
     12   inline const G4AffineTransform& GetTransform(G4int n) const;
     13     // Returns specified transformation.
     14 
     15   inline G4int GetReplicaNo(G4int n) const;
     16     // Returns specified replica no record.
     17 
     18   inline EVolume GetVolumeType(G4int n) const;
     19     // Returns specified volume type.
     20 
     21   inline G4VPhysicalVolume* GetVolume(G4int n) const;
     22     // Returns specified physical volume pointer.
     23 
     24   inline void NewLevel(G4VPhysicalVolume *pNewMother,
     25                        EVolume vType=kNormal,
     26                        G4int nReplica=-1);
     27     // Changes navigation level to that of the new mother.
     28 
     29   inline void BackLevel();
     30     // Back up one level in history: from mother to grandmother.
     31     // It does not erase history record of current mother.
     32 
     33   inline void BackLevel(G4int n);
     34     // Back up specified number of levels in history.
     35 
     36  private:
     37 
     38   inline void EnlargeHistory();
     39     // Enlarge history if required: increase size by kHistoryStride.
     40     // Note that additional history entries are `dirty' (non zero) apart
     41     // from the volume history.
     42 
     43  private:
     44 
     45 #if defined(WIN32)
     46   std::vector<G4NavigationLevel> fNavHistory;
     47 #else
     48   std::vector<G4NavigationLevel,
     49               G4EnhancedVecAllocator<G4NavigationLevel> > fNavHistory;
     50     // The geometrical tree; uses specialized allocator to optimize memory
     51     // handling, reduce possible fragmentation and use of malloc in MT mode
     52 #endif
     53 
     54   G4int fStackDepth;
     55     // Depth of stack: effectively depth in geometrical tree
     56 
     57 };
     58 
     59 #include "G4NavigationHistory.icc"

`source/geometry/volumes/include/G4NavigationHistory.icc`::

    092 inline
    093 const G4AffineTransform* G4NavigationHistory::GetPtrTopTransform() const
    094 {
    095   return fNavHistory[fStackDepth].GetPtrTransform();
    096 }
    097 
    098 inline
    099 const G4AffineTransform& G4NavigationHistory::GetTopTransform() const
    100 {
    101   return fNavHistory[fStackDepth].GetTransform();
    102 }
    103 
    104 inline
    105 G4int G4NavigationHistory::GetTopReplicaNo() const
    106 {
    107   return fNavHistory[fStackDepth].GetReplicaNo();
    108 }
    109 
    110 inline
    111 EVolume G4NavigationHistory::GetTopVolumeType() const
    112 {
    113   return fNavHistory[fStackDepth].GetVolumeType();
    114 }
    115 
    116 inline
    117 G4VPhysicalVolume* G4NavigationHistory::GetTopVolume() const
    118 {
    119   return fNavHistory[fStackDepth].GetPhysicalVolume();
    120 }
    ...
    191 inline
    192 void G4NavigationHistory::NewLevel( G4VPhysicalVolume *pNewMother,
    193                                     EVolume vType,
    194                                     G4int nReplica )
    195 {
    196   fStackDepth++;
    197   EnlargeHistory();  // Enlarge if required
    198   fNavHistory[fStackDepth] =
    199     G4NavigationLevel( pNewMother,
    200                        fNavHistory[fStackDepth-1].GetTransform(),
    201                        G4AffineTransform(pNewMother->GetRotation(),
    202                        pNewMother->GetTranslation()),
    203                        vType,
    204                        nReplica );
    205   // The constructor computes the new global->local transform
    206 }






`source/geometry/management/include/G4VTouchable.hh`::

     30 // class G4VTouchable
     31 //
     32 // Class description:
     33 //
     34 // Base class for `touchable' objects capable of maintaining an
     35 // association between parts of the geometrical hierarchy (volumes
     36 // &/or solids) and their resultant transformation.
     37 //
     38 // Utilisation:
     39 // -----------
     40 // A touchable is a geometrical volume (solid) which has a unique
     41 // placement in a detector description. It is an abstract base class which 
     42 // can be implemented in a variety of ways. Each way must provide the 
     43 // capabilities of obtaining the transformation and solid that is described
     44 // by the touchable. 
     45 //
     46 // All G4VTouchable implementations must respond to the two following 
     47 // "requests": 
     48 //
     49 //   1) GetTranslation and GetRotation that return the components of the
     50 //      volume's transformation.
     51 //
     52 //   2) GetSolid that gives the solid of this touchable.
     53 //
     54 //
     55 // Additional capabilities are available from implementations with more
     56 // information. These have a default implementation that causes an exception. 
     57 //
     58 // Several capabilities are available from touchables with physical volumes:
     59 //
     60 //   3) GetVolume gives the physical volume.
     61 //
     62 //   4) GetReplicaNumber or GetCopyNumber gives the copy number of the
     63 //      physical volume, either if it is replicated or not.
     64 //
     65 // Touchables that store volume hierarchy (history) have the whole stack of
     66 // parent volumes available. Thus it is possible to add a little more state
     67 // in order to extend its functionality. We add a "pointer" to a level and a
     68 // member function to move the level in this stack. Then calling the above
     69 // member functions for another level, the information for that level can be
     70 // retrieved.  
     71 //
     72 // The top of the history tree is, by convention, the world volume.
     73 //
     74 //   5) GetHistoryDepth gives the depth of the history tree.
     75 //
     76 //   6) GetReplicaNumber/GetCopyNumber, GetVolume, GetTranslation and
     77 //      GetRotation each can be called with a depth argument.
     78 //      They return the value of the respective level of the touchable.
     79 // 
     80 //   7) MoveUpHistory(num) moves the current pointer inside the touchable
     81 //      to point "num" levels up the history tree. Thus, eg, calling 
     82 //      it with num=1 will cause the internal pointer to move to the mother 
     83 //      of the current volume.
     84 //      NOTE: this method MODIFIES the touchable.
     85 //   
     86 // An update method, with different arguments is available, so that the
     87 // information in a touchable can be updated: 
     88 //
     89 //   8) UpdateYourself takes a physical volume pointer and can additionally
     90 //      take a NavigationHistory.
     91 
     92 // History:
     93 // Created: Paul Kent, August 1996

::

    107 class G4VTouchable
    108 {
    109 
    110  public:  // with description
    111 
    112   G4VTouchable();
    113   virtual ~G4VTouchable();
    114     // Constructor and destructor.
    115 
    116   virtual const G4ThreeVector& GetTranslation(G4int depth=0) const = 0;
    117   virtual const G4RotationMatrix* GetRotation(G4int depth=0) const = 0;
    118     // Accessors for translation and rotation.
    119   virtual G4VPhysicalVolume* GetVolume(G4int depth=0) const;
    120   virtual G4VSolid* GetSolid(G4int depth=0) const;
    121     // Accessors for physical volumes and solid.
    122 
    123   virtual G4int GetReplicaNumber(G4int depth=0) const;
    124   inline  G4int GetCopyNumber(G4int depth=0) const;
    125   virtual G4int GetHistoryDepth() const;
    126   virtual G4int MoveUpHistory(G4int num_levels=1);
    127     // Methods for touchables with history.
    128 
    129   virtual void  UpdateYourself(G4VPhysicalVolume* pPhysVol,
    130                    const G4NavigationHistory* history=0);
    131     // Update method.
    132 
    133  public:  // without description
    134 
    135   // virtual void  ResetLevel();
    136 
    137   virtual const G4NavigationHistory* GetHistory() const;
    138     // Should this method be deprecated ?  It is used in G4Navigator!
    139 };
    140 
    141 #include "G4VTouchable.icc"
    142 
    143 #endif

::

    delta:geant4.10.00.p01 blyth$ find . -name '*.cc' -exec grep -H UpdateYourself {} \;
    ./source/geometry/management/src/G4VTouchable.cc:void G4VTouchable::UpdateYourself(G4VPhysicalVolume*,
    ./source/geometry/management/src/G4VTouchable.cc:  G4Exception("G4VTouchable::UpdateYourself()", "GeomMgt0001",
    ./source/geometry/navigation/src/G4MultiNavigator.cc:    touchHist->UpdateYourself( locatedVolume, touchHist->GetHistory() );
    ./source/geometry/navigation/src/G4PathFinder.cc:     touchHist->UpdateYourself( locatedVolume, touchHist->GetHistory() );
    delta:geant4.10.00.p01 blyth$ 



::

    461 G4TouchableHistoryHandle
    462 G4MultiNavigator::CreateTouchableHistoryHandle() const
    463 {
    464   G4Exception( "G4MultiNavigator::CreateTouchableHistoryHandle()",
    465                "GeomNav0001", FatalException,
    466                "Getting a touchable from G4MultiNavigator is not defined.");
    467 
    468   G4TouchableHistory* touchHist;
    469   touchHist= fpNavigator[0] -> CreateTouchableHistory();
    470 
    471   G4VPhysicalVolume* locatedVolume= fLocatedVolume[0];
    472   if( locatedVolume == 0 )
    473   {
    474     // Workaround to ensure that the touchable is fixed !! // TODO: fix
    475     //
    476     touchHist->UpdateYourself( locatedVolume, touchHist->GetHistory() );
    477   }
    478 
    479   return G4TouchableHistoryHandle(touchHist);
    480 }
    481 


::

     764 G4TouchableHandle
     765 G4PathFinder::CreateTouchableHandle( G4int navId ) const
     766 {
     767 #ifdef G4DEBUG_PATHFINDER
     768   if( fVerboseLevel > 2 )
     769   {
     770     G4cout << "G4PathFinder::CreateTouchableHandle : navId = "
     771            << navId << " -- " << GetNavigator(navId) << G4endl;
     772   }
     773 #endif
     774 
     775   G4TouchableHistory* touchHist;
     776   touchHist= GetNavigator(navId) -> CreateTouchableHistory();
     777 
     778   G4VPhysicalVolume* locatedVolume= fLocatedVolume[navId];
     779   if( locatedVolume == 0 )
     780   {
     781      // Workaround to ensure that the touchable is fixed !! // TODO: fix
     782 
     783      touchHist->UpdateYourself( locatedVolume, touchHist->GetHistory() );
     784   }
     785 
     786 #ifdef G4DEBUG_PATHFINDER
     787   if( fVerboseLevel > 2 )
     788   {
     789     G4String VolumeName("None");
     790     if( locatedVolume ) { VolumeName= locatedVolume->GetName(); }
     791     G4cout << " Touchable History created at address " << touchHist
     792            << "  volume = " << locatedVolume << " name= " << VolumeName
     793            << G4endl;
     794   }
     795 #endif
     796 
     797   return G4TouchableHandle(touchHist);
     798 }




`source/geometry/navigation/include/G4Navigator.hh`::

    073 class G4Navigator
    074 {
    075   public:  // with description
    ...
    085   virtual G4double ComputeStep(const G4ThreeVector &pGlobalPoint,
    086                                const G4ThreeVector &pDirection,
    087                                const G4double pCurrentProposedStepLength,
    088                                      G4double  &pNewSafety);
    089     // Calculate the distance to the next boundary intersected
    090     // along the specified NORMALISED vector direction and
    091     // from the specified point in the global coordinate
    092     // system. LocateGlobalPointAndSetup or LocateGlobalPointWithinVolume 
    093     // must have been called with the same global point prior to this call.
    094     // The isotropic distance to the nearest boundary is also
    095     // calculated (usually an underestimate). The current
    096     // proposed Step length is used to avoid intersection
    097     // calculations: if it can be determined that the nearest
    098     // boundary is >pCurrentProposedStepLength away, kInfinity
    099     // is returned together with the computed isotropic safety
    100     // distance. Geometry must be closed.
    ...
    153   inline void LocateGlobalPointAndUpdateTouchableHandle(
    154                 const G4ThreeVector&       position,
    155                 const G4ThreeVector&       direction,
    156                       G4TouchableHandle&   oldTouchableToUpdate,
    157                 const G4bool               RelativeSearch = true);
    158     // First, search the geometrical hierarchy like the above method
    159     // LocateGlobalPointAndSetup(). Then use the volume found and its
    160     // navigation history to update the touchable.
    161 
    162   inline void LocateGlobalPointAndUpdateTouchable(
    163                 const G4ThreeVector&       position,
    164                 const G4ThreeVector&       direction,
    165                       G4VTouchable*        touchableToUpdate,
    166                 const G4bool               RelativeSearch = true);
    167     // First, search the geometrical hierarchy like the above method
    168     // LocateGlobalPointAndSetup(). Then use the volume found and its
    169     // navigation history to update the touchable.
    170 
    171   inline void LocateGlobalPointAndUpdateTouchable(
    172                 const G4ThreeVector&       position,
    173                       G4VTouchable*        touchableToUpdate,
    174                 const G4bool               RelativeSearch = true);
    175     // Same as the method above but missing direction.
    ...
    202   inline G4TouchableHistory* CreateTouchableHistory() const;
    203   inline G4TouchableHistory* CreateTouchableHistory(const G4NavigationHistory*) const;
    204     // `Touchable' creation methods: caller has deletion responsibility.
    205 
    206   virtual G4TouchableHistoryHandle CreateTouchableHistoryHandle() const;
    207     // Returns a reference counted handle to a touchable history.
    ...
    344  protected:  // without description
    345 
    346   G4double kCarTolerance;
    347     // Geometrical tolerance for surface thickness of shapes.
    348 
    349   //
    350   // BEGIN State information
    351   //
    352 
    353   G4NavigationHistory fHistory;
    354     // Transformation and history of the current path
    355     // through the geometrical hierarchy.


`examples/extended/parameterisations/Par01/src/Par01PionShowerModel.cc`::

    199 void Par01PionShowerModel::FillFakeStep(const Par01EnergySpot &eSpot)
    200 {
    201   //-----------------------------------------------------------
    202   // find in which volume the spot is.
    203   //-----------------------------------------------------------
    204   if (!fNaviSetup)
    205     {
    206       fpNavigator->
    207         SetWorldVolume(G4TransportationManager::GetTransportationManager()->
    208                        GetNavigatorForTracking()->GetWorldVolume());
    209       fpNavigator->
    210         LocateGlobalPointAndUpdateTouchableHandle(eSpot.GetPosition(),
    211                                             G4ThreeVector(0.,0.,0.),
    212                                             fTouchableHandle,
    213                                             false);
    214       fNaviSetup = true;
    215     }
    216   else
    217     {
    218       fpNavigator->
    219         LocateGlobalPointAndUpdateTouchableHandle(eSpot.GetPosition(),
    220                                             G4ThreeVector(0.,0.,0.),
    221                                             fTouchableHandle);
    222      }


G4TouchableHistory to DetectorElement
--------------------------------------

::

    ...
    ...
    231 const DetectorElement* DsPmtSensDet::SensDetElem(const G4TouchableHistory& hist)
    232 {
    233     const IDetectorElement* idetelem = 0;
    234     int steps=0;
    235 
    236     if (!hist.GetHistoryDepth()) {
    237         error() << "DsPmtSensDet::SensDetElem given empty touchable history" << endreq;
    238         return 0;
    239     }
    240 
    241     StatusCode sc =
    242         m_t2de->GetBestDetectorElement(&hist,m_sensorStructures,idetelem,steps);
    243     if (sc.isFailure()) {      // verbose warning
    244         warning() << "Failed to find detector element in:\n";
    245         for (size_t ind=0; ind<m_sensorStructures.size(); ++ind) {
    246             warning() << "\t\t" << m_sensorStructures[ind] << "\n";
    247         }
    248         warning() << "\tfor touchable history:\n";
    249         for (int ind=0; ind < hist.GetHistoryDepth(); ++ind) {
    250             warning() << "\t (" << ind << ") "
    251                       << hist.GetVolume(ind)->GetName() << "\n";
    252         }
    253         warning() << endreq;
    254         return 0;
    255     }
    256 
    257     return dynamic_cast<const DetectorElement*>(idetelem);
    258 }
    ...
    ...   //
 



