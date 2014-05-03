Geant4 Navigator
=================

* Hmm, very wide API (an internal class) : **dont touch** 


`source/geometry/navigation/include/G4Navigator.hh`::

    030 // class G4Navigator
    031 //
    032 // Class description:
    033 //
    034 // A class for use by the tracking management, able to obtain/calculate
    035 // dynamic tracking time information such as the distance to the next volume,
    036 // or to find the physical volume containing a given point in the world
    037 // reference system. The navigator maintains a transformation history and
    038 // other information to optimise the tracking time performance.
    039 
    040 // History:
    041 // - Created.                                  Paul Kent,     Jul 95/96
    042 // - Zero step protections                     J.A. / G.C.,   Nov  2004
    043 // - Added check mode                          G. Cosmo,      Mar  2004
    044 // - Made Navigator Abstract                   G. Cosmo,      Nov  2003
    045 // *********************************************************************
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
    101 
    102   G4double CheckNextStep(const G4ThreeVector &pGlobalPoint,
    103                          const G4ThreeVector &pDirection,
    104                          const G4double pCurrentProposedStepLength,
    105                                G4double &pNewSafety);
    106     // Same as above, but do not disturb the state of the Navigator.
    107 
    108   virtual
    109   G4VPhysicalVolume* ResetHierarchyAndLocate(const G4ThreeVector &point,
    110                                              const G4ThreeVector &direction,
    111                                              const G4TouchableHistory &h);
    112 
    113     // Resets the geometrical hierarchy and search for the volumes deepest
    114     // in the hierarchy containing the point in the global coordinate space.
    115     // The direction is used to check if a volume is entered.
    116     // The search begin is the geometrical hierarchy at the location of the
    117     // last located point, or the endpoint of the previous Step if
    118     // SetGeometricallyLimitedStep() has been called immediately before.
    119     // 
    120     // Important Note: In order to call this the geometry MUST be closed.
    121 
    122   virtual
    123   G4VPhysicalVolume* LocateGlobalPointAndSetup(const G4ThreeVector& point,
    124                                              const G4ThreeVector* direction=0,
    125                                              const G4bool pRelativeSearch=true,
    126                                              const G4bool ignoreDirection=true);
    127     // Search the geometrical hierarchy for the volumes deepest in the hierarchy
    128     // containing the point in the global coordinate space. Two main cases are:
    129     //  i) If pRelativeSearch=false it makes use of no previous/state
    130     //     information. Returns the physical volume containing the point, 
    131     //     with all previous mothers correctly set up.
    132     // ii) If pRelativeSearch is set to true, the search begin is the
    133     //     geometrical hierarchy at the location of the last located point,
    134     //     or the endpoint of the previous Step if SetGeometricallyLimitedStep()
    135     //     has been called immediately before.
    136     // The direction is used (to check if a volume is entered) if either
    137     //   - the argument ignoreDirection is false, or
    138     //   - the Navigator has determined that it is on an edge shared by two or
    139     //     more volumes.  (This is state information.)
    140     // 
    141     // Important Note: In order to call this the geometry MUST be closed.
    142 
    143   virtual
    144   void LocateGlobalPointWithinVolume(const G4ThreeVector& position);
    145     // Notify the Navigator that a track has moved to the new Global point
    146     // 'position', that is known to be within the current safety.
    147     // No check is performed to ensure that it is within  the volume. 
    148     // This method can be called instead of LocateGlobalPointAndSetup ONLY if
    149     // the caller is certain that the new global point (position) is inside the
    150     // same volume as the previous position.  Usually this can be guaranteed
    151     // only if the point is within safety.

