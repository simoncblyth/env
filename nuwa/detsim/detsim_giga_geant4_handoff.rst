DetSim GiGa Geant4 Handoff
============================

Identity Heist Approach is becoming clearer
----------------------------------------------

#. perform a recursive Geant4 traverse, for every step of the traverse

   * form G4TouchableHistory
   * look for any associated DetectorElement, and any PmtId
   * compose identifiers that match the COLLADA identifies

   * maybe improve the COLLADA identifiers, to reflect the heirarchy  
     and avoid the daenode kludge uniqing  


How to get IDMAP through to Chroma ?
--------------------------------------

Approaches:

#. Look for Geant4 native way to lodge channel identifiers
   against pieces of geometry (look at GDML Aux ?)
   then can read this in the COLLADA export
   
   * advantage is that exporter can then be general purpose, 
     if Geant4 user has labelled the geometry in the prescribed manner

   * there might be no way to do this in Geant4 (or older Geant4) ? 
     Otherwise why all that DetDesc mess

#. modify exporter to accept a vector of identifiers the same length as 
   the traversal volumes, represent that in COLLADA extra node, 
   parse in daenode

   * feels a bit kludgy, but simple
   * advantage: avoids juggling .dae and .idmap 

#. shortcut: dont bother with passing into COLLADA representation, instead 
   on loading a .dae in daenode look for companion .idmap and parse 
   it if present 

   * even more kludgy, but fastest, have to juggle .dae and .idmap
   * advantage: no need to change exporter


Shortcut approach
~~~~~~~~~~~~~~~~~~~

* https://bitbucket.org/simoncblyth/env/commits/a2d79f476349b62b7c26d5e05af42b840217944a

Seems to be swallowed by Chroma OK, but not getting any SURFACE_DETECT.

Model mismatch ?

* Chroma expecting sensitive detectors (PMT Cathodes) modelled as surfaces ?


XML representation of IDMAP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. listing all volumes, with most identifiers being zero seems a bit dumb
   so need to use physvolrefs : makes non-trivial

::

    <channelid name="12345" >
        <physvolref ref="__dd__Geometry__AdDetails__...bitofpmtreference...0xc266468"/>  
    </channelid>



IDMAP observations Experience
--------------------------------

*  http://dayabay.ihep.ac.cn/tracs/dybsvn/changeset/23245

#. entire PMT assembly labelled under the DetectorElement PMTID, not just Cathode
#. manual check of pvnames/index of 10 random volumes matches against daeserver and `g4daeview.py`

   * http://belle7.nuu.edu.tw/dae/tree/5981.html 


`/data1/env/local/env/geant4/geometry/export/DayaBay_MX_20140916-2050/g4_00.idmap`::

    # GiGaRunActionExport::WriteIdMap fields: index,pmtid,pmtid(hex),pvname  npv:12230
    0 0 0  Universe
    1 0 0  /dd/Structure/Sites/db-rock
    2 0 0  /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop
    3 0 0  /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover
    4 0 0  /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1
    5 0 0  /dd/Geometry/RPC/lvRPCMod#pvRPCFoam
    6 0 0  /dd/Geometry/RPC/lvRPCFoam#pvBarCham14Array#pvBarCham14ArrayOne:1#pvBarCham14Unit
    7 0 0  /dd/Geometry/RPC/lvRPCBarCham14#pvRPCGasgap14
    8 0 0  /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit
    9 0 0  /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit
    10 0 0  /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:3#pvStrip14Unit
    ...
    3192 0 0  /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:5#IavTopRibRot
    3193 0 0  /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:6#IavTopRibRot
    3194 0 0  /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:7#IavTopRibRot
    3195 0 0  /dd/Geometry/AD/lvOAV#pvOcrGdsLsoInOav
    3196 0 0  /dd/Geometry/AdDetails/lvOcrGdsLsoInOav#pvOcrGdsTfbInOav
    3197 0 0  /dd/Geometry/AdDetails/lvOcrGdsTfbInOav#pvOcrGdsInOav
    3198 0 0  /dd/Geometry/AD/lvOAV#pvOcrCalLsoInOav
    3199 16843009 1010101  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt
    3200 16843009 1010101  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
    3201 16843009 1010101  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
    3202 16843009 1010101  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom
    3203 16843009 1010101  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode
    3204 0 0  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar
    3205 16843010 1010102  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmt
    3206 16843010 1010102  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
    3207 16843010 1010102  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
    3208 16843010 1010102  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom
    3209 16843010 1010102  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode
    3210 0 0  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmtCollar
    3211 16843011 1010103  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:3#pvAdPmtUnit#pvAdPmt
    3212 16843011 1010103  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
    3213 16843011 1010103  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
    3214 16843011 1010103  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom
    3215 16843011 1010103  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode
    3216 0 0  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:3#pvAdPmtUnit#pvAdPmtCollar
    ...
    4338 0 0  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:8#pvAdPmtInRing:22#pvAdPmtUnit#pvAdPmtCollar
    4339 16844823 1010817  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:8#pvAdPmtInRing:23#pvAdPmtUnit#pvAdPmt
    4340 16844823 1010817  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
    4341 16844823 1010817  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
    4342 16844823 1010817  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom
    4343 16844823 1010817  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode
    4344 0 0  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:8#pvAdPmtInRing:23#pvAdPmtUnit#pvAdPmtCollar
    4345 16844824 1010818  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:8#pvAdPmtInRing:24#pvAdPmtUnit#pvAdPmt
    4346 16844824 1010818  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
    4347 16844824 1010818  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
    4348 16844824 1010818  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom
    4349 16844824 1010818  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode
    4350 0 0  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:8#pvAdPmtInRing:24#pvAdPmtUnit#pvAdPmtCollar
    4351 16842753 1010001  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAd2inPmt:1#pvHeadonPmtAssy
    4352 16842753 1010001  /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtGlass
    4353 16842753 1010001  /dd/Geometry/PMT/lvHeadonPmtGlass#pvHeadonPmtVacuum
    4354 16842753 1010001  /dd/Geometry/PMT/lvHeadonPmtVacuum#pvHeadonPmtCathode
    4355 16842753 1010001  /dd/Geometry/PMT/lvHeadonPmtVacuum#pvHeadonPmtBehindCathode
    4356 16842753 1010001  /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtBase
    4357 0 0  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAd2inPmt:1#pvHeadonPmtMount
    4358 16842754 1010002  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAd2inPmt:2#pvHeadonPmtAssy
    4359 16842754 1010002  /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtGlass
    4360 16842754 1010002  /dd/Geometry/PMT/lvHeadonPmtGlass#pvHeadonPmtVacuum
    4361 16842754 1010002  /dd/Geometry/PMT/lvHeadonPmtVacuum#pvHeadonPmtCathode
    4362 16842754 1010002  /dd/Geometry/PMT/lvHeadonPmtVacuum#pvHeadonPmtBehindCathode
    4363 16842754 1010002  /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtBase
    4364 0 0  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAd2inPmt:2#pvHeadonPmtMount
    4365 16842755 1010003  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAd2inPmt:3#pvHeadonPmtAssy
    4366 16842755 1010003  /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtGlass
    4367 16842755 1010003  /dd/Geometry/PMT/lvHeadonPmtGlass#pvHeadonPmtVacuum
    4368 16842755 1010003  /dd/Geometry/PMT/lvHeadonPmtVacuum#pvHeadonPmtCathode
    4369 16842755 1010003  /dd/Geometry/PMT/lvHeadonPmtVacuum#pvHeadonPmtBehindCathode
    4370 16842755 1010003  /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtBase
    ...
    11405 17172487 1060807  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:7#pvVetoPmtUnit#pvPmtMount#pvMountRib3s#pvMountRib3s:1#pvMountRib3unit
    11406 17172487 1060807  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:7#pvVetoPmtUnit#pvPmtMount#pvMountRib3s#pvMountRib3s:2#pvMountRib3unit
    11407 17172487 1060807  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:7#pvVetoPmtUnit#pvPmtMount#pvMountRib3s#pvMountRib3s:3#pvMountRib3unit
    11408 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtHemi
    11409 17172488 1060808  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
    11410 17172488 1060808  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
    11411 17172488 1060808  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom
    11412 17172488 1060808  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode
    11413 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvPmtTopRing
    11414 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvPmtBaseRing
    11415 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvMountRib1s#pvMountRib1s:1#pvMountRib1unit
    11416 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvMountRib1s#pvMountRib1s:2#pvMountRib1unit
    11417 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvMountRib1s#pvMountRib1s:3#pvMountRib1unit
    11418 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvPmtTee
    11419 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvMountRib2s#pvMountRib2s:1#pvMountRib2unit
    11420 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvMountRib2s#pvMountRib2s:2#pvMountRib2unit
    11421 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvMountRib2s#pvMountRib2s:3#pvMountRib2unit
    11422 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvMountRib3s#pvMountRib3s:1#pvMountRib3unit
    11423 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvMountRib3s#pvMountRib3s:2#pvMountRib3unit
    11424 17172488 1060808  /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFaceout#pvNearOutFaceoutWall8#pvNearOutFaceoutWall8:8#pvVetoPmtUnit#pvPmtMount#pvMountRib3s#pvMountRib3s:3#pvMountRib3unit
    11425 0 0  /dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:1#pvLegInOWSUnit
    11426 0 0  /dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:2#pvLegInOWSUnit
    11427 0 0  /dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:3#pvLegInOWSUnit
    ...
    12224 0 0  /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab4
    12225 0 0  /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab5
    12226 0 0  /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab6
    12227 0 0  /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab7
    12228 0 0  /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab8
    12229 0 0  /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab9




External Propagation Approaches
--------------------------------

collect and kill tracks, create hit collections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. in `DsChromaStackAction::ClassifyNewTrack` 

   * collect optical photon G4Track info into ChromaPhotonList member of StackAction
     and kill the G4Tracks, avoiding memory expense

#. in `DsChromaStackAction::NewStage` 

   * send the ChromaPhotonList off for external propagation, 
   * wait for response whilst GPU propagation proceeds 
     (need to arrange for standard PMTID to get thru to the GPU)
   * from the propagated photon response construct hit collection

#. in `DsChromaPullEvent::execute` OR  `DsPullEvent::execute` (if standard one can be used)

   * access the hit collection and send the hits along 
     as normally as possible


How to associate pmtid with triangles in GPU realm ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Triangles are associated to solids.

* Its looking likely that the DE parameters (PmtId) never get 
  down to Geant4 level, living only at the Gaudi level in the form
  of a pv tree/name association to the parameter.  

* Need list of sensitive detector identifiers, of length equal to 
  the list of solids (or could extend to all triangles). 
  With placeholder zeros for non-sensitive solids


How are pmtid associated with bits of geometry ?  Using Detector Element
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 Detector]$ pwd
    /data1/env/local/dyb/NuWa-trunk/dybgaudi/Detector
    [blyth@belle7 Detector]$ find . -name '*.cc' -exec grep -l pmtid {} \;
    ./DetHelpers/src/components/PmtGeomInfo.cc
    ./DetHelpers/src/components/PmtGeomInfoSvc.cc


`NuWa-trunk/dybgaudi/Detector/DetHelpers/src/components/PmtGeomInfo.h`::

     18 class PmtGeomInfo : public virtual IPmtGeomInfo
     19 {
     20 public:
     21     PmtGeomInfo(unsigned int pmtid,
     22                 IDetectorElement* me,
     23                 IDetectorElement* parent);
     24     virtual ~PmtGeomInfo();
     25 
     26     /// Return the fully qualified packed ID
     27     virtual unsigned int pmtid() const;
     28 
     29     /// Return the position in world coordinates
     30     virtual const CLHEP::Hep3Vector& globalPosition() const;
     31 
     32     /// Return the position in the natural coordinate system for doing
     33     /// reconstruction (AD local or Pool local coordinates)
     34     virtual const CLHEP::Hep3Vector& localPosition() const;
     35 
     36     /// Return the direction normal to the PMT face in the global
     37     /// coordinate system
     38     virtual const CLHEP::Hep3Vector& globalDirection() const;
     39 
     40     /// Return the direction normal to the PMT face in the natural
     41     /// coordinate system for doing reconstruction (AD local or Pool
     42     /// local coordinates)
     43     virtual const CLHEP::Hep3Vector& localDirection() const;
     44 
     45     /// Return coresponding DetectorElement
     46     virtual const IDetectorElement& detectorElement() const;
     47 
     48     /// Return DE for detector volume containing PMT
     49     virtual const IDetectorElement& parentDetector() const;
     50 
     51 private:
     52     unsigned int m_id;
     53     IDetectorElement* m_me;
     54     IDetectorElement* m_parent;
     55 
     56     mutable CLHEP::Hep3Vector *m_gp, *m_lp, *m_gd, *m_ld;
     57 
     58 };
     59 

`NuWa-trunk/dybgaudi/Detector/DetHelpers/src/components/PmtGeomInfo.cc`::

     36 const Hep3Vector& PmtGeomInfo::globalPosition() const
     37 {
     38     if (m_gp) return *m_gp;
     39 
     40     Gaudi::XYZPoint zero(0,0,0);
     41     Gaudi::XYZPoint gp = m_me->geometry()->toGlobal(zero);
     42 
     43     m_gp = new Hep3Vector(gp.x(),gp.y(),gp.z());
     44     return *m_gp;
     45 }


`NuWa-trunk/dybgaudi/Detector/DetHelpers/src/components/PmtGeomInfoSvc.h`::

     28 class PmtGeomInfoSvc : public Service,
     29                        virtual public IPmtGeomInfoSvc
     30 {
     31 public:
     32     // Service interface
     33     PmtGeomInfoSvc(const std::string& name, ISvcLocator *svc);
     34     ~PmtGeomInfoSvc();
     35     virtual StatusCode initialize();
     36     virtual StatusCode reinitialize();
     37     virtual StatusCode finalize();
     38     virtual StatusCode queryInterface(const InterfaceID& riid,
     39                                       void** ppvInterface);
     40 
     41     /// IPmtGeomInfoSvc interface
     42 
     43     /// Look up by TDS path of detector element 
     44     IPmtGeomInfo* get(std::string structure_path);
     45 
     46     /// Look up by DetectorElement
     47     IPmtGeomInfo* get(IDetectorElement* pmtde);
     48 
     49     /// Look up by fully qualified PMT id (see Conventions/Detectors.h)
     50     IPmtGeomInfo* get(unsigned int pmtid);



`NuWa-trunk/dybgaudi/Simulation/DetSim/python/DetSim/Default.py`::

     16 class Configure:
     17     '''
     18     Do default DetSim configuration.
     19     '''
     20 
     21     # Available geometry broken up by site
     22     giga_far_items = [
     23         "/dd/Structure/Sites/far-rock",
     24         "/dd/Geometry/AdDetails/AdSurfacesAll",
     25         "/dd/Geometry/AdDetails/AdSurfacesFar",
     26         "/dd/Geometry/PoolDetails/FarPoolSurfaces",
     27         "/dd/Geometry/PoolDetails/PoolSurfacesAll",
     28         ]
     29     giga_dayabay_items = [
     30         "/dd/Structure/Sites/db-rock",
     31         "/dd/Geometry/AdDetails/AdSurfacesAll",
     32         "/dd/Geometry/AdDetails/AdSurfacesNear",
     33         "/dd/Geometry/PoolDetails/NearPoolSurfaces",
     34         "/dd/Geometry/PoolDetails/PoolSurfacesAll",
     35         ]
     36     giga_lingao_items = [
     37         "/dd/Structure/Sites/la-rock",
     38         "/dd/Geometry/AdDetails/AdSurfacesAll",
     39         "/dd/Geometry/AdDetails/AdSurfacesNear",
     40         "/dd/Geometry/PoolDetails/NearPoolSurfaces",
     41         "/dd/Geometry/PoolDetails/PoolSurfacesAll",
     42         ]
     43 
     44     def __init__(self,site="far,dayabay,lingao",
     45                  physlist = physics_list_basic+physics_list_nuclear,
     46                  use_push_algs = True,
     47                  use_sim_subseq=False ):
     ..
     68         physics_list = GiGaPhysListModular("GiGa.GiGaPhysListModular")
     69         physics_list.CutForElectron = 100*units.micrometer
     70         physics_list.CutForPositron = 100*units.micrometer
     71         physics_list.CutForGamma = 1*units.millimeter
     72         physics_list.PhysicsConstructors = physlist
     73         self.physics_list = physics_list
     74 
     75         from GiGa.GiGaConf import GiGa
     76         giga = GiGa()
     77         giga.PhysicsList = physics_list
     78 
     79         # Start empty step action sequence to hold historian/unobserver
     80         from GaussTools.GaussToolsConf import GiGaStepActionSequence
     81         sa = GiGaStepActionSequence('GiGa.GiGaStepActionSequence')
     82         giga.SteppingAction = sa
     83 
     84         self.giga = giga
     85 
     86         # Tell GiGa the size of the world.
     87         # Set default world material to be vacuum to speed propagation of
     88         # particles in regions of little interest.
     89         from GiGaCnv.GiGaCnvConf import GiGaGeo
     90         giga_geom = GiGaGeo()
     91         giga_geom.XsizeOfWorldVolume = 2.4*units.kilometer
     92         giga_geom.YsizeOfWorldVolume = 2.4*units.kilometer
     93         giga_geom.ZsizeOfWorldVolume = 2.4*units.kilometer
     94         giga_geom.WorldMaterial = "/dd/Materials/Vacuum"
     95         self.gigageo = giga_geom
     96 
     97         # Set up for telling GiGa what geometry to use, but don't
     98         # actually set that.
     99         from GaussTools.GaussToolsConf import GiGaInputStream
     00         giga_items = GiGaInputStream()
     01         giga_items.ExecuteOnce = True
     02         giga_items.ConversionSvcName = "GiGaGeo"
     03         giga_items.DataProviderSvcName = "DetectorDataSvc"
     04         giga_items.StreamItems = [ ]
     05         site = site.lower()
     06         if "far" in site:
     07             giga_items.StreamItems += self.giga_far_items
     08         if "dayabay" in site:
     09             giga_items.StreamItems += self.giga_dayabay_items
     10         if "lingao" in site:
     11             giga_items.StreamItems += self.giga_lingao_items
     12         self.giga_items = giga_items
     13
     14         # Make sequencer alg to run all this stuff as subalgs
     15         from GaudiAlg.GaudiAlgConf import GaudiSequencer
     16         giga_sequence = GaudiSequencer()
     17         giga_sequence.Members = [ self.giga_items ]
     18         self.giga_sequence=giga_sequence
     19         if use_push_algs:
     20             # DetSim's algs
     21             from DetSim.DetSimConf import DsPushKine, DsPullEvent
     22             self.detsim_push_kine = DsPushKine()
     23             self.detsim_pull_event = DsPullEvent()
     24             giga_sequence.Members += [self.detsim_push_kine,
     25                                       self.detsim_pull_event]
     26             pass
     27 
     28         if not use_sim_subseq:
     29             from Gaudi.Configuration import ApplicationMgr
     30             theApp = ApplicationMgr()
     31             theApp.TopAlg.append(giga_sequence)
     32 
     33         return




`NuWa-trunk/dybgaudi/Detector/DetHelpers/src/components/PmtGeomInfoSvc.cc`::

     16 PmtGeomInfoSvc::PmtGeomInfoSvc(const std::string& name, ISvcLocator *svc)
     17     : Service(name,svc)
     18     , m_detSvc(0)
     19     , m_detector(0)
     20 {
     21     declareProperty("SiteIdUserParameter",m_SiteIdUserParameter="SiteID",
     22                     "Name of the user parameter attached to Site detector "
     23                     "elements that gives the packed Site ID number.");
     24     declareProperty("DetectorIdUserParameter",
     25                     m_DetectorIdUserParameter="DetectorID",
     26                     "Name of the user parameter attached to Detector "
     27                     "detector elements that gives the fully qualified "
     28                     "packed Detector ID.");
     29     declareProperty("PmtIdUserParameter",m_PmtIdUserParameter="PmtID",
     30                     "Name of the user parameter attached to PMT detector "
     31                     "elements that gives the fully qualified packed PMT ID");
     32     std::vector<std::string> defaultStreamItems;
     33     defaultStreamItems.push_back("/dd/Structure/DayaBay");
     34     declareProperty("StreamItems",m_StreamItems=defaultStreamItems,
     35                     "List of top level Detector Elements.");
     36     declareProperty("EnableSabGeometry",m_enableSabGeometry=true,
     37                     "Workaround to allow for non-existent SAB geometry");
     38 }
     ..
     ..
     ..     Pull topDE out of DetectorDataSvc for each of the StreamItems
     ..
     46 StatusCode PmtGeomInfoSvc::initialize()
     47 {
     48     this->Service::initialize();
     49 
     50     MsgStream msg(msgSvc(),name());
     51     msg << MSG::DEBUG << "PmtGeomInfoSvc::initialize()" << endreq;
     52 
     53     StatusCode sc = service("DetectorDataSvc",m_detSvc,true);
     54     if (sc.isFailure()) return sc;
     55 
     56     msg << MSG::DEBUG << "Using IDs:"
     57         << " site: " << m_SiteIdUserParameter
     58         << " det: " << m_DetectorIdUserParameter
     59         << " pmt: " << m_PmtIdUserParameter
     60         << endreq;
     61 
     62     if (! m_StreamItems.size()) {
     63         msg << MSG::WARNING << "did not get any StreamItems, can not lookup PMTs" << endreq;
     64         return StatusCode::FAILURE;
     65     }
     66     for (size_t ind=0; ind<m_StreamItems.size(); ++ind) {
     67         string dename = m_StreamItems[ind];
     68         SmartDataPtr<IDetectorElement> obj(m_detSvc,dename);
     69         if (!obj) {
     70             MsgStream msg(msgSvc(),name());
     71             msg << MSG::WARNING << "Failed to get top Detector Element: \""
     72                 << dename << "\", skipping" << endreq;
     73             sc = StatusCode::FAILURE;
     74             continue;
     75         }
     76         msg << MSG::DEBUG << "Adding top level Detector Element: \""
     77             << dename << endreq;
     78         m_topDEs.push_back(obj);
     79     }
     80     return sc;
     81 }

Hmm, where do the DE parameters get set ?::

    201 IPmtGeomInfo* PmtGeomInfoSvc::find(unsigned int pmtid, IDetectorElement* de)
    202 {
    203     //nomsg MsgStream msg(msgSvc(),name());
    204     //nomsg msg << MSG::DEBUG << "PmtGeomInfoSvc::find(int "<<(void*)pmtid        <<","<<de->name()<<")" << endreq;
    205 
    206     const ParamValidDataObject* params = de->params();
    207     // Check if DE is a PMT.  If current DE has a PmtID we are done
    208     // for good or bad.
    209     if (de->params()->exists(m_PmtIdUserParameter)) {
    210         unsigned int this_pmtid = (unsigned int)(params->param<int>(m_PmtIdUserParameter));
    211         if (pmtid == this_pmtid) {
    212             //nomsg msg << MSG::DEBUG << "found PMT ID " << (void*)pmtid << endreq;
    213             return this->add(pmtid,"",de);
    214         }
    215         //nomsg msg << MSG::DEBUG << "got PMT ID but wrong one "            << (void*)this_pmtid << " != " << (void*)pmtid << endreq;
    216         return 0;
    217     }


Too many PMTs to be manual, must be generated::

    [blyth@belle7 XmlDetDescGen]$ find . -name '*.py' -exec grep -H pmtid {} \;
    ./AdPmtStructure/gen.py:    def pmtid(self,site,adn,icol,iring):
    ./AdPmtStructure/gen.py:                    pmtid = self.pmtid( self.siteid, adn,icol,iring)
    ./AdPmtStructure/gen.py:                        'pmtid':pmtid
    ./AdPmtStructure/gen.py:                    de.refs = [UserParameter("PmtID","int",['0x%x'%pmtid],desc="Packed PMT ID")]
    ./AdPmtStructure/gen.py:                pmtid = self.pmtid( self.siteid, adn,icol,iring)
    ./AdPmtStructure/gen.py:                        'pmtid':pmtid
    ./AdPmtStructure/gen.py:                de.refs = [UserParameter("PmtID","int",['0x%x'%pmtid],desc="Packed PMT ID")]
    ./PoolPmtStructure/gen.py:    def pmtid(self,site,pooln,iid,iwall):
    ./PoolPmtStructure/gen.py:                    pmtid = self.pmtid(siteid,pooln,iid,iwall)
    ./PoolPmtStructure/gen.py:                             'idnum':iid,'pmtid':pmtid,'tmp1':tmp1,'tmp2':tmp2,'tmp3':tmp3,'tmp4':tmp4 }
    ./PoolPmtStructure/gen.py:                    de.refs = [UserParameter("PmtID","int",['0x%x'%pmtid],desc="Packed PMT ID")]
    ./PoolPmtStructure/gen.py:    def pmtid(self,site,pooln,iid,iwall):
    ./PoolPmtStructure/gen.py:                    pmtid = self.pmtid(siteid,pooln,iid,iwall)
    ./PoolPmtStructure/gen.py:                    data = { 'site':site,'siteid':siteid,'poolpv':poolp,'poolnum':pooln,'wallnum':iwall,'idnum':iid,'pmtid':pmtid}
    ./PoolPmtStructure/gen.py:                    de.refs = [UserParameter("PmtID","int",['0x%x'%pmtid],desc="Packed PMT ID")]


`NuWa-trunk/dybgaudi/Detector/XmlDetDesc/python/XmlDetDescGen/AdPmtStructure/gen.py`::

     03 """
     04 Generate AD PMT Detector Elements.
     05 
     06 This generates the Structure XML for all AD PMTs into the TDS at
     07 /dd/Structure/AdPmts and files in to DDDB/AdPmtStructure/*.xml.
     08 
     09 This needs to match the hand-written XML in DDDB/AdPmts/geometry.xml.
     10 
     11 """
     12 
     13 
     14 Eight = True
     15 suffix = ''
     16 style = '2-2-4'
     17 
     18 
     19 class AdPmtStructure:
     20 

`NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/AdPmtStructure/db1.xml`::

      04 <!-- Detector Element "db-ad1-ring1-column1" -->
      05 <detelem name="db-ad1-ring1-column1">
      06   <geometryinfo lvname="/dd/Geometry/PMT/lvPmtHemi"
      07                 npath="pvAdPmtArray/pvAdPmtArrayRotated/pvAdPmtRingInCyl:1/pvAdPmtInRing:1/pvAdPmtUnit/pvAdPmt"
      08                 support="/dd/Structure/AD/db-oil1" />
      09   <!-- Packed PMT ID -->
      10   <userParameter name="PmtID" type="int" comment="Packed PMT ID">
      11     0x1010101
      12   </userParameter>
      13 </detelem>


Now what reads that::

    [blyth@belle7 lhcb]$ find . -name '*.cpp' -exec grep -l detelem {} \;
    ./Sim/GiGaCnv/src/component/GiGaLVolumeCnv.cpp
    ./Det/DetDescSvc/src/TransportSvc.cpp
    ./Det/DetDescCnv/src/Lib/XmlBaseDetElemCnv.cpp
    ./Det/DetDescCnv/src/Lib/XmlGenericCnv.cpp
    ./Det/DetDescCnv/src/component/XmlCatalogCnv.cpp


`NuWa-trunk/lhcb/Det/DetDescCnv/src/Lib/XmlBaseDetElemCnv.cpp`::

    187 // -----------------------------------------------------------------------
    188 // Fill an object with a new child element
    189 // -----------------------------------------------------------------------
    190 StatusCode XmlBaseDetElemCnv::i_fillObj (xercesc::DOMElement* childElement,
    191                                          DataObject* refpObject,
    192                                          IOpaqueAddress* address) {
    193   MsgStream log(msgSvc(), "XmlBaseDetElemCnv" );
    194 
    195   // gets the object
    196   DetectorElement* dataObj = dynamic_cast<DetectorElement*> (refpObject);
    197   // gets the element's name
    198   const XMLCh* tagName = childElement->getNodeName();
    ... 
    ...
    ...
    231   } else if (0 == xercesc::XMLString::compareString
    232              (geometryinfoString, tagName)) {
    233     // Everything is in the attributes
    234     std::string logVolName =
    235       dom2Std (childElement->getAttribute (lvnameString));
    236     std::string conditionPath =
    237       dom2Std (childElement->getAttribute (conditionString));
    238     std::string support =
    239       dom2Std (childElement->getAttribute (supportString));
    240     std::string replicaPath =
    241       dom2Std (childElement->getAttribute (rpathString));
    242     std::string namePath =
    243       dom2Std (childElement->getAttribute (npathString));
    244     log << MSG::VERBOSE << std::endl
    245         << "GI volume        : " << logVolName    << std::endl
    246         << "GI support       : " << support       << std::endl
    247         << "GI rpath         : " << replicaPath   << std::endl
    248         << "GI npath         : " << namePath      << std::endl
    249         << "GI conditionPath : " << conditionPath << endmsg;
    250 
    251     // creates a geometryInfo child
    252     if (logVolName.empty()) {
    253       dataObj->createGeometryInfo();
    254     } else if (support.empty()) {
    255       dataObj->createGeometryInfo (logVolName);
    256     } else if (!namePath.empty()) {
    257       dataObj->createGeometryInfo (logVolName, support,
    258                                    namePath, conditionPath);
    259     } else if (!replicaPath.empty()) {


`NuWa-trunk/lhcb/Det/DetDesc/src/Lib/DetectorElement.cpp`::

    205 const IGeometryInfo*
    206 DetectorElement::createGeometryInfo( const std::string& LogVol   ,
    207                                      const std::string& Support  ,
    208                                      const std::string& NamePath )
    209 {
    210   Assert( 0 == geometry() ,
    211           "Could not create REGULAR(1): Geometry already exist!" );
    212   m_de_iGeometry = GeoInfo::createGeometryInfo( this     ,
    213                                                 LogVol   ,
    214                                                 Support  ,
    215                                                 NamePath );
    216   return geometry();
    217 };


`NuWa-trunk/lhcb/Det/DetDesc/DetDesc/GeoInfo.h`::


     083   /** create regular geometry infor element 
     084    *  @exception GeometryInfoException null IDetectorElement pointer   
     085    *  @param de              pointer to detector element 
     086    *  @param LogVol          name of logical volume
     087    *  @param Support         name of support element 
     088    *  @param ReplicaNamePath replica path/address 
     089    *  @param alignmentPath   address of alignment condition
     090    */
     091   IGeometryInfo*
     092   createGeometryInfo( IDetectorElement*  de              ,
     093                       const std::string& LogVol          ,
     094                       const std::string& Support         ,
     095                       const std::string& ReplicaNamePath ,
     096                       const std::string& alignmentPath="");

     098   /** create regular geometry infor element 
     099    *  @exception GeometryInfoException null IDetectorElement pointer   
     100    *  @param de              pointer to detector element 
     101    *  @param LogVol          name of logical volume
     102    *  @param Support         name of support element 
     103    *  @param ReplicaPath     replica path 
     104    *  @param alignmentPath   address of alignment condition
     105    */
     106   IGeometryInfo*
     107   createGeometryInfo( IDetectorElement*  de              ,
     108                       const std::string& LogVol          ,
     109                       const std::string& Support         ,
     110                       const ILVolume::ReplicaPath& ReplicaPath,
     111                       const std::string& alignmentPath="");
     112 


`NuWa-trunk/lhcb/Det/DetDesc/src/Lib/GeoInfo.cpp`::

    099 IGeometryInfo*
    100 GeoInfo::createGeometryInfo( IDetectorElement*  de              ,
    101                              const std::string& LogVol          ,
    102                              const std::string& Support         ,
    103                              const std::string& ReplicaNamePath ,
    104                              const std::string& alignmentPath)
    105 {
    106   return new GeometryInfoPlus( de,
    107                                LogVol,
    108                                Support,
    109                                ReplicaNamePath,
    110                                alignmentPath);
    111 
    112 }

`NuWa-trunk/lhcb/Det/DetDesc/src/Lib/GeometryInfoPlus.cpp`::

     155 /// create regular  with name path
     156 GeometryInfoPlus::GeometryInfoPlus( IDetectorElement*  de,
     157                                     const std::string& LogVol,
     158                                     const std::string& Support,
     159                                     const std::string& ReplicaNamePath,
     160                                     const std::string& alignmentPath   )
     161   :
     162   m_log                 (       0     ),
     163   m_gi_has_logical      (    true         ),
     164   m_gi_lvolumeName      (   LogVol        ),
     165   m_gi_lvolume          (       0         ),
     166   m_hasAlignment        (     false       ),
     167   m_alignmentPath       ( alignmentPath   ),
     168   m_alignmentCondition  (       0         ),
     169   m_matrix              (       0     ),
     170   m_idealMatrix         (       0     ),
     171   m_localIdealMatrix    (       0     ),
     172   m_localDeltaMatrix    (       0     ),
     173   m_matrixInv           (       0     ),
     174   m_idealMatrixInv      (       0     ),
     175   m_gi_has_support      (    true         ),
     176   m_gi_supportName      (   Support       ),
     177   m_gi_support          (       0         ) ,
     178   m_gi_supportPath      (                 ),
     179   m_gi_supportNamePath  ( ReplicaNamePath ),
     180   m_gi_iDetectorElement (      de         ),
     181   m_gi_parentLoaded     (    false        ),
     182   m_gi_parent           (      0          ),
     183   m_gi_childLoaded      (    false        ) ,
     184   m_gi_childrens        (                 ) ,
     185   m_gi_childrensNames   (                 ),
     186   m_services            (      0          ){
     187   if( 0 == de  )
     188     { throw GeometryInfoException("IDetectorElement* points to NULL!"    ) ; }
     189 
     190   if ( initialize().isFailure() )
     191   { throw GeometryInfoException("Failed to initialize!") ; }
     192 
     193 }


`NuWa-trunk/lhcb/Det/DetDesc/DetDesc/IGeometryInfo.h`::

    034 class IGeometryInfo : virtual public IInterface
    035 {
    036 public:
    ...
    450   /** the name of associated Logical Volume
    451    *  @return the name of associated Logical Volume
    452    */
    453   virtual const std::string& lvolumeName() const = 0 ;
    454 
    455   /** associated Logical Volume
    456    *  @return the pointer to associated Logical Volume
    457    */
    458   virtual const ILVolume* lvolume () const = 0 ;
    ...
    512   /** the Logical Volume, addressed by  start and Replica Path
    513    *  @param start start
    514    *  @param replicaPath replicaPath
    515    *  @return pointer to Logical Volume
    516    */
    517   virtual const ILVolume* lvolume
    518   ( IGeometryInfo*               start       ,
    519     const ILVolume::ReplicaPath& replicaPath ) = 0;
    520  
    521   /// retrive reference to replica path (mistrerious "rpath" or "npath")
    522   virtual const ILVolume::ReplicaPath& supportPath() const = 0;
    523 
        
`NuWa-trunk/lhcb/Det/DetDesc/DetDesc/ILVolume.h`::

    036 class ILVolume : virtual public IInterface
    037 {
    038   ///
    039 public:
    040 
    041   /**  general typedefs  */
    042   typedef  std::vector<IPVolume*>                        PVolumes;
    043   typedef  PVolumes::size_type                           ReplicaType;
    044   typedef  std::vector<ReplicaType>                      ReplicaPath;
    045   typedef  std::vector<const IPVolume*>                  PVolumePath;    


    279   /** name of sensitive "detector" - needed for simulation 
    280    *  @return name of sensitive "detector"
    281    */
    282   virtual const std::string& sdName   ()                const = 0 ;
    283 




`NuWa-trunk/lhcb/Det/DetDescCnv/src/component/XmlDetectorElementCnv.cpp`::

     20 XmlDetectorElementCnv::XmlDetectorElementCnv (ISvcLocator* svc) :
     21   XmlBaseDetElemCnv (svc) {
     22 }



Resort to debugger to see where this comes into play
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    #54 0xb3431500 in GiGaGeo::createRep (this=0xa8febe0, object=0xa4fd1a0, address=@0xbf889a08) at ../src/component/GiGaGeo.cpp:647
    #55 0xb3435e85 in GiGaGeo::volume (this=0xa8febe0, Name=@0xa767e40) at ../src/component/GiGaGeo.cpp:186
    #56 0xb345681e in GiGaLVolumeCnv::updateRep (this=0xa41c830, Object=0xa768258) at ../src/component/GiGaLVolumeCnv.cpp:156
    #57 0xb3455d87 in GiGaLVolumeCnv::createRep (this=0xa41c830, Object=0xa768258, Address=@0xbf889fd8) at ../src/component/GiGaLVolumeCnv.cpp:108
    #58 0xb3431500 in GiGaGeo::createRep (this=0xa8febe0, object=0xa768258, address=@0xbf889fd8) at ../src/component/GiGaGeo.cpp:647
    #59 0xb3435e85 in GiGaGeo::volume (this=0xa8febe0, Name=@0xa6fd0d4) at ../src/component/GiGaGeo.cpp:186
    #60 0xb3426c33 in GiGaDetectorElementCnv::updateRep (this=0xa41c5c8, Object=0xa4fbac0) at ../src/component/GiGaDetectorElementCnv.cpp:194
    #61 0xb3427db0 in GiGaDetectorElementCnv::createRep (this=0xa41c5c8, Object=0xa4fbac0, Address=@0xbf88a620) at ../src/component/GiGaDetectorElementCnv.cpp:132
    #62 0xb3431500 in GiGaGeo::createRep (this=0xa8febe0, object=0xa4fbac0, address=@0xbf88a620) at ../src/component/GiGaGeo.cpp:647
    #63 0xb6273ecd in GiGaInputStream::execute (this=0xa3c8300) at ../src/Components/GiGaInputStream.cpp:76
    #64 0x04667408 in Algorithm::sysExecute (this=0xa3c8304) at ../src/Lib/Algorithm.cpp:558
    #65 0x03cddfd4 in GaudiSequencer::execute (this=0xa3b86d8) at ../src/lib/GaudiSequencer.cpp:100
    #66 0x04667408 in Algorithm::sysExecute (this=0xa3b86d8) at ../src/Lib/Algorithm.cpp:558
    #67 0x03c7568f in GaudiAlgorithm::sysExecute (this=0xa3b86d8) at ../src/lib/GaudiAlgorithm.cpp:161
    #68 0x046e341a in MinimalEventLoopMgr::executeEvent (this=0x9f768c8) at ../src/Lib/MinimalEventLoopMgr.cpp:450
    #69 0x03a1c956 in DybEventLoopMgr::executeEvent (this=0x9f768c8, par=0x0) at ../src/DybEventLoopMgr.cpp:125
    #70 0x03a1d18a in DybEventLoopMgr::nextEvent (this=0x9f768c8, maxevt=100) at ../src/DybEventLoopMgr.cpp:188
    #71 0x046e1dbd in MinimalEventLoopMgr::executeRun (this=0x9f768c8, maxevt=100) at ../src/Lib/MinimalEventLoopMgr.cpp:400
    #72 0xb77796d9 in ApplicationMgr::executeRun (this=0x9c43aa0, evtmax=100) at ../src/ApplicationMgr/ApplicationMgr.cpp:867
    #73 0x0623df57 in method_3426 (retaddr=0xa9ec320, o=0x9c43ecc, arg=@0x9cafc20) at ../i686-slc5-gcc41-dbg/dict/GaudiKernel/dictionary_dict.cpp:4375
    #74 0x00356add in ROOT::Cintex::Method_stub_with_context (context=0x9cafc18, result=0xaa39264, libp=0xaa392bc) at cint/cintex/src/CINTFunctional.cxx:319
    #75 0x02d56034 in ?? ()
    #76 0x09cafc18 in ?? ()
    #77 0x0aa39264 in ?? ()
    #78 0x00000000 in ?? ()
    (gdb) c


`NuWa-trunk/lhcb/Sim/GiGaCnv/src/component/GiGaGeo.cpp`::

    628 //=============================================================================
    629 // Convert the transient object to the requested representation.
    630 //  e.g. conversion to persistent objects.
    631 //=============================================================================
    632 StatusCode GiGaGeo::createRep
    633 ( DataObject*      object  ,
    634   IOpaqueAddress*& address )
    635 {
    636   ///
    637   if( 0 == object )
    638     { return Error(" createRep:: DataObject* points to NULL!");}
    639   ///
    640   const IDetectorElement* de = dynamic_cast<IDetectorElement*> ( object ) ;
    641   IConverter* cnv =
    642     converter( 0 == de ? object->clID() : CLID_DetectorElement );
    643   if( 0 == cnv )
    644     { return Error(" createRep:: converter is not found for '"
    645                    + object->registry()->identifier() + "'" );}
    646   ///
    647   return cnv->createRep( object , address );
    648 };


`NuWa-trunk/lhcb/Sim/GiGaCnv/src/component/GiGaDetectorElementCnv.cpp`::

    139 StatusCode GiGaDetectorElementCnv::updateRep( DataObject*     Object  ,
    140                                               IOpaqueAddress* /* Address */ )
    141 {
    142   ///
    143   MsgStream log( msgSvc() , name() );
    144   log << MSG::DEBUG << "updateRep::start "
    145       << Object->registry()->identifier() << endreq;
    146   ///
    147   if( 0 == Object                 )
    148     { return Error("updateRep::DataObject* points to NULL"); }
    149   ///
    150   IDetectorElement* de = 0 ;
    151   try        { de = dynamic_cast<IDetectorElement*>( Object ) ; }
    152   catch(...) { de =                                 0 ; }
    153   if( 0 == de        )
    154     { return Error("updateRep::Bad cast to IDetectorElement*"); }
    155   if( 0 == geoSvc()  )
    156     { return Error("updateRep::Conversion Service is unavailable"); }
    157   ///
    158   IGeometryInfo* gi = de->geometry() ;
    159   if( 0 == gi )
    160     { return Error("updateRep:: IGeometryInfo* is not available for " +
    161                    de->name() ); }
    162   const ILVolume*      lv = gi->lvolume () ;
    163   if( 0 == lv )
    164     { return Error("updateRep:: ILVolume*      is not available for " +
    165                    de->name() ); }
    166   //
    167   // // look at G4 physical volume store and check 
    168   // //  if it was converted exlicitely or imlicitely
    169   //    {
    170   //      std::string path ( de->name() );
    171   //      do
    172   //      {
    173   //      G4VPhysicalVolume* pv = 0; 
    174   //      G4PhysicalVolumeStore& store = *G4PhysicalVolumeStore::GetInstance();
    175   //      for( unsigned int indx = 0 ; indx < store.size() ; ++indx )
    176   //      { if( path == store[indx]->GetName() ) { pv = store[indx] ; break; } }
    177   //      /// it was converted EXPLICITELY or IMPLICITELY !!!
    178   //      if( 0 != pv ) 



:google:`gaudi detector element set params`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://lhcb-comp.web.cern.ch/lhcb-comp/Frameworks/Gaudi/Gaudi_v9/GUG/Output/GUG_DetDescription.html
* http://lhcb-comp.web.cern.ch/lhcb-comp/Frameworks/DetDesc/Documents/detElemExtension.pdf

Once user parameters are defined in XML, they are converted by the regular
converter for detector elements and are then reachable in the C++ code, 
with DetectorElement methods.


How to access all DE ? detSvc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://lhcb-comp.web.cern.ch/lhcb-comp/Frameworks/Gaudi/Tutorial/9_Detector_Description.pdf

accessing detector data is done using the DetectorDataSvc (detSvc()) and with the help of a SmartDataPtr().


::

    [blyth@belle7 dybgaudi]$ find . -name '*.py' -exec grep -H detSvc {} \;
    ./Simulation/Historian/python/pmtbox.py:    det = app.detSvc()
    ./Tutorial/Calibration/python/Calibration/ACUNeutronCapturePosition.py:        ad1 = self.detSvc("/dd/Structure/AD/db-oil1")
    ./Detector/XmlDetDesc/python/XmlDetDesc/dumper.py:    dsv = g.detSvc()
    ./Detector/XmlDetDesc/python/XmlDetDesc/dumper.py:    det = app.detSvc()
    ./Detector/XmlDetDesc/python/xmldetdesc.py:    dsv = g.detSvc()
    ./Detector/XmlDetDesc/python/xmldetdesc.py:    det = app.detSvc()
    ./Detector/DetDescVis/python/dump.py:    det = g.detSvc()
    ./Production/MDC09b/python/MDC09b/chkGamma/__init__.py:        det = self.detSvc(self.target_de_name)   ## '/dd/Structure/AD/db-ade1/db-sst1/db-oil1'
    ./Production/MDC09b/python/MDC09b/chkIBD/__init__.py:#        det = self.detSvc(self.target_de_name)
    ./Production/MDC09b/python/MDC09b/chkIBD/__init__.py:#        det_gds = self.detSvc(self.gds_de_name)
    ./Production/MDC09b/python/MDC09b/chkIBD/__init__.py:#        det_lso = self.detSvc(self.lso_de_name)
    ./Production/MDC09a/python/MDC09a/chkGamma/__init__.py:        det = self.detSvc(self.target_de_name)
    ./Production/MDC09a/python/MDC09a/chkIBD15/__init__.py:        det = self.detSvc(self.target_de_name)
    ./Production/MDC09a/python/MDC09a/chkIBD15/__init__.py:        det_gds = self.detSvc(self.gds_de_name)
    ./Production/MDC09a/python/MDC09a/chkIBD15/__init__.py:        det_lso = self.detSvc(self.lso_de_name)
    ./Production/MDC09a/python/MDC09a/chkIBD/__init__.py:        det = self.detSvc(self.target_de_name)
    ./Production/MDC09a/python/MDC09a/chkIBD/__init__.py:        det_gds = self.detSvc(self.gds_de_name)
    ./Production/MDC09a/python/MDC09a/chkIBD/__init__.py:        det_lso = self.detSvc(self.lso_de_name)
    ./Production/MDC09a/python/MDC09a/chkIBD/AdPerformance.py:#        det = self.detSvc(self.target_de_name)
    ./Production/MDC09a/python/MDC09a/chkIBD/AdPerformance.py:#        det_gds = self.detSvc(self.gds_de_name)
    ./Production/MDC09a/python/MDC09a/chkIBD/AdPerformance.py:#        det_lso = self.detSvc(self.lso_de_name)

    36 
    37         self.target_de_name = '/dd/Structure/AD/db-ade1/db-sst1/db-oil1'
    38         self.gds_de_name = '/dd/Structure/AD/db-gds1'
    39         self.lso_de_name = '/dd/Structure/AD/db-lso1'


`NuWa-trunk/dybgaudi/Detector/XmlDetDesc/python/XmlDetDesc/dump_geo.py`::

     11 def configure(argv=None):
     12     if argv:
     13         path = argv[0]
     14     else:
     15         path = '/dd/Geometry'
     16 
     17     from XmlDetDescChecks.XmlDetDescChecksConf import XddDumpAlg
     18 
     19     da = XddDumpAlg()
     20     da.Paths = [path]
     21 
     22     from Gaudi.Configuration import ApplicationMgr
     23     app = ApplicationMgr()
     24     app.TopAlg.append(da)



de.sh
~~~~~~

Lists DetectorElement names with UserParameter called `PmtId` 

* `NuWa-trunk/dybgaudi/Detector/XmlDetDescChecks/python/XmlDetDescChecks/dedump.py`
* `NuWa-trunk/dybgaudi/Detector/XmlDetDescChecks/src/DeDumpAlg.cc`
* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Detector/XmlDetDescChecks/src/DeDumpAlg.cc

* all DE names inhabit `/dd/Structure/` 

  * can GeometryInfo relate that to real PV names ?

* may be truncated as crashed on `/dd/Structure/CalibrationBox/db-ad1-strongAmC`
* is 16355+1 = 16356 correct for all three halls PMT count ?  

::

    [blyth@belle7 ~]$ de.sh last 
    Importing module "XmlDetDescChecks.dedump" ["/dd/Structure/AD/far-oil4/far-ad4-ring0-column6"]
    ...
    DetectorDataSvc                    SUCCESS Detector description database: /data1/env/local/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/dayabay.xml
    EventClockSvc.FakeEventTime           INFO Event times generated from 0 with steps of 0
    DE   67371014         0 /dd/Structure/AD/far-oil4/far-ad4-ring0-column6
    ApplicationMgr                        INFO Application Manager Stopped successfully
    ToolSvc                               INFO Removing all tools created by ToolSvc
    ApplicationMgr                        INFO Application Manager Finalized successfully
    ApplicationMgr                        INFO Application Manager Terminated successfully
    [blyth@belle7 ~]$ 


::

    [blyth@belle7 ~]$ de.sh all
    ...
    Importing module "XmlDetDescChecks.dedump" ["/dd"]
    Trying to call configure() on XmlDetDescChecks.dedump
    ...
    Dumping /dd
    DetectorPersistencySvc                INFO  'CnvServices':[ 'XmlCnvSvc/XmlCnvSvc' ]
    DetectorPersistencySvc                INFO Added successfully Conversion service:XmlCnvSvc
    DetectorDataSvc                    SUCCESS Detector description database: /data1/env/local/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/dayabay.xml
    ...
    DE   16843009         0 /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column1
    DE   16843010         1 /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column2
    DE   16843011         2 /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column3
    DE   16843012         3 /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column4
    DE   16843013         4 /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column5
    DE   16843014         5 /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column6
    DE   16843015         6 /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column7
    DE   16843016         7 /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column8
    DE   16843017         8 /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column9
    DE   16843018         9 /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column10
    DE   16843019        10 /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column11
    DE   16843020        11 /dd/Structure/DayaBay/db-rock/db-ows/db-curtain/db-iws/db-ade1/db-sst1/db-oil1/db-ad1-ring1-column12
    ...
    DE   67373078     16347 /dd/Structure/AD/far-oil4/far-ad4-ring8-column22
    DE   67373079     16348 /dd/Structure/AD/far-oil4/far-ad4-ring8-column23
    DE   67373080     16349 /dd/Structure/AD/far-oil4/far-ad4-ring8-column24
    DE   67371009     16350 /dd/Structure/AD/far-oil4/far-ad4-ring0-column1
    DE   67371010     16351 /dd/Structure/AD/far-oil4/far-ad4-ring0-column2
    DE   67371011     16352 /dd/Structure/AD/far-oil4/far-ad4-ring0-column3
    DE   67371012     16353 /dd/Structure/AD/far-oil4/far-ad4-ring0-column4
    DE   67371013     16354 /dd/Structure/AD/far-oil4/far-ad4-ring0-column5
    DE   67371014     16355 /dd/Structure/AD/far-oil4/far-ad4-ring0-column6
    XmlGenericCnv                        FATAL An exception went out of the conversion process : *GeometryInfoException*    GeometryInfoPlus:: error during retrieve of Replica Path     StatusCode=FAILURE
    DeDumpAlg                            FATAL DeDumpAlg:: Exception throw: get():: No valid data at '/dd/Structure/CalibrationBox/db-ad1-strongAmC' StatusCode=FAILURE
    DeDumpAlg.sysExecute()               FATAL  Exception with tag= is caught 
    DeDumpAlg.sysExecute()               ERROR  DeDumpAlg:: get():: No valid data at '/dd/Structure/CalibrationBox/db-ad1-strongAmC'     StatusCode=FAILURE
    ChronoStatSvc                         INFO  Number of skipped events for MemStat-1
    MinimalEventLoopMgr.executeEvent()   FATAL  Exception with tag= thrown by DeDumpAlg
    MinimalEventLoopMgr.executeEvent()   ERROR  DeDumpAlg:: get():: No valid data at '/dd/Structure/CalibrationBox/db-ad1-strongAmC'     StatusCode=FAILURE
    EventLoopMgr                       WARNING Execution of algorithm DeDumpAlg failed
    EventLoopMgr                         ERROR Error processing event loop.
    EventLoopMgr                         ERROR Terminating event processing loop due to errors
    EventLoopMgr                         ERROR Terminating event processing loop due to errors
    ApplicationMgr                        INFO Application Manager Stopped successfully





::

    [blyth@belle7 ~]$ de.sh 111
    ...
    Importing module "XmlDetDescChecks.dedump" ["/dd/Structure/AD/db-oil1/db-ad1-ring1-column1"]
    Dumping /dd/Structure/AD/db-oil1/db-ad1-ring1-column1
    DetectorDataSvc                    SUCCESS Detector description database: /data1/env/local/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/dayabay.xml
    ...
    PmtId  0x 1010101 0d   16843009 nn          1 de /dd/Structure/AD/db-oil1/db-ad1-ring1-column1
    lvn /dd/Geometry/PMT/lvPmtHemi
    ilv 
     LVolume (17)  name = '/dd/Geometry/PMT/lvPmtHemi'  #physvols1#0  class PVolume (154) [ name='pvPmtHemiVacuum' logvol='/dd/Geometry/PMT/lvPmtHemiVacuum']

     SolidType='SolidUnion'     name='pmt-hemi'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(     -131/      131,     -131/      131,     -179/      131,      179,      131) 
     ** 'Main' solid is 
     SolidType='SolidIntersection'  name='pmt-hemi-glass-bulb'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(     -131/      131,     -131/      131,     -131/      131,      131,      131) 
     ** 'Main' solid is 
     SolidType='SolidSphere'    name='pmt-hemi-face-glass'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(     -131/      131,     -131/      131,     -131/      131,      131,      131) 
    outerRadius[mm]      131
     ** 'Booled' with 
     SolidType='SolidChild'     name='Child For pmt-hemi-glass-bulb'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(     -102/      102,     -102/      102,      -59/      145,      145,      102) 
     SolidType='SolidSphere'    name='pmt-hemi-top-glass'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(     -102/      102,     -102/      102,     -102/      102,      102,      102) 
    outerRadius[mm]      102
     ** 'Booled' with 
     SolidType='SolidChild'     name='Child For pmt-hemi-glass-bulb'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(     -102/      102,     -102/      102,      -33/      171,      171,      102) 
     SolidType='SolidSphere'    name='pmt-hemi-bot-glass'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(     -102/      102,     -102/      102,     -102/      102,      102,      102) 
    outerRadius[mm]      102

     ** 'Booled' with 
     SolidType='SolidChild'     name='Child For pmt-hemi'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(    -94.5/     94.5,    -94.5/     94.5,     -179/     9.97,      179,     94.5) 
     SolidType='SolidTubs'  name='pmt-hemi-base'
     BPs: (x,y,z,r,rho)[Min/Max][mm]=(    -42.2/     42.2,    -42.2/     42.2,    -84.5/     84.5,     94.5,     42.2) 
    [ sizeZ[mm]=      169 outerRadius[mm]=     42.2]


    Material name='/dd/Materials/Pyrex' 

        GeometryInfo @ 0xa7809b8
        globally @ (-16572.9,-801470,-8842.5) mm
    ApplicationMgr                        INFO Application Manager Stopped successfully
    ...
    [blyth@belle7 ~]$ 







DsPmtSensDet::ProcessHits HC population
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `step -> preStepPoint -> touchableHistory -> DetectorElement -> SensDetId`

* where are `SensDetId` associated with `DetectorElement` ?
* how do the `DetectorElement` and `touchableHistory` correspond to PVs ?

::

    318 bool DsPmtSensDet::ProcessHits(G4Step* step,
    319                                G4TouchableHistory* /*history*/)
    320 {
    321     //if (!step) return false; just crash for now if not defined
    322 
    323     // Find out what detector we are in (ADx, IWS or OWS)
    324     G4StepPoint* preStepPoint = step->GetPreStepPoint();
    325 
    326     double energyDep = step->GetTotalEnergyDeposit();
    327 
    328     if (energyDep <= 0.0) {
    329         //debug() << "Hit energy too low: " << energyDep/CLHEP::eV << endreq;
    330         return false;
    331     }
    332 
    333     const G4TouchableHistory* hist =
    334         dynamic_cast<const G4TouchableHistory*>(preStepPoint->GetTouchable());
    335     if (!hist or !hist->GetHistoryDepth()) {
    336         error() << "ProcessHits: step has no or empty touchable history" << endreq;
    337         return false;
    338     }
    339 
    340     const DetectorElement* de = this->SensDetElem(*hist);
    341     if (!de) return false;
    342 
    343     // wangzhe QE calculation starts here.
    344     int pmtid = this->SensDetId(*de);
    345     DayaBay::Detector detector(pmtid);
    ...
    ...     hit formation
    ...
    459     DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();
    460 
    461     // base hit
    462 
    463     // Time since event created
    464     sphit->setHitTime(preStepPoint->GetGlobalTime());
    465 
    466     //#include "G4NavigationHistory.hh"
    467 
    468     const G4AffineTransform& trans = hist->GetHistory()->GetTopTransform();
    469     const G4ThreeVector& global_pos = preStepPoint->GetPosition();
    470     G4ThreeVector pos = trans.TransformPoint(global_pos);
    471     sphit->setLocalPos(pos);
    472     sphit->setSensDetId(pmtid);
    473    
    474     // pmt hit
    475     // sphit->setDir(...);       // for now
    476     G4ThreeVector pol = trans.TransformAxis(track->GetPolarization());
    477     pol = pol.unit();
    478     G4ThreeVector dir = trans.TransformAxis(track->GetMomentum());
    479     dir = dir.unit();
    480     sphit->setPol(pol);
    481     sphit->setDir(dir);
    482     sphit->setWavelength(wavelength);
    483     sphit->setType(0);
    484     // G4cerr<<"PMT: set hit weight "<<weight<<G4endl; //gonchar
    485     sphit->setWeight(weight);
    ...
    ...
    505     int trackid = track->GetTrackID();
    506     this->StoreHit(sphit,trackid);
    507     debug() << "Stored photon " << trackid << " weight " << weight << " pmtid " << (void*)pmtid << " wavelength(nm) " << wavelength/CLHEP::nm << endreq;
    508     return true;
    509 }
    ...
    ...     pmt_id -> sdid -> m_hc[sdid] (cache) -> hc
    ...
    ...     Hmm, can externally invoke to populate the standard HCs, 
    ...
    ...             sdpmt->StoreHit(sphit,trackid)
    ...
    ...     * How exactly do pmtid get attached to detector elements ?
    ...     * Also need transform matrix for local conversion ?
    ...
    ...       * CPU side 
    ...
    ...
    ...
    511 void DsPmtSensDet::StoreHit(DayaBay::SimPmtHit* hit, int trackid)
    512 {
    513     int did = hit->sensDetId();
    514     DayaBay::Detector det(did);
    515     short int sdid = det.siteDetPackedData();
    516 
    517     G4DhHitCollection* hc = m_hc[sdid];
    ...
    537     hc->insert(new G4DhHit(hit,trackid));
    538 }





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
    ...   // recurse up DetectorElement heirarchy until find an idParameter to return
    ...   // where are these int ID set ?  
    ...   //     * presumably generated by GiGaCnv 
    ...   //
    ...
    260 int  DsPmtSensDet::SensDetId(const DetectorElement& de)
    261 {
    262     const DetectorElement* detelem = &de;
    263 
    264     while (detelem) {
    265         if (detelem->params()->exists(m_idParameter)) {
    266             break;
    267         }
    268         detelem = dynamic_cast<const DetectorElement*>(detelem->parentIDetectorElement());
    269     }
    270     if (!detelem) {
    271         warning() << "Could not get PMT detector element starting from " << de << endreq;
    272         return 0;
    273     }
    274 
    275     return detelem->params()->param<int>(m_idParameter);
    276 }




GetTouchable
--------------

::

    delta:geant4.10.00.p01 blyth$ find . -name '*.hh' -exec grep -H GetTouchable {} \;
    ./source/parameterisations/gflash/include/G4GFlashSpot.hh:    G4TouchableHandle GetTouchableHandle() const {return theHandle;}
    ./source/parameterisations/gflash/include/G4VGFlashSensitiveDetector.hh:            tmpPoint->SetTouchableHandle(aSpot->GetTouchableHandle());
    ./source/track/include/G4ParticleChangeForLoss.hh:  aTrack->SetTouchableHandle(currentTrack->GetTouchableHandle());
    ./source/track/include/G4ParticleChangeForTransport.hh:    const G4TouchableHandle& GetTouchableHandle() const;
    ./source/track/include/G4StepPoint.hh:   const G4VTouchable* GetTouchable() const;
    ./source/track/include/G4StepPoint.hh:   const G4TouchableHandle& GetTouchableHandle() const;
    ./source/track/include/G4Track.hh:   const G4VTouchable*      GetTouchable() const;
    ./source/track/include/G4Track.hh:   const G4TouchableHandle& GetTouchableHandle() const;
    ./source/tracking/include/G4SteppingManager.hh:   const G4TouchableHandle& GetTouchableHandle();
    ./source/tracking/include/G4SteppingManager.hh:  inline const G4TouchableHandle& G4SteppingManager::GetTouchableHandle() {
    delta:geant4.10.00.p01 blyth$ 



HC Creation
------------

::

    [blyth@belle7 dybgaudi]$ find . -name '*.cc' -exec grep -H G4DhHitCollection {} \;
    ./Simulation/DetSim/src/DsPmtSensDet.cc:    G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,collectionName[0]);
    ./Simulation/DetSim/src/DsPmtSensDet.cc:            G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,name.c_str());
    ./Simulation/DetSim/src/DsPmtSensDet.cc:    G4DhHitCollection* hc = m_hc[sdid];
    ./Simulation/DetSim/src/DsRpcSensDet.cc:    G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,collectionName[0]);
    ./Simulation/DetSim/src/DsRpcSensDet.cc:            G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,name.c_str());
    ./Simulation/DetSim/src/DsRpcSensDet.cc:    G4DhHitCollection* hc = m_hc[sdid];
    ./Simulation/DetSim/src/DsPullEvent.cc:        G4DhHitCollection* g4hc = dynamic_cast<G4DhHitCollection*>(hcs->GetHC(ihc));
    ./Simulation/Fifteen/DetSimProc/src/DetSimProc.cc:  G4DhHitCollection* g4hc = dynamic_cast<G4DhHitCollection*>(hcs->GetHC(ihc));



Watershed : DsPullEvent
-------------------------

* watershed between python/pyroot/Gaudi/GiGa and underlying Geant4 at **DsPullEvent**


Stack Trace during propagation
--------------------------------

::

    513 
    514     494           operator[](size_type __n) const
    515     (gdb) bt
    516     #0  0x041f811a in std::vector<G4NavigationLevel, std::allocator<G4NavigationLevel> >::operator[] (this=0xc4045f4, __n=12) at /usr/lib/gcc/i386-redhat-linux/4.1.2/../.    ./../../include/c++/4.1.2/bits/stl_vector.h:494
    517     #1  0x041f81a3 in G4NavigationHistory::GetTopTransform (this=0xc4045f4) at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/geometry/volume    s/include/G4NavigationHistory.icc:102
    518     #2  0x0703aa3c in G4Navigator::ComputeLocalAxis (this=0xc4045e8, pVec=@0xbfd17220) at include/G4Navigator.icc:57
    519     #3  0x070365cb in G4Navigator::ComputeStep (this=0xc4045e8, pGlobalpoint=@0xbfd17208, pDirection=@0xbfd17220, pCurrentProposedStepLength=47809528.913293302, pNewSafet    y=@0xbfd17238) at src/G4Navigator.cc:628
    520     #4  0x04e096fa in G4Transportation::AlongStepGetPhysicalInteractionLength (this=0xc06d4e8, track=@0x10a5a5c8, currentMinimumStep=47809528.913293302, currentSafety=@0x    bfd173b8, selection=0xc4042fc) at src/G4Transportation.cc:225
    521     #5  0x06e23e1b in G4VProcess::AlongStepGPIL (this=0xc06d4e8, track=@0x10a5a5c8, previousStepSize=17.522238749144233, currentMinimumStep=47809528.913293302, proposedSa    fety=@0xbfd173b8, selection=0xc4042fc)
    522         at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/processes/management/include/G4VProcess.hh:447
    523     #6  0x06e22849 in G4SteppingManager::DefinePhysicalStepLength (this=0xc4041f0) at src/G4SteppingManager2.cc:235
    524     #7  0x06e1ee2c in G4SteppingManager::Stepping (this=0xc4041f0) at src/G4SteppingManager.cc:181
    525     #8  0x06e2d50a in G4TrackingManager::ProcessOneTrack (this=0xc4041c8, apValueG4Track=0x10a5a5c8) at src/G4TrackingManager.cc:126
    526     #9  0x06ea024f in G4EventManager::DoProcessing (this=0xc4039d8, anEvent=0x102ccca8) at src/G4EventManager.cc:185
    527     #10 0x06ea09e6 in G4EventManager::ProcessOneEvent (this=0xc4039d8, anEvent=0x102ccca8) at src/G4EventManager.cc:335
    528     #11 0xb4d2b5e8 in GiGaRunManager::processTheEvent (this=0xc403170) at ../src/component/GiGaRunManager.cpp:207
    529     #12 0xb4d2a522 in GiGaRunManager::retrieveTheEvent (this=0xc403170, event=@0xbfd17cf8) at ../src/component/GiGaRunManager.cpp:158
    530     #13 0xb4d0664f in GiGa::retrieveTheEvent (this=0xc402778, event=@0xbfd17cf8) at ../src/component/GiGa.cpp:469
    531     #14 0xb4d03564 in GiGa::operator>> (this=0xc402778, event=@0xbfd17cf8) at ../src/component/GiGaIGiGaSvc.cpp:73
    532     #15 0xb4d012fa in GiGa::retrieveEvent (this=0xc402778, event=@0xbfd17cf8) at ../src/component/GiGaIGiGaSvc.cpp:211
    533     #16 0xb4f4acd3 in DsPullEvent::execute (this=0xc3f5d00) at ../src/DsPullEvent.cc:54
    534     #17 0x069c1408 in Algorithm::sysExecute (this=0xc3f5d00) at ../src/Lib/Algorithm.cpp:558
    535     #18 0x0350ed4e in DybBaseAlg::sysExecute (this=0xc3f5d00) at ../src/lib/DybBaseAlg.cc:53
    536     #19 0x02cc6fd4 in GaudiSequencer::execute (this=0xbeb8140) at ../src/lib/GaudiSequencer.cpp:100
    537     #20 0x069c1408 in Algorithm::sysExecute (this=0xbeb8140) at ../src/Lib/Algorithm.cpp:558
    538     #21 0x02c5e68f in GaudiAlgorithm::sysExecute (this=0xbeb8140) at ../src/lib/GaudiAlgorithm.cpp:161
    539     #22 0x06a3d41a in MinimalEventLoopMgr::executeEvent (this=0xba77900) at ../src/Lib/MinimalEventLoopMgr.cpp:450
    540     #23 0x038ba956 in DybEventLoopMgr::executeEvent (this=0xba77900, par=0x0) at ../src/DybEventLoopMgr.cpp:125
    541     #24 0x038bb18a in DybEventLoopMgr::nextEvent (this=0xba77900, maxevt=1) at ../src/DybEventLoopMgr.cpp:188
    542     #25 0x06a3bdbd in MinimalEventLoopMgr::executeRun (this=0xba77900, maxevt=1) at ../src/Lib/MinimalEventLoopMgr.cpp:400
    543     #26 0x093096d9 in ApplicationMgr::executeRun (this=0xb744aa0, evtmax=1) at ../src/ApplicationMgr/ApplicationMgr.cpp:867
    544     #27 0x0829bf57 in method_3426 (retaddr=0xc4f7d00, o=0xb744ecc, arg=@0xb7b0c20) at ../i686-slc5-gcc41-dbg/dict/GaudiKernel/dictionary_dict.cpp:4375
    545     #28 0x001d6add in ROOT::Cintex::Method_stub_with_context (context=0xb7b0c18, result=0xc53d26c, libp=0xc53d2c4) at cint/cintex/src/CINTFunctional.cxx:319
    546     #29 0x0330e034 in ?? ()
    547     #30 0x0b7b0c18 in ?? ()
    548     #31 0x0c53d26c in ?? ()
    549     #32 0x00000000 in ?? ()
    550     Current language:  auto; currently c++
    551     (gdb) 




DsPullEvent
-----------


`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPullEvent.cc`::

     40 StatusCode DsPullEvent::execute()
     41 {
     42     DayaBay::SimHeader* header = MakeHeaderObject();
     43 
     44     // Just pass through GenHeader's timestamp.  This also causes
     45     // GenHeader to be registered as input, something that would
     46     // normally just happen if DsPushKine and DsPullEvent were the
     47     // same algorithm.
     48     DayaBay::GenHeader* gen_header = getTES<DayaBay::GenHeader>(m_genLocation);
     49     header->setTimeStamp(gen_header->timeStamp());
     50 
     51     //////////////////////////
     52     // Primary event vertices.
     53     const G4Event* g4event = 0;
     54     m_giga->retrieveEvent(g4event);
     55     if (!g4event) {
     56         error() << "No G4Event!" << endreq;
     57         return StatusCode::FAILURE;
     58     }
     59 
     60     // reset Capture
     61     G4DhNeutronCapture capture;
     62     m_capinfo->addCapture(capture);
     63 
     64     int nverts = g4event->GetNumberOfPrimaryVertex();
     65     if( nverts == 0 ) {
     66         warning() << "The g4event has zero primary vertices!" << endreq;
     67         return StatusCode::SUCCESS;
     68     }
     69 
     70 
     71     debug() << "Pulled event with " << nverts
     72            << " primary vertices, event id:" << g4event->GetEventID() << endreq;
     73     G4PrimaryVertex* g4vtx = g4event->GetPrimaryVertex(0);
     74     while (g4vtx) {
     75         debug() << "\n\tat (" << g4vtx->GetX0() << "," << g4vtx->GetY0() << "," << g4vtx->GetZ0() << ")";
     76         g4vtx = g4vtx->GetNext();
     77         break;
     78     }
     79     debug() << endreq;
     80 
     81     //////////////////////////
     82     // particle histories.
     83     // Do this first so we can use it below.
     84     DayaBay::SimParticleHistory* history =0;
     85     m_historyKeeper->ClaimCurrentHistory(history); // This takes ownership from the Keeper.
     86     header->setParticleHistory(history);
     87 
     88     //////////////////////////
     89     // Unobservable Statistics
     90     DayaBay::SimUnobservableStatisticsHeader* unobs =0;
     91     m_historyKeeper->ClaimCurrentUnobservable(unobs); // This takes ownership from the Keeper.
     92     header->setUnobservableStatistics(unobs);
     93 
     94     //////////////////////////
     95     // Hit collections.
     96     G4HCofThisEvent* hcs = g4event->GetHCofThisEvent();
     97     if (!hcs) {
     98         warning() << "No HitCollections in this event" << endreq;
     99         return StatusCode::SUCCESS;
     00     }
     01     int nhc = hcs->GetNumberOfCollections();
     02     if (!nhc) {
     03         warning() << "Number of HitCollections is zero" << endreq;
     04         return StatusCode::SUCCESS;
     05     }
     06     debug () << "# HitCollections = " << nhc << endreq;
     07 
     08     // introduce the headers to each other
     09     DayaBay::SimHitHeader* hit_header = new DayaBay::SimHitHeader(header);
     10     header->setHits(hit_header);
     11 
     12     double earliestTime = 0;
     13     double latestTime = 0;
     14     Context context;
     15     context.SetSimFlag(SimFlag::kMC);
     16     bool firstDetector = true;
     17     int hitcount=0;  // deal with no hits situation
     18 
     19     for (int ihc=0; ihc<nhc; ++ihc) {
     20         G4DhHitCollection* g4hc = dynamic_cast<G4DhHitCollection*>(hcs->GetHC(ihc));
     21         if (!g4hc) {
     22             error() << "Failed to get hit collection #" << ihc << endreq;
     23             return StatusCode::FAILURE;
     24         }
     25 
     26         // DetSim produces hit collections even for unsimulated detectors
     27         size_t nhits = g4hc->GetSize();
     28     hitcount+=nhits;
     29         if (!nhits) continue;
     30 
     31     bool firstHit = true;
     32         DayaBay::SimHitCollection::hit_container hits;
     33     DayaBay::Detector detector;
     34         DayaBay::SimHitCollection* shc =
     35       new DayaBay::SimHitCollection(hit_header,detector,hits);
     36         for (size_t ihit=0; ihit<nhits; ++ihit) {

