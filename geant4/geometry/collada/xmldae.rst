XMLDAE : raw G4DAE node structure
=====================================

.. contents:: :local:

Questions
----------

#. Would the xml documument ids be unique without the pointers 0x.... ?

   * checking the .gdml there are dupes in the subtraction/union solids and between element/material names
     see ~/e/tools/checkxml.py 
   * checking the GDML writing on which the DAE writing is based,  there is currently only a global addPointerToName 
     switch so cannot easily turn it off for volumes and not for solids as would break references to solids

#. Can I reproduce VRML2 output from the DAE ? As a validation of all those transformations and everything else.

   * PV count now matches
   * PV name matching, the NCName IDref XML restriction forced replacing 3 chars ":/#" with  "_"
   
     * that is difficult to reverse, need some more unused acceptable chars (single chars would be best)
     * iterating on dae-edit;dae-validate find that "." and "-" are acceptable on other than the first char 

     * http://www.schemacentral.com/sc/xsd/t-xsd_NCName.html
     * http://stackoverflow.com/questions/1631396/what-is-an-xsncname-type-and-when-should-it-be-used
     * http://docs.marklogic.com/xdmp:encode-for-NCName
     * :google:`NCName encoding decoding`
     * https://nees.org/tools/vees/browser/xerces/src/xercesc/util/XMLString.cpp
     * http://msdn.microsoft.com/en-us/library/system.xml.xmlconvert.aspx 

  * TODO:

    * add checkxml.py collection of all id characters to see if "." is used 


Reversible Char Swaps
~~~~~~~~~~~~~~~~~~~~~~~

::

    /  ->   _
    :  ->   -      (colon always precedes digits eg :1 )  
    #  ->   .


The only '-' containg names that beings with '/'::

    /dd/Structure/Sites/db-rock0xc633af8
    /dd/Structure/Sites/db-rock0xc633af8_pos
    /dd/Structure/Sites/db-rock0xc633af8_rot



Raw Node Tree
--------------

Raw Node tree has the lv as well as pv, wherese VRML2 tree has only pv ?

* the raw collada node tree triples the Geant4 volume tree nodes into a regular PPV/LV/GEO per volume structure


::

    simon:~ blyth$ xmldae.py -w -i 0,100 -z 9
    2013-10-10 16:34:03,580 env.geant4.geometry.collada.xmldae INFO     /Users/blyth/env/bin/xmldae.py -w -i 0,100 -z 9
    2013-10-10 16:34:03,583 env.geant4.geometry.collada.xmldae INFO     reading /usr/local/env/geant4/geometry/xdae/g4_01.dae 
    2013-10-10 16:34:03,840 env.geant4.geometry.collada.xmldae INFO     create_tree starting from root #World0xad7b048 
    2013-10-10 16:34:03,951 env.geant4.geometry.collada.xmldae INFO     collect_xmlcache found 5892 nodes 
    2013-10-10 16:34:34,411 env.geant4.geometry.collada.xmldae INFO     create_tree completed from root
    registry 24459 
    xmlcache 5892 
    effect: 36 
    material: 36 
    geometry: 249 
    scene: 1 
    rooturl: #World0xad7b048 
    2013-10-10 16:34:34,415 env.geant4.geometry.collada.xmldae INFO     walk starting from root   0    World0xad7b048.0                                                                                      1    tgt:_dd_Materials_Vacuum0x8b746a0  ref:None matrix:None  
    0 0 World0xad7b048.0
    1 1 _dd_Structure_Sites_db-rock0xad7b188.0
    2 2 _dd_Geometry_Sites_lvNearSiteRock0xad7af08.0
    3 3 _dd_Geometry_Sites_lvNearSiteRock_pvNearHallTop0xad7ad70.0
    4 4 _dd_Geometry_Sites_lvNearHallTop0xabc3670.0
    5 5 _dd_Geometry_Sites_lvNearHallTop_pvNearTopCover0xabc3390.0
    6 6 _dd_Geometry_PoolDetails_lvNearTopCover0xabaffe8.0
    5 7 _dd_Geometry_Sites_lvNearHallTop_pvNearTeleRpc_pvNearTeleRpc_10xabc36c8.0
    6 8 _dd_Geometry_RPC_lvRPCMod0xabb1b80.0
    7 9 _dd_Geometry_RPC_lvRPCMod_pvRPCFoam0xabb1b48.0
    8 10 _dd_Geometry_RPC_lvRPCFoam0xabb1778.0
    5 91 _dd_Geometry_Sites_lvNearHallTop_pvNearTeleRpc_pvNearTeleRpc_20xabc3800.0
    6 92 _dd_Geometry_RPC_lvRPCMod0xabb1b80.1
    7 93 _dd_Geometry_RPC_lvRPCMod_pvRPCFoam0xabb1b48.1
    8 94 _dd_Geometry_RPC_lvRPCFoam0xabb1778.1

                                                                                                                                                           
    sqlite> select count(*) from shape ;
    count(*)                                                                                                                                                                                                                                                        
    ---------------------------------------------------------------------------------------------                                                                                                                                                                   
    12229                                    

::

    116851     <node id="World0xad7b048">
    116852       <instance_geometry url="#WorldBox0xabaff60">
    116853         <bind_material>
    116854           <technique_common>
    116855             <instance_material symbol="WHITE" target="#_dd_Materials_Vacuum0x8b746a0"/>
    116856           </technique_common>
    116857         </bind_material>
    116858       </instance_geometry>
    116859       <node id="_dd_Structure_Sites_db-rock0xad7b188">
    116860         <matrix>
    116861                 -0.543174 0.83962 0 -16520
    116862 -0.83962 -0.543174 0 -802110
    116863 0 0 1 -2110
    116864 0.0 0.0 0.0 1.0
    116865 </matrix>
    116866         <instance_node url="#_dd_Geometry_Sites_lvNearSiteRock0xad7af08"/>
    116867       </node>
    116868     </node>


    116824     <node id="_dd_Geometry_Sites_lvNearSiteRock0xad7af08">    ########### LV OMITTED FROM THE VRML2 SHAPE LIST 
    116825       <instance_geometry url="#near_rock0xabafe30">
    116826         <bind_material>
    116827           <technique_common>
    116828             <instance_material symbol="WHITE" target="#_dd_Materials_Rock0x8b58188"/>
    116829           </technique_common>
    116830         </bind_material>
    116831       </instance_geometry>
    116832       <node id="_dd_Geometry_Sites_lvNearSiteRock_pvNearHallTop0xad7ad70">    #### PV INCLUDED IN VRML2
    116833         <matrix>
    116834                 1 0 0 2500
    116835 0 1 0 -500
    116836 0 0 1 7500
    116837 0.0 0.0 0.0 1.0
    116838 </matrix>
    116839         <instance_node url="#_dd_Geometry_Sites_lvNearHallTop0xabc3670"/>
    116840       </node>
    116841       <node id="_dd_Geometry_Sites_lvNearSiteRock_pvNearHallBot0xad7b0b0">    #### SIBLING PV INCLUDED IN VRML2
    116842         <matrix>
    116843                 1 0 0 0
    116844 0 1 0 0
    116845 0 0 1 -5150
    116846 0.0 0.0 0.0 1.0
    116847 </matrix>
    116848         <instance_node url="#_dd_Geometry_Sites_lvNearHallBot0xad7a618"/>
    116849       </node>
    116850     </node>


Looks to be a pattern that the LV referenced by instance_node are skipped in the VRML2 list.

  * yes, the VRML2 has just the PV 

::

    sqlite> select id, name from shape where name like '/dd/Geometry/Sites/lvNearSiteRock%' ;
    id          name                                                                                                
    ----------  ---------------------------------------------------------------------------------------------       
    2           /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000                                                
    3147        /dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot.1001                                                
    sqlite> 

::

    sqlite> select id, name from shape where name like '/dd/Geometry/Sites/lvNearHall%' ;
    id          name                                                                                                
    ----------  ---------------------------------------------------------------------------------------------       
    3           /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover.1000                                                
    4           /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1.1                                    
    46          /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:2.2                                    
    88          /dd/Geometry/Sites/lvNearHallTop#pvNearRPCRoof.1003                                                 
    2357        /dd/Geometry/Sites/lvNearHallTop#pvNearRPCSptRoof.1004                                              
    3148        /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead.1000                                                
    12221       /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab1.1001                         
    12222       /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab2.1002                         
    12223       /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab3.1003                         
    12224       /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab4.1004                         
    12225       /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab5.1005                         
    12226       /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab6.1006                         
    12227       /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab7.1007                         
    12228       /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab8.1008                         
    12229       /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab9.1009                         
    sqlite> 





Compare daenames with wrlnames : tracing the id
----------------------------------------------------

Succeed to match the bases, but not the `.1001` extensions::

    echo "select rtrim(substr(name,0,instr(name,'.'))) from shape ;" | sqlite3 -noheader $(shapedb-path) > wrlnames.txt
    cat wrlnames.txt | cut -d" " -f1 > wrlnames.cut.txt    # get rid of bizarre whitespace padding 
    diff wrlnames.cut.txt daenames.txt   # they match 

The WRL names, are actually coming from `G4PhysicalVolumeModel::GetCurrentTag`

external/build/LCG/geant4.9.2.p01/source/visualization/VRML/src/G4VRML2SceneHandlerFunc.icc::

    182     // Current Model
    183     const G4VModel* pv_model  = GetModel();
    184     G4String pv_name = "No model";
    185         if (pv_model) pv_name = pv_model->GetCurrentTag() ;
    186 
    187     // VRML codes are generated below
    188 
    189     fDest << "#---------- SOLID: " << pv_name << "\n";


external/build/LCG/geant4.9.2.p01/source/visualization/modeling/include/G4VModel.hh::

    74   virtual G4String GetCurrentTag () const;
    75   // A tag which depends on the current state of the model.

::

    [blyth@cms01 source]$ find . -name '*.hh' -exec grep -H GetCurrentTag {} \;
    ./visualization/modeling/include/G4PhysicalVolumeModel.hh:  G4String GetCurrentTag () const;
    ./visualization/modeling/include/G4VModel.hh:  virtual G4String GetCurrentTag () const;


external/build/LCG/geant4.9.2.p01/source/visualization/modeling/include/G4PhysicalVolumeModel.hh::

     67 class G4PhysicalVolumeModel: public G4VModel {
     68 
     69 public: // With description
     70 
     71   enum {UNLIMITED = -1};
     72 
     73   enum ClippingMode {subtraction, intersection};
     74 
     75   class G4PhysicalVolumeNodeID {
     76   public:
     77     G4PhysicalVolumeNodeID
     78     (G4VPhysicalVolume* pPV = 0, G4int iCopyNo = 0, G4int depth = 0):
     79       fpPV(pPV), fCopyNo(iCopyNo), fNonCulledDepth(depth) {}
     80     G4VPhysicalVolume* GetPhysicalVolume() const {return fpPV;}
     81     G4int GetCopyNo() const {return fCopyNo;}
     82     G4int GetNonCulledDepth() const {return fNonCulledDepth;}
     83     G4bool operator< (const G4PhysicalVolumeNodeID& right) const;
     84   private:
     85     G4VPhysicalVolume* fpPV;
     86     G4int fCopyNo;
     87     G4int fNonCulledDepth;
     88   };
     89   // Nested class for identifying physical volume nodes.
     ...
     205   G4VPhysicalVolume* fpCurrentPV;    // Current physical volume.

Suspect the CopyNo should hail from::

    geometry/volumes/src/G4PVPlacement.cc
    geometry/volumes/include/G4PVPlacement.hh


G4PhysicalVolumeNodeID::

    [blyth@cms01 source]$ find . -name '*.cc' -exec grep -H G4PhysicalVolumeNodeID {} \;
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:G4bool G4PhysicalVolumeModel::G4PhysicalVolumeNodeID::operator<
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:  (const G4PhysicalVolumeModel::G4PhysicalVolumeNodeID& right) const
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:  (std::ostream& os, const G4PhysicalVolumeModel::G4PhysicalVolumeNodeID node)
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:    (G4PhysicalVolumeNodeID(fpCurrentPV,copyNo,fCurrentDepth));
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:      (G4PhysicalVolumeNodeID(fpCurrentPV,copyNo,fCurrentDepth));
    ./visualization/Tree/src/G4ASCIITreeSceneHandler.cc:  typedef G4PhysicalVolumeModel::G4PhysicalVolumeNodeID PVNodeID;
    ./visualization/Tree/src/G4VTreeSceneHandler.cc:  typedef G4PhysicalVolumeModel::G4PhysicalVolumeNodeID PVNodeID;
    ./visualization/HepRep/src/G4HepRepFileSceneHandler.cc:                 typedef G4PhysicalVolumeModel::G4PhysicalVolumeNodeID PVNodeID;
    ./visualization/XXX/src/G4XXXSGSceneHandler.cc:    typedef G4PhysicalVolumeModel::G4PhysicalVolumeNodeID PVNodeID;
    ./visualization/OpenInventor/src/G4OpenInventorSceneHandler.cc:    typedef G4PhysicalVolumeModel::G4PhysicalVolumeNodeID PVNodeID;
    [blyth@cms01 source]$ 

PVPath::

    [blyth@cms01 source]$ find . -name '*.cc' -exec grep -l PVPath {} \;
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc
    ./visualization/Tree/src/G4ASCIITreeSceneHandler.cc
    ./visualization/Tree/src/G4VTreeSceneHandler.cc
    ./visualization/HepRep/src/G4HepRepFileSceneHandler.cc
    ./visualization/XXX/src/G4XXXSGSceneHandler.cc
    ./visualization/OpenInventor/src/G4OpenInventorSceneHandler.cc


external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4PhysicalVolumeModel.cc::

    181 G4String G4PhysicalVolumeModel::GetCurrentTag () const
    182 {
    183   if (fpCurrentPV) {
    184     std::ostringstream o;
    185     o << fpCurrentPV -> GetCopyNo ();
    186     return fpCurrentPV -> GetName () + "." + o.str();
    187   }
    188   else {
    189     return "WARNING: NO CURRENT VOLUME - global tag is " + fGlobalTag;
    190   }
    191 }



