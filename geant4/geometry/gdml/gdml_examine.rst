GDML Examination
==================

.. contents:: :local:

TODO
----

#. try to create the geometry Russian doll hierarchy (starting from World) from the GDML parse, to grok the format


physvol
--------

::

     2172   <structure>
     2173     <volume name="/dd/Geometry/PoolDetails/lvNearTopCover0xbad46a0">
     2174       <materialref ref="/dd/Materials/PPE0xb8310e0"/>
     2175       <solidref ref="near_top_cover_box0xbad4490"/>
     2176     </volume>
     2177     <volume name="/dd/Geometry/RPC/lvRPCStrip0xb839910">
     2178       <materialref ref="/dd/Materials/MixGas0xbad5d28"/>
     2179       <solidref ref="RPCStrip0xb751cc0"/>
     2180     </volume>
     2181     <volume name="/dd/Geometry/RPC/lvRPCGasgap140xb7491f8">
     2182       <materialref ref="/dd/Materials/Air0xb830740"/>
     2183       <solidref ref="RPCGasgap140xbad5938"/>
     2184       <physvol name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit0xbc1e930">
     2185         <volumeref ref="/dd/Geometry/RPC/lvRPCStrip0xb839910"/>
     2186         <position name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit0xbc1e930_pos" unit="mm" x="-910" y="0" z="0"/>
     2187         <rotation name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit0xbc1e930_rot" unit="deg" x="0" y="0" z="-90"/>
     2188       </physvol>
     2189       <physvol name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit0xbc1f8b8">
     2190         <volumeref ref="/dd/Geometry/RPC/lvRPCStrip0xb839910"/>
     2191         <position name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit0xbc1f8b8_pos" unit="mm" x="-650" y="0" z="0"/>
     2192         <rotation name="/dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit0xbc1f8b8_rot" unit="deg" x="0" y="0" z="-90"/>
     2193       </physvol>


Divide by 2 for open/close XML tags::

    [blyth@belle7 gdml]$ grep physvol $LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml | echo $(( $(wc -l)/2 ))
    5642

::

    simon:gdml blyth$ echo "select count(*) from physvol ;" | sqlite3 $LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml.db
    5642      

Matches the distinct name count in shapedb from vrml2 file::

    sqlite> select count(*) from shape ;
    12229     
    sqlite> select count(distinct(name)) from shape ;
    5642                 
    sqlite> select count(distinct(hash)) from shape ;
    12223         
    sqlite> select count(distinct(substr(name,0,instr(name,'.')))) from shape  ;
    5642      

ROOT Load of GDML, counts 12230 nodes matching the VRML2 shape count (assume extra 1 is world that was culled in VRML2)::


    root [0]  TGeoManager::Import("/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml")
    Info in <TGeoManager::Import>: Reading geometry from file: /data1/env/local/env/geant4/geometry/gdml/g4_01.gdml
    Info in <TGeoManager::TGeoManager>: Geometry Geometry, default geometry created
    Error: Unsupported GDML Tag Used :isotope. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :atom. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :fraction. Please Check Geometry/Schema.
    ...
    Info in <TGeoManager::SetTopVolume>: Top volume is World. Master volume is World
    Info in <TGeoManager::CheckGeometry>: Fixing runtime shapes...
    Info in <TGeoManager::CheckGeometry>: ...Nothing to fix
    Info in <TGeoManager::CloseGeometry>: Counting nodes...
    Info in <TGeoManager::Voxelize>: Voxelizing...
    Info in <TGeoManager::CloseGeometry>: Building cache...
    Info in <TGeoNavigator::BuildCache>: --- Maximum geometry depth set to 100
    Info in <TGeoManager::CloseGeometry>: 12230 nodes/ 249 volume UID's in default geometry
    Info in <TGeoManager::CloseGeometry>: ----------------modeler ready----------------
    (class TGeoManager*)0x89be148
    root [1] 


VRML2 GDML mismatch ?
----------------------

Hmm, was expecting the phyvol to correspond to the 12k ?

  * so what are all the other VRML2 shapes `12229-5642=6587`
  * clearly the algorithmically transformed/duplicated volumes, eg 2 AD, 672 PMT
  * the structure of the GDML heirarchy of volume/physvol/solid is doing the duplicatation that VRML2 spelt out in shapes

::

    sqlite> select count(*) as N, name, max(dx)||":"||max(dy)||":"||max(dz) as dxyz from xshape group by name having N>1; 
    N           name                                                                                                                                                                                                      dxyz                        
    ----------  ---------------------------------------------------------------------------------------------                                                                                                             ----------------------------
    2           /dd/Geometry/AD/lvADE#pvAdVertiCableTray.1005                                                                                                                                                             66.2000000000007:72.0:5000.0
    2           /dd/Geometry/AD/lvADE#pvCenterCalibE.1001                                                                                                                                                                 798.1:798.0:688.98          
    2           /dd/Geometry/AD/lvADE#pvElectricalDistributionBoxE.1007                                                                                                                                                   608.200000000001:609.0:260.3
    2           /dd/Geometry/AD/lvADE#pvGCatCalibE.1004                         
    ...
    2           /dd/Geometry/AD/lvLSO#pvIAV.1000                                                                                                                                                                          3126.0:3126.0:3174.49    
    2           /dd/Geometry/AD/lvOAV#pvLSO.1000                                                                                                                                                                          3958.9:3959.0:4076.53       
    ...
    672         /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000                                                                                                                                                           297.0:297.0:292.5           
    672         /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom.1001                                                                                                                                                     196.100000000002:196.0:196.2
    672         /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000                                                                                                                                                    196.100000000002:196.0:196.2
    672         /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode.1002                                                                                                                                                     174.900000000001:175.0:166.0





Maybe the AD split happedns here
----------------------------------

::

    26985       <physvol name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:1#pvLegInOWSUnit0xb905030">
    26986         <volumeref ref="/dd/Geometry/PoolDetails/lvLegInOWSTub0xc4c2208"/>
    26987         <position name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:1#pvLegInOWSUnit0xb905030_pos" unit="mm" x="4713" y="1842" z="-4456"/>
    26988       </physvol>
    26989       <physvol name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:2#pvLegInOWSUnit0xc12ecb0">
    26990         <volumeref ref="/dd/Geometry/PoolDetails/lvLegInOWSTub0xc4c2208"/>
    26991         <position name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:2#pvLegInOWSUnit0xc12ecb0_pos" unit="mm" x="1029" y="1842" z="-4456"/>
    26992         <rotation name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:2#pvLegInOWSUnit0xc12ecb0_rot" unit="deg" x="0" y="0" z="-90"/>
    26993       </physvol>
    26994       <physvol name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:3#pvLegInOWSUnit0xb76f880">
    26995         <volumeref ref="/dd/Geometry/PoolDetails/lvLegInOWSTub0xc4c2208"/>
    26996         <position name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:3#pvLegInOWSUnit0xb76f880_pos" unit="mm" x="1029" y="-1842" z="-4456"/>
    26997         <rotation name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:3#pvLegInOWSUnit0xb76f880_rot" unit="deg" x="0" y="0" z="-180"/>
    26998       </physvol>
    26999       <physvol name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:4#pvLegInOWSUnit0xc4c26e0">
    27000         <volumeref ref="/dd/Geometry/PoolDetails/lvLegInOWSTub0xc4c2208"/>
    27001         <position name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:4#pvLegInOWSUnit0xc4c26e0_pos" unit="mm" x="4713" y="-1842" z="-4456"/>
    27002         <rotation name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE1OWSLegs#pvLegInOWS:4#pvLegInOWSUnit0xc4c26e0_rot" unit="deg" x="0" y="0" z="90"/>
    27003       </physvol>
    27004       <physvol name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE2OWSLegs#pvLegInOWS:1#pvLegInOWSUnit0xc4c27e0">
    27005         <volumeref ref="/dd/Geometry/PoolDetails/lvLegInOWSTub0xc4c2208"/>
    27006         <position name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE2OWSLegs#pvLegInOWS:1#pvLegInOWSUnit0xc4c27e0_pos" unit="mm" x="-1029" y="1842" z="-4456"/>
    27007       </physvol>
    27008       <physvol name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE2OWSLegs#pvLegInOWS:2#pvLegInOWSUnit0xc4c2890">
    27009         <volumeref ref="/dd/Geometry/PoolDetails/lvLegInOWSTub0xc4c2208"/>
    27010         <position name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE2OWSLegs#pvLegInOWS:2#pvLegInOWSUnit0xc4c2890_pos" unit="mm" x="-4713" y="1842" z="-4456"/>
    27011         <rotation name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE2OWSLegs#pvLegInOWS:2#pvLegInOWSUnit0xc4c2890_rot" unit="deg" x="0" y="0" z="-90"/>
    27012       </physvol>
    27013       <physvol name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE2OWSLegs#pvLegInOWS:3#pvLegInOWSUnit0xc4c2990">
    27014         <volumeref ref="/dd/Geometry/PoolDetails/lvLegInOWSTub0xc4c2208"/>
    27015         <position name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE2OWSLegs#pvLegInOWS:3#pvLegInOWSUnit0xc4c2990_pos" unit="mm" x="-4713" y="-1842" z="-4456"/>
    27016         <rotation name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE2OWSLegs#pvLegInOWS:3#pvLegInOWSUnit0xc4c2990_rot" unit="deg" x="0" y="0" z="-180"/>
    27017       </physvol>
    27018       <physvol name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE2OWSLegs#pvLegInOWS:4#pvLegInOWSUnit0xc4c2a90">
    27019         <volumeref ref="/dd/Geometry/PoolDetails/lvLegInOWSTub0xc4c2208"/>
    27020         <position name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE2OWSLegs#pvLegInOWS:4#pvLegInOWSUnit0xc4c2a90_pos" unit="mm" x="-1029" y="-1842" z="-4456"/>
    27021         <rotation name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearADE2OWSLegs#pvLegInOWS:4#pvLegInOWSUnit0xc4c2a90_rot" unit="deg" x="0" y="0" z="90"/>
    27022       </physvol>





Pick a volume to see whats going down in the GDML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:~ blyth$ echo "select id,uname,name from physvol where name like '/dd/Geometry/AD/lvOAV#pvLSO%' ;" |  sqlite3 $LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml.db
    377 /dd/Geometry/AD/lvOAV#pvLSO0xba431e8  /dd/Geometry/AD/lvOAV#pvLSO

::

     4064     <volume name="/dd/Geometry/AD/lvOAV0xb78abc0">
     4065       <materialref ref="/dd/Materials/Acrylic0xb85a2d8"/>
     4066       <solidref ref="oav0xb78bb68"/>
     4067       <physvol name="/dd/Geometry/AD/lvOAV#pvLSO0xba431e8">
     4068         <volumeref ref="/dd/Geometry/AD/lvLSO0xbac5950"/>
     4069         <position name="/dd/Geometry/AD/lvOAV#pvLSO0xba431e8_pos" unit="mm" x="0" y="0" z="31.5"/>
     4070       </physvol>
     4071       <physvol name="/dd/Geometry/AD/lvOAV#pvOcrGdsLsoInOav0xbc16ab8">
     4072         <volumeref ref="/dd/Geometry/AdDetails/lvOcrGdsLsoInOav0xbc15650"/>
     4073         <position name="/dd/Geometry/AD/lvOAV#pvOcrGdsLsoInOav0xbc16ab8_pos" unit="mm" x="0" y="0" z="2024.81037191693"/>
     4074       </physvol>
     4075       <physvol name="/dd/Geometry/AD/lvOAV#pvOcrCalLsoInOav0xbc16860">
     4076         <volumeref ref="/dd/Geometry/AdDetails/lvOcrCalLsoInOav0xbc172a0"/>
     4077         <position name="/dd/Geometry/AD/lvOAV#pvOcrCalLsoInOav0xbc16860_pos" unit="mm" x="0" y="0" z="2024.81037191693"/>
     4078       </physvol>
     4079     </volume>






Compare VRML2 shape names with physvol names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:gdml blyth$ vi ~/.sqliterc  # avoid truncation with ".width 256" 
    simon:gdml blyth$ echo "select distinct(name) from physvol order by name ;" | sqlite3 $LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml.db > physvol_names

    simon:gdml blyth$ scp N:/data1/env/local/env/geant4/geometry/vrml2/g4_01.db /usr/local/env/geant4/geometry/vrml2/
    simon:gdml blyth$ echo "select distinct(name) from shape order by name ;" | sqlite3 $LOCAL_BASE/env/geant4/geometry/vrml2/g4_01.db > shape_names


Apparently the VRML2 names have been de-duped via the `.1005` etc::

    simon:gdml blyth$ diff -y physvol_names shape_names | head -10 
    name                                                            name                                                         
    -------------------------------------------------------------   -------------------------------------------------------------
    /dd/Geometry/AD/lvADE#pvAdVertiCableTray                      | /dd/Geometry/AD/lvADE#pvAdVertiCableTray.1005                
    /dd/Geometry/AD/lvADE#pvCenterCalibE                          | /dd/Geometry/AD/lvADE#pvCenterCalibE.1001                    
    /dd/Geometry/AD/lvADE#pvElectricalDistributionBoxE            | /dd/Geometry/AD/lvADE#pvElectricalDistributionBoxE.1007      
    /dd/Geometry/AD/lvADE#pvGCatCalibE                            | /dd/Geometry/AD/lvADE#pvGCatCalibE.1004                      
    /dd/Geometry/AD/lvADE#pvGasDistributionBoxE                   | /dd/Geometry/AD/lvADE#pvGasDistributionBoxE.1006             
    /dd/Geometry/AD/lvADE#pvMOClarityBoxE                         | /dd/Geometry/AD/lvADE#pvMOClarityBoxE.1008                   
    /dd/Geometry/AD/lvADE#pvOffCenterCalibE                       | /dd/Geometry/AD/lvADE#pvOffCenterCalibE.1003                 
    /dd/Geometry/AD/lvADE#pvOflTnkContainer                       | /dd/Geometry/AD/lvADE#pvOflTnkContainer.1002                 

    simon:gdml blyth$ diff -y physvol_names shape_names | tail -10 
    /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHal | /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHal
    /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead               | /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead.1000         
    /dd/Geometry/Sites/lvNearHallTop#pvNearRPCRoof                | /dd/Geometry/Sites/lvNearHallTop#pvNearRPCRoof.1003          
    /dd/Geometry/Sites/lvNearHallTop#pvNearRPCSptRoof             | /dd/Geometry/Sites/lvNearHallTop#pvNearRPCSptRoof.1004       
    /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc: | /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:
    /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc: | /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:
    /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover               | /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover.1000         
    /dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot               | /dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot.1001         
    /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop               | /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000         
    /dd/Structure/Sites/db-rock                                   | /dd/Structure/Sites/db-rock.1000                             
    simon:gdml blyth$ 


Using a very recent sqlite3 (for the *instr* function) chop off the VRML2 de-dupe::

    [blyth@belle7 gdml]$ sqlite3-path
    [blyth@belle7 gdml]$ sqlite3 -version
    -- Loading resources from /home/blyth/.sqliterc

    3.8.0.2 2013-09-03 17:11:13 7dd4968f235d6e1ca9547cda9cf3bd570e1609ef

    [blyth@belle7 gdml]$ vi ~/.sqliterc
    [blyth@belle7 gdml]$ echo "select distinct(substr(name,0,instr(name,'.'))) as name from shape order by name ;" | sqlite3 $LOCAL_BASE/env/geant4/geometry/vrml2/g4_01.db  > shape_names_2
    simon:gdml blyth$ scp N:env/geant4/geometry/gdml/shape_names_2 .
    simon:gdml blyth$ diff physvol_names shape_names_2
    simon:gdml blyth$ 

Thus there are exactly the same VRML2 de-duped distinct shape names as GDML physvol names.



volume names
---------------

Only 249 `volume/\@name` in the GDML::

    simon:gdml blyth$ grep volume\  $LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml
        <volume name="/dd/Geometry/PoolDetails/lvNearTopCover0xbad46a0">
        <volume name="/dd/Geometry/RPC/lvRPCStrip0xb839910">
        <volume name="/dd/Geometry/RPC/lvRPCGasgap140xb7491f8">
        <volume name="/dd/Geometry/RPC/lvRPCBarCham140xbad5978">
        <volume name="/dd/Geometry/RPC/lvRPCGasgap230xb83ee78">
        ...
        <volume name="/dd/Geometry/RadSlabs/lvNearRadSlab80xc505ff8">
        <volume name="/dd/Geometry/RadSlabs/lvNearRadSlab90xc632930">
        <volume name="/dd/Geometry/Sites/lvNearHallBot0xb7dd4a8">
        <volume name="/dd/Geometry/Sites/lvNearSiteRock0xb82e578">
        <volume name="World0xc6337a8">
    simon:gdml blyth$ grep volume\  $LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml | wc -l
         249

    simon:gdml blyth$ echo "select count(*) from volume ;" | sqlite3 /usr/local/env/geant4/geometry/gdml/g4_01.gdml.db 
    249           



