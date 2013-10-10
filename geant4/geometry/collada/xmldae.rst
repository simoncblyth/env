XMLDAE
=======

Raw Node tree has the lv as well as pv, wherese VRML2 tree has only pv ?


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


::

    sqlite> select name from shape limit 10 ;
    name                                                                                                                                                                                                                                                            
    ---------------------------------------------------------------------------------------------                                                                                                                                                                   
    /dd/Structure/Sites/db-rock.1000                                                                                                                                                                                                                                
    /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000                                                                                                                                                                                                            
    /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover.1000                                                                                                                                                                                                            
    /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1.1                                                                                                                                                                                                
    /dd/Geometry/RPC/lvRPCMod#pvRPCFoam.1000                                                                                                                                                                                                                        
    /dd/Geometry/RPC/lvRPCFoam#pvBarCham14Array#pvBarCham14ArrayOne:1#pvBarCham14Unit.1                                                                                                                                                                             
    /dd/Geometry/RPC/lvRPCBarCham14#pvRPCGasgap14.1000                                                                                                                                                                                                              
    /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit.1                                                                                                                                                                               
    /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit.2                                                                                                                                                                               
    /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:3#pvStrip14Unit.3                                                                                                                                                                               
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



Does is make more sense to pass the matrix ?

::

    0   0    World0xad7b048.0                                                                                      1    tgt:_dd_Materials_Vacuum0x8b746a0  ref:None matrix:None 
    1   1    _dd_Structure_Sites_db-rock0xad7b188.0                                                                1    tgt:None  ref:#_dd_Geometry_Sites_lvNearSiteRock0xad7af08 matrix:-0.543174 0.83962 0 -16520, -0.83962 -0.543174 0 -802110, 0 0 1 -2110, 0.0 0.0 0.0 1.0 
    2   2    _dd_Geometry_Sites_lvNearSiteRock0xad7af08.0                                                          2    tgt:_dd_Materials_Rock0x8b58188  ref:None matrix:None 
    3   3    _dd_Geometry_Sites_lvNearSiteRock_pvNearHallTop0xad7ad70.0                                            1    tgt:None  ref:#_dd_Geometry_Sites_lvNearHallTop0xabc3670 matrix:1 0 0 2500, 0 1 0 -500, 0 0 1 7500, 0.0 0.0 0.0 1.0 
    4   4    _dd_Geometry_Sites_lvNearHallTop0xabc3670.0                                                           5    tgt:_dd_Materials_Air0x8b28278  ref:None matrix:None 
    5   5    _dd_Geometry_Sites_lvNearHallTop_pvNearTopCover0xabc3390.0                                            1    tgt:None  ref:#_dd_Geometry_PoolDetails_lvNearTopCover0xabaffe8 matrix:1 0 0 -2500, 0 1 0 500, 0 0 1 -7478, 0.0 0.0 0.0 1.0 
    6   6    _dd_Geometry_PoolDetails_lvNearTopCover0xabaffe8.0                                                    0    tgt:_dd_Materials_PPE0x8b066b8  ref:None matrix:None 
    5   7    _dd_Geometry_Sites_lvNearHallTop_pvNearTeleRpc_pvNearTeleRpc_10xabc36c8.0                             1    tgt:None  ref:#_dd_Geometry_RPC_lvRPCMod0xabb1b80 matrix:0.99995 -0.0100372 0 -2560.55, 0.0100372 0.99995 0 -5305.87, 0 0 1 -4706.1, 0.0 0.0 0.0 1.0 
    6   8    _dd_Geometry_RPC_lvRPCMod0xabb1b80.0                                                                  1    tgt:_dd_Materials_Aluminium0x8b291b8  ref:None matrix:None 
    7   9    _dd_Geometry_RPC_lvRPCMod_pvRPCFoam0xabb1b48.0                                                        1    tgt:None  ref:#_dd_Geometry_RPC_lvRPCFoam0xabb1778 matrix:1 0 0 -10, 0 1 0 5, 0 0 1 0, 0.0 0.0 0.0 1.0 
    8   10   _dd_Geometry_RPC_lvRPCFoam0xabb1778.0                                                                 4    tgt:_dd_Materials_Foam0x8b28a98  ref:None matrix:None 
    5   91   _dd_Geometry_Sites_lvNearHallTop_pvNearTeleRpc_pvNearTeleRpc_20xabc3800.0                             1    tgt:None  ref:#_dd_Geometry_RPC_lvRPCMod0xabb1b80 matrix:-0.999932 -0.011669 0 -2508.09, 0.011669 -0.999932 0 6048.3, 0 0 1 -4667.34, 0.0 0.0 0.0 1.0 
    6   92   _dd_Geometry_RPC_lvRPCMod0xabb1b80.1                                                                  1    tgt:_dd_Materials_Aluminium0x8b291b8  ref:None matrix:None 
    7   93   _dd_Geometry_RPC_lvRPCMod_pvRPCFoam0xabb1b48.1                                                        1    tgt:None  ref:#_dd_Geometry_RPC_lvRPCFoam0xabb1778 matrix:1 0 0 -10, 0 1 0 5, 0 0 1 0, 0.0 0.0 0.0 1.0 
    8   94   _dd_Geometry_RPC_lvRPCFoam0xabb1778.1                                                                 4    tgt:_dd_Materials_Foam0x8b28a98  ref:None matrix:None 






