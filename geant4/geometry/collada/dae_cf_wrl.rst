DAE cf WRL
============

.. contents:: :local:

Questions
-----------

#. why do the boolean volume vertices depend on which export is done first DAE or WRL  ?

   * this causes between process exports to differ (for boolean volumes), if they differ in export ordering 

#. do WRL and DAE face indices correspond ? 


Answered Questions
--------------------

#. why are AD PMTs in the iPad and meshlab renders all pointing in same direction, and not towards center ?

   * that render was without the ``monkey_matrix_load`` fix ? 
   * YES, looking at WRL render in meshlab shows expected PMT directions
   * after fixing invrot in DAE, that is also OK : I was wrong in thinking that the daenode 
     subcopy was doing this 

#. VRML2 Y is being rounded to the nearest 1 mm, and X often to nearest 0.1 mm

   * applying the g4-vrml- fix get max (and avg) offsets down to 0.03 mm 


Create DAE DB
---------------

Preexisting DB are deleted::

    daedb.py --daepath '$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae'    # 2nd generatation, from a GDML import  
    daedb.py --daepath '$LOCAL_BASE/env/geant4/geometry/gdml/g4_10.dae'    # 1st generation, direct from NuWa/Geant4 detdesc creation 

Note that with system python but NuWa PYTHONPATH get::

      File "/data1/env/system/python/Python-2.5.1/lib/python2.5/site-packages/pycollada-0.4-py2.5.egg/collada/xmlutil.py", line 11, in <module>
        from xml.etree import ElementTree as etree
      ImportError: No module named etree

Workaround is to not be in NuWa env or skip the PYTHONPATH::

    [blyth@belle7 ~]$ PYTHONPATH=  daedb.py --daepath '$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae'  



Attaching two DB in sqlite3
------------------------------
::

    cat << EOS > /tmp/dae_cf_wrl.sql 
    attach database "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.db" as dae ;
    attach database "$LOCAL_BASE/env/geant4/geometry/vrml2/g4_01.db" as wrl ;
    .databases
    EOS
    sqlite3 -init /tmp/dae_cf_wrl.sql 

Function for that::

    simon:e blyth$ t g4dae-cf
    g4dae-cf is a function
    g4dae-cf () 
    { 
        local sql=$(g4dae-cf-path);
        mkdir -p $(dirname $sql);
        $FUNCNAME- > $sql;
        sqlite3 -init $sql
    }
    simon:e blyth$ g4dae-cf-
    attach database "/usr/local/env/geant4/geometry/xdae/g4_01.db" as dae ;
    attach database "/usr/local/env/geant4/geometry/vrml2/g4_01.db" as wrl ;
    .databases
    simon:e blyth$ 


Consistent Volume counts
--------------------------

::

    sqlite> select count(*) from dae.geom ;
    12230                                                                                                                                                                                                                                                         
    sqlite> select count(*) from wrl.shape ;   # world volume was culled for this wrl export
    12229
    sqlite> select count(*) from wrl.xshape ;  # world volume was culled for this wrl export , xshape is faster than shape as smaller
    12229


Volume index join between WRL and DAE tables in separate DB
------------------------------------------------------------

::

    sqlite> select d.idx, w.name, d.name from wrl.xshape w inner join dae.geom d on w.sid = d.idx limit 10 ;
    idx         name                                                                                                  name                                                                                                
    ----------  ---------------------------------------------------------------------------------------------         ---------------------------------------------------------------------------------------------       
    1           /dd/Structure/Sites/db-rock.1000                                                                      __dd__Structure__Sites__db-rock0xaa8b0f8.0                                                          
    2           /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000                                                  __dd__Geometry__Sites__lvNearSiteRock--pvNearHallTop0xaa8ace0.0                                     
    3           /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover.1000                                                  __dd__Geometry__Sites__lvNearHallTop--pvNearTopCover0xa8d3790.0                                     
    4           /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1.1                                      __dd__Geometry__Sites__lvNearHallTop--pvNearTeleRpc--pvNearTeleRpc..10xa8d3ac8.0                    
    5           /dd/Geometry/RPC/lvRPCMod#pvRPCFoam.1000                                                              __dd__Geometry__RPC__lvRPCMod--pvRPCFoam0xa8c1d58.0                                                 
    6           /dd/Geometry/RPC/lvRPCFoam#pvBarCham14Array#pvBarCham14ArrayOne:1#pvBarCham14Unit.1                   __dd__Geometry__RPC__lvRPCFoam--pvBarCham14Array--pvBarCham14ArrayOne..1--pvBarCham14Unit0xa8c19e0.0
    7           /dd/Geometry/RPC/lvRPCBarCham14#pvRPCGasgap14.1000                                                    __dd__Geometry__RPC__lvRPCBarCham14--pvRPCGasgap140xa8c10f0.0                                       
    8           /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit.1                     __dd__Geometry__RPC__lvRPCGasgap14--pvStrip14Array--pvStrip14ArrayOne..1--pvStrip14Unit0xa8c02c0.0  
    9           /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit.2                     __dd__Geometry__RPC__lvRPCGasgap14--pvStrip14Array--pvStrip14ArrayOne..2--pvStrip14Unit0xa8c0390.0  
    10          /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:3#pvStrip14Unit.3                     __dd__Geometry__RPC__lvRPCGasgap14--pvStrip14Array--pvStrip14ArrayOne..3--pvStrip14Unit0xa8c08a0.0  

    sqlite> select count(*) from wrl.xshape w inner join dae.geom d on w.sid = d.idx  ;
    count(*)  
    ----------
    12229     


Vertex Count Discrepancy for 1688/12230 volumes (14 %)
--------------------------------------------------------

::

    sqlite> select count(*) from wrl.xshape w inner join dae.geom d on w.sid = d.idx where w.npo != d.nvertex ;
    1688              # ouch 14% of volumes have different vertex counts  

    sqlite> select count(*) from wrl.xshape w inner join dae.geom d on w.sid = d.idx where w.npo = d.nvertex ;
    10541     


With first generation DAE makes no difference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 gdml]$ g4dae-cf 10
    -- Loading resources from /data1/env/local/env/geant4/geometry/collada/g4dae-cf.sql
    seq  name             file                                                      
    ---  ---------------  ----------------------------------------------------------
    0    main                                                                       
    2    dae              /data1/env/local/env/geant4/geometry/gdml/g4_10.dae.db    
    3    wrl              /data1/env/local/env/geant4/geometry/vrml2/g4_01.db       

    SQLite version 3.8.0.2 2013-09-03 17:11:13
    Enter ".help" for instructions
    Enter SQL statements terminated with a ";"
    sqlite> 
    sqlite> select count(*) from wrl.xshape w inner join dae.geom d on w.sid = d.idx where w.npo != d.nvertex ;
    1688
    sqlite> 


* is the VRML2 first gen ? I thought they all were ?


Discrepancies grouped by geometry id
------------------------------------------

#. 34 shapes out of 249 are vertex count discrepant
#. all are discrepant in the same way : with same vertex counts for all instances of that geometry

::

    sqlite> select count(distinct(geoid)) from dae.geom ;   
    249

    sqlite> select d.geoid, group_concat(distinct(d.nvertex)) as dae_nvtx, group_concat(distinct(w.npo)) as wrl_npo, w.npo-d.nvertex, count(*) as N, group_concat(distinct(d.idx)) from wrl.xshape w inner join dae.geom d on w.sid = d.idx where w.npo != d.nvertex  group by d.geoid ;
    geoid                    dae_nvtx    wrl_npo     w.npo-d.nvertex  N           group_concat(distinct(d.idx))
    -----------------------  ----------  ----------  ---------------  ----------  -----------------------------
    AmCCo60AcrylicContainer  342         233         -109             6           4567,4655,4737,6227,6315,6397      # union of union
    AmCCo60Cavity            150         194         44               6           4568,4656,4738,6228,6316,6398      # u of u 
    IavTopRib                22          16          -6               16          3187,3188,3189,3190,3191,3192      # subtraction of subtraction
    LsoOflTnk                480         192         -288             2           4606,6266                          # u of u  
    OavTopRib                16          33          17               16          4497,4498,4499,4500,4501,4502      # s of s 
    OcrCalLso                49          98          49               2           4520,6180                          #    
    OcrCalLsoPrt             288         192         -96              2           4517,6177                    
    OcrGdsInLsoOfl           49          98          49               2           4516,6176                    
    OcrGdsLsoInOil           49          98          49               2           4514,6174                    
    OcrGdsLsoPrt             288         192         -96              2           4511,6171                    
    OcrGdsPrt                192         288         96               2           3165,4825                    
    OcrGdsTfbInLsoOfl        98          49          -49              2           4515,6175                    
    OflTnkContainer          344         366         22               2           4604,6264                    
    SstBotRib                15          35          20               16          4431,4432,4433,4434,4435,4436
    SstTopCirRibBase         48          34          -14              16          4465,4466,4467,4468,4469,4470
    SstTopHub                192         96          -96              2           4464,6124                    
    amcco60-source-assy      775         296         -479             6           4566,4654,4736,6226,6314,6396
    headon-pmt-assy          122         100         -22              12          4351,4358,4365,4372,4379,4386    # union
    headon-pmt-mount         192         96          -96              12          4357,4364,4371,4378,4385,4392    # union
    led-source-assy          778         629         -149             6           4540,4628,4710,6200,6288,6370
    led-source-shell         342         50          -292             6           4541,4629,4711,6201,6289,6371
    lso                      170         168         -2               2           3157,4817                        # union
    near-radslab-box-9       34          50          16               1           12229                        
    near_hall_top_dwarf      20          16          -4               1           2                            
    near_pentagon_iron_box   10          12          2                144         2389,2390,2391,2392,2393,2394
    near_pool_dead_box       50          34          -16              1           3148                         
    near_pool_liner_box      34          50          16               1           3149                         
    near_pool_ows_box        78          53          -25              1           3150                         
    near_top_cover_box       34          40          6                1           3                            
    pmt-hemi                 360         362         2                672         3199,3205,3211,3217,3223,3229
    pmt-hemi-vac             334         338         4                672         3200,3206,3212,3218,3224,3230
    source-assy              780         357         -423             6           4551,4639,4721,6211,6299,6381
    source-shell             342         50          -292             6           4552,4640,4722,6212,6300,6382
    wall-led-assy            316         360         44               6           4521,4524,4527,6181,6184,6187
    weight-shell             342         50          -292             36          4543,4547,4558,4562,4591,4595


Caution System SQLite3 is glacial on N
----------------------------------------

Multi-DB joins with system sqlite3 on N (SQLite version 3.3.6) taking minutes whereas
source sqlite3 (SQLite version 3.8.0.2 2013-09-03 17:11:13) takes a few seconds, just like on G.
Note cannot upgrade it as used by yum.

Dont use ``sqlite3`` instead ``sqlite3--``::

    [blyth@belle7 gdml]$ sqlite3-- -init  /data1/env/local/env/geant4/geometry/collada/g4dae-cf.sql
    -- Loading resources from /data1/env/local/env/geant4/geometry/collada/g4dae-cf.sql
    seq  name             file                                                      
    ---  ---------------  ----------------------------------------------------------
    0    main                                                                       
    2    dae              /data1/env/local/env/geant4/geometry/xdae/g4_01.dae.db    
    3    wrl              /data1/env/local/env/geant4/geometry/vrml2/g4_01.db       

    SQLite version 3.8.0.2 2013-09-03 17:11:13
    Enter ".help" for instructions
    Enter SQL statements terminated with a ";"
    sqlite> select count(*) from wrl.xshape w inner join dae.geom d on w.sid = d.idx  ;
    12229
    sqlite> 




Check GDML
------------

Sampling the GDML, all checked are unions or subtraction solids.

::

     1456     <union name="AmCCo60AcrylicContainer0xbb640b8">
     1457       <first ref="AcrylicCylinder+ChildForAmCCo60AcrylicContainer0xbb63c38"/>
     1458       <second ref="LowerAcrylicHemisphere0xbb648e8"/>
     1459       <position name="AmCCo60AcrylicContainer0xbb640b8_pos" unit="mm" x="0" y="0" z="-14.865"/>
     1460       <rotation name="AmCCo60AcrylicContainer0xbb640b8_rot" unit="deg" x="-90" y="0" z="0"/>
     1461     </union>

::

     1436     <union name="AmCCo60MainCavity+ChildForAmCCo60Cavity0xbb64188">
     1437       <first ref="AmCCo60MainCavity0xb91bd38"/>
     1438       <second ref="UpperAmCCo60SideCavity0xb91bfd0"/>
     1439       <position name="AmCCo60MainCavity+ChildForAmCCo60Cavity0xbb64188_pos" unit="mm" x="0" y="0" z="16.76"/>
     1440     </union>
     1441     <tube aunit="deg" deltaphi="360" lunit="mm" name="LowerAmCCo60SideCavity0xb91c1a0" rmax="6.35" rmin="0" startphi="0" z="3.8"/>
     1442     <union name="AmCCo60Cavity0xb91c2a0">
     1443       <first ref="AmCCo60MainCavity+ChildForAmCCo60Cavity0xbb64188"/>
     1444       <second ref="LowerAmCCo60SideCavity0xb91c1a0"/>
     1445       <position name="AmCCo60Cavity0xb91c2a0_pos" unit="mm" x="0" y="0" z="-16.76"/>
     1446     </union>


IavTopRib subtraction of subtraction::

      607     <subtraction name="IavTopRibBase-ChildForIavTopRib0xba42f70">
      608       <first ref="IavTopRibBase0xba428e0"/>
      609       <second ref="IavTopRibSidCut0xba42f30"/>
      610       <position name="IavTopRibBase-ChildForIavTopRib0xba42f70_pos" unit="mm" x="639.398817652391" y="0" z="40.875"/>
      611       <rotation name="IavTopRibBase-ChildForIavTopRib0xba42f70_rot" unit="deg" x="0" y="30" z="0"/>
      612     </subtraction>
      613     <cone aunit="deg" deltaphi="360" lunit="mm" name="IavTopRibBotCut0xba43130" rmax1="1520.39278882354" rmax2="100" rmin1="0" rmin2="0" startphi="0" z="74.4396317718873"/>
      614     <subtraction name="IavTopRib0xba43230">
      615       <first ref="IavTopRibBase-ChildForIavTopRib0xba42f70"/>
      616       <second ref="IavTopRibBotCut0xba43130"/>
      617       <position name="IavTopRib0xba43230_pos" unit="mm" x="-810.196394411769" y="0" z="-17.2801841140563"/>
      618     </subtraction>


lso union of cylinder and polycone::

      619     <tube aunit="deg" deltaphi="360" lunit="mm" name="lso_cyl0xb85b498" rmax="1982" rmin="0" startphi="0" z="3964"/>
      620     <polycone aunit="deg" deltaphi="360" lunit="mm" name="lso_polycone0xbbd58d0" startphi="0">
      621       <zplane rmax="1930" rmin="0" z="3964"/>
      622       <zplane rmax="125" rmin="0" z="4058.59604160589"/>
      623       <zplane rmax="50" rmin="0" z="4058.59604160589"/>
      624       <zplane rmax="50" rmin="0" z="4076.62074383385"/>
      625     </polycone>
      626     <union name="lso0xb85b048">
      627       <first ref="lso_cyl0xb85b498"/>
      628       <second ref="lso_polycone0xbbd58d0"/>
      629       <position name="lso0xb85b048_pos" unit="mm" x="0" y="0" z="-1982"/>
      630     </union>




Visual check of discrepants : lots of interesting shapes
----------------------------------------------------------


* http://belle7.nuu.edu.tw/dae/tree/4567.html  AmCCo60AcrylicContainer 

  * funny shape, looks like some internal triangles are scrubbed in WRL case

* http://belle7.nuu.edu.tw/dae/tree/4568.html  AmCCo60Cavity (Air)

  * concentric cylinders with inner one poking out, again internal triangles are not scrubbed

* http://belle7.nuu.edu.tw/dae/tree/3187.html  IavTopRib (Acrylic)
* http://belle7.nuu.edu.tw/dae/tree/4497.html  OavTopRib 

  * looks like a broken triangle

* http://belle7.nuu.edu.tw/dae/tree/4606.html LsoOflTnk 

  * wheel shape, concave

* http://belle7.nuu.edu.tw/dae/tree/4520.html OcrCalLso 
* http://belle7.nuu.edu.tw/dae/tree/4516.html OcrGdsInLsoOfl 

  * cylindrical, with tris inscribed into a circle at one end

* http://belle7.nuu.edu.tw/dae/tree/4517.html OcrCalLsoPrt 

  * complicated shape

* http://belle7.nuu.edu.tw/dae/tree/4511.html OcrGdsLsoPrt   

  * appears to have disconnected halo

* http://belle7.nuu.edu.tw/dae/tree/3165.html OcrGdsPrt 

  * with a hole 

* http://belle7.nuu.edu.tw/dae/tree/4515.html  OcrGdsTfbInLsoOfl 
 
  * disconnected disc

* http://belle7.nuu.edu.tw/dae/tree/4604.html OflTnkContainer 

  * dustbin lid

* http://belle7.nuu.edu.tw/dae/tree/4431.html SstBotRib 
* http://belle7.nuu.edu.tw/dae/tree/4465.html SstTopCirRibBase  

  * clamshell telephone offset from origin

* http://belle7.nuu.edu.tw/dae/tree/4464.html SstTopHub
* http://belle7.nuu.edu.tw/dae/tree/4566.html amcco60-source-assy
* http://belle7.nuu.edu.tw/dae/tree/4540.html led-source-assy 
* http://belle7.nuu.edu.tw/dae/tree/4551.html source-assy

  * 3 disconnected cylindal objs with a wire 

* http://belle7.nuu.edu.tw/dae/tree/4351.html headon-pmt-assy

  * parent is mineral oil 

* http://belle7.nuu.edu.tw/dae/tree/4357.html headon-pmt-mount  

  * with hole

* http://belle7.nuu.edu.tw/dae/tree/4541.html led-source-shell 
* http://belle7.nuu.edu.tw/dae/tree/4552.html source-shell 
* http://belle7.nuu.edu.tw/dae/tree/4543.html weight-shell

  * internal tris

* http://belle7.nuu.edu.tw/dae/tree/3157.html lso
* http://belle7.nuu.edu.tw/dae/tree/12229.html near-radslab-box-9
* http://belle7.nuu.edu.tw/dae/tree/2.html   near_hall_top_dwarf 

  * clearly a subtraction solid

* http://belle7.nuu.edu.tw/dae/tree/2389.html near_pentagon_iron_box  
* http://belle7.nuu.edu.tw/dae/tree/3148.html near_pool_dead_box   
* http://belle7.nuu.edu.tw/dae/tree/3149.html near_pool_liner_box 
* http://belle7.nuu.edu.tw/dae/tree/3150.html near_pool_ows_box   

  * many children

* http://belle7.nuu.edu.tw/dae/tree/3.html near_top_cover_box 
* http://belle7.nuu.edu.tw/dae/tree/3199.html  pmt-hemi 
* http://belle7.nuu.edu.tw/dae/tree/3200.html  pmt-hemi-vac (only child of 3199)
* http://belle7.nuu.edu.tw/dae/tree/4521.html wall-led-assy   

  * cylinder touching a sphere


Same process DAE then WRL : vertex counts match
-------------------------------------------------

::

    simon:gdml_dae_wrl blyth$ sqlite3 -init cf.sql
    -- Loading resources from cf.sql
    seq  name             file                                                      
    ---  ---------------  ----------------------------------------------------------
    0    main                                                                       
    2    dae              /usr/local/env/geant4/geometry/gdml/gdml_dae_wrl/g4_00.dae
    3    wrl              /usr/local/env/geant4/geometry/gdml/gdml_dae_wrl/g4_00.wrl


    sqlite> select d.idx, w.name, d.name from wrl.geom w inner join dae.geom d on w.idx = d.idx + 1 limit 10 ;
    idx         name                                                                                                  name                                                                                                
    ----------  ---------------------------------------------------------------------------------------------         ---------------------------------------------------------------------------------------------       
    0           Universe.0                                                                                            top.0                                                                                               
    1           /dd/Structure/Sites/db-rock.1000                                                                      __dd__Structure__Sites__db-rock0xc109960.0                                                          
    2           /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000                                                  __dd__Geometry__Sites__lvNearSiteRock--pvNearHallTop0xb4f3440.0                                     
    3           /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover.1000                                                  __dd__Geometry__Sites__lvNearHallTop--pvNearTopCover0xb1ff6c8.0                                     
    4           /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1.1                                      __dd__Geometry__Sites__lvNearHallTop--pvNearTeleRpc--pvNearTeleRpc..10xb3dee08.0                    
    5           /dd/Geometry/RPC/lvRPCMod#pvRPCFoam.1000                                                              __dd__Geometry__RPC__lvRPCMod--pvRPCFoam0xb2fc9e0.0                                                 
    6           /dd/Geometry/RPC/lvRPCFoam#pvBarCham14Array#pvBarCham14ArrayOne:1#pvBarCham14Unit.1                   __dd__Geometry__RPC__lvRPCFoam--pvBarCham14Array--pvBarCham14ArrayOne..1--pvBarCham14Unit0xb6cd140.0
    7           /dd/Geometry/RPC/lvRPCBarCham14#pvRPCGasgap14.1000                                                    __dd__Geometry__RPC__lvRPCBarCham14--pvRPCGasgap140xb6cc3e8.0                                       
    8           /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit.1                     __dd__Geometry__RPC__lvRPCGasgap14--pvStrip14Array--pvStrip14ArrayOne..1--pvStrip14Unit0xb6cb9b8.0  
    9           /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit.2                     __dd__Geometry__RPC__lvRPCGasgap14--pvStrip14Array--pvStrip14ArrayOne..2--pvStrip14Unit0xb6cc940.0  

    sqlite> select d.idx, w.name, d.name from wrl.geom w inner join dae.geom d on w.idx = d.idx + 1 limit 10000,10 ;
    idx         name                                                                                                  name                                                                                                
    ----------  ---------------------------------------------------------------------------------------------         ---------------------------------------------------------------------------------------------       
    10000       /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom.1001                                                 __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiBottom0xb5e55c8.588                                  
    10001       /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode.1002                                                 __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiDynode0xb2e6ff0.588                                  
    10002       /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFacein#pvNearOutFaceinWall9#pvNearOutFaceinWall9:4#p  __dd__Geometry__Pool__lvNearPoolOWS--pvVetoPmtNearOutFacein--pvNearOutFaceinWall9--pvNearOutFaceinWa
    10003       /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFacein#pvNearOutFaceinWall9#pvNearOutFaceinWall9:4#p  __dd__Geometry__Pool__lvNearPoolOWS--pvVetoPmtNearOutFacein--pvNearOutFaceinWall9--pvNearOutFaceinWa
    10004       /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFacein#pvNearOutFaceinWall9#pvNearOutFaceinWall9:4#p  __dd__Geometry__Pool__lvNearPoolOWS--pvVetoPmtNearOutFacein--pvNearOutFaceinWall9--pvNearOutFaceinWa
    10005       /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFacein#pvNearOutFaceinWall9#pvNearOutFaceinWall9:4#p  __dd__Geometry__Pool__lvNearPoolOWS--pvVetoPmtNearOutFacein--pvNearOutFaceinWall9--pvNearOutFaceinWa
    10006       /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFacein#pvNearOutFaceinWall9#pvNearOutFaceinWall9:4#p  __dd__Geometry__Pool__lvNearPoolOWS--pvVetoPmtNearOutFacein--pvNearOutFaceinWall9--pvNearOutFaceinWa
    10007       /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFacein#pvNearOutFaceinWall9#pvNearOutFaceinWall9:4#p  __dd__Geometry__Pool__lvNearPoolOWS--pvVetoPmtNearOutFacein--pvNearOutFaceinWall9--pvNearOutFaceinWa
    10008       /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFacein#pvNearOutFaceinWall9#pvNearOutFaceinWall9:4#p  __dd__Geometry__Pool__lvNearPoolOWS--pvVetoPmtNearOutFacein--pvNearOutFaceinWall9--pvNearOutFaceinWa
    10009       /dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFacein#pvNearOutFaceinWall9#pvNearOutFaceinWall9:4#p  __dd__Geometry__Pool__lvNearPoolOWS--pvVetoPmtNearOutFacein--pvNearOutFaceinWall9--pvNearOutFaceinWa
    sqlite> 

    sqlite> select count(*) from wrl.geom w inner join dae.geom d on w.idx = d.idx + 1 ;
    count(*)  
    ----------
    12230     

    sqlite> select count(*) from wrl.geom w inner join dae.geom d on w.idx = d.idx + 1 where w.nvertex != d.nvertex ;
    count(*)  
    ----------
    0         

    sqlite> select count(*) from wrl.geom w inner join dae.geom d on w.idx = d.idx + 1 where w.nvertex = d.nvertex ;
    count(*)  
    ----------
    12230     



Same process WRL then DAE : vertex counts match
-------------------------------------------------

::

    simon:wrl_gdml_dae blyth$ vrml2file.py -c -P g4_00.wrl
    2013-11-16 18:45:28,206 env.geant4.geometry.vrml2.vrml2file INFO     /Users/blyth/env/bin/vrml2file.py -c -P g4_00.wrl
    2013-11-16 18:45:28,208 env.geant4.geometry.vrml2.vrml2file INFO     create
    2013-11-16 18:46:27,520 env.geant4.geometry.vrml2.vrml2file INFO     gathering geometry, using idoffset True idlabel 1 
    2013-11-16 18:46:32,328 env.geant4.geometry.vrml2.vrml2file INFO     start persisting to /usr/local/env/geant4/geometry/gdml/wrl_gdml_dae/g4_00.wrl.db 


    simon:wrl_gdml_dae blyth$ sqlite3 -init cf.sql
    -- Loading resources from cf.sql
    seq  name             file                                                      
    ---  ---------------  ----------------------------------------------------------
    0    main                                                                       
    2    dae              /usr/local/env/geant4/geometry/gdml/wrl_gdml_dae/g4_00.dae
    3    wrl              /usr/local/env/geant4/geometry/gdml/wrl_gdml_dae/g4_00.wrl


    sqlite> select count(*) from wrl.geom w inner join dae.geom d on w.idx = d.idx + 1 ;
    12230     
    sqlite> select count(*) from wrl.geom w inner join dae.geom d on w.idx = d.idx + 1 where w.nvertex != d.nvertex ;
    0         
    sqlite> select count(*) from wrl.geom w inner join dae.geom d on w.idx = d.idx + 1 where w.nvertex = d.nvertex ;
    12230     


Built Meshlab in order to read DAE and WRL 
-------------------------------------------

But its real slow at reading DAE, 30 min import. 
Initially X3D/WRL/VRML plugin failed to load into meshlab. But 
a recompilation of x3d plugin succeeds.

The WRL import took under 8 min, thats almost 5 times faster than DAE import.::

    LOG: 0 Opened mesh /usr/local/env/geant4/geometry/gdml/wrl_gdml_dae/g4_00.wrl in 441612 msec
    LOG: 0 All files opened in 441615 msec

Navigation is painful at 0.3 fps though. 

   * BUT: **the PMT rotations look correct** 


Nov 18 2013 : Same Process export : WRL then DAE
--------------------------------------------------

Prep the DB ``g4dae-prep``::

    daedb.py --daepath g4_00.dae
    vrml2file.py --save --noshape g4_00.wrl 

Make point comparison ``g4dae-cf``::

    simon:wrl_gdml_dae blyth$ cat cf.sql 
    attach database "g4_00.dae.db" as dae ;
    attach database "g4_00.wrl.db" as wrl ;
    .databases
    .mode column
    .header on 
    --
    -- sqlite3 -init cf.sql
    --
    simon:wrl_gdml_dae blyth$ sqlite3 -init cf.sql
    -- Loading resources from cf.sql
    seq  name             file                                                      
    ---  ---------------  ----------------------------------------------------------
    0    main                                                                       
    2    dae              /usr/local/env/geant4/geometry/gdml/wrl_gdml_dae/g4_00.dae
    3    wrl              /usr/local/env/geant4/geometry/gdml/wrl_gdml_dae/g4_00.wrl

    SQLite version 3.8.0.2 2013-09-03 17:11:13


    sqlite> select count(*) from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id ; 
    count(*)  
    ----------
    1246046   

    sqlite> select count(*) from dae.point ;
    count(*)  
    ----------
    1246046   

    sqlite> select count(*) from wrl.point ;
    count(*)  
    ----------
    1246046   

    sqlite> select d.idx, max(abs(d.x - w.x)), max(abs(d.y - w.y)), max(abs(d.z - w.z))  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx ;

            -- maximum x,y,z absolute deviations for each solid , 
            --
            --      y deviations up to 0.5 mm      <<<< ROUNDED TO   1 MM 
            --      x,z more like 0.05 mm          <<<< ROUNDED TO 0.1 MM      
            --
            --  I THOUGHT I PATCHED THE VRML2 EXPORT TO AVOID THIS Y ROUNDING ?
            --

    ....
    12223       0.0394991636276245   0.499330341815948    0.0                
    12224       0.0418918146169744   0.46747952979058     0.0                
    12225       0.0464650988578796   0.250274777412415    0.0                
    12226       0.0406980668867618   0.454132347600535    0.0                
    12227       0.0394991636276245   0.499330341815948    0.0                
    12228       0.0418918146169744   0.46747952979058     0.0                
    12229       0.0516570425825194   0.415786688914523    0.0482940673828125 
    sqlite> 


WRL coordinate roundings again
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WRL x/y roundings: 0.1/1 mm::

                geometry IndexedFaceSet {
                        coord Coordinate {
                                point [
                                        -11149.5 -797803 668.904,
                                        -12907.2 -798915 668.904,
                                        -12768.2 -799135 668.904,
                                        -11010.5 -798023 668.904,
                                        -11149.5 -797803 670.904,
                                        -12907.2 -798915 670.904,
                                        -12768.2 -799135 670.904,
                                        -11010.5 -798023 670.904,
                                ]


OOPS : FIX IS ON CMS02 BUT NOT BELLE7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 src]$ grep SCB *.*
    G4VRML2FileSceneHandler.cc:#include <iomanip>   // SCB
    G4VRML2FileSceneHandler.cc:    G4cerr << "Using setprecision(5) and fixed floating point notation for veracity of output [SCB PATCH] " << G4endl; 
    G4VRML2FileSceneHandler.cc:    fDest << std::setprecision(5) << std::fixed ; // SCB
    [blyth@cms01 src]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/VRML/src


    [blyth@belle7 src]$ grep SCB *.*
    G4VRML2SceneHandlerFunc.icc:    std::cerr << "SCB " << pv_name << "\n";
    [blyth@belle7 src]$ pwd
    /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/source/visualization/VRML/src


DAE does not suffer from Y rounding as using local (not world) coordinates
of much smaller magnitude, which do not push precsion.


max squared offset per volume, before VRML mm rounding fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    sqlite> select d.idx, max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) as mds  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx having mds > 1 ;
    sqlite> select d.idx, max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) as mds  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx having mds > 0.8 ;
    sqlite> select d.idx, max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) as mds  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx having mds > 0.4 ;
               --
               -- NO volumes with maximum squared deviations more than 0.4 mm^2
               --

    sqlite> select d.idx, max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) as mds  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx having mds > 0.25  ;

                -- most deviate at about 0.25 mm^2 

    idx         mds              
    ----------  -----------------
    102         0.252105649424013
    110         0.252105649424024
    118         0.252051923645839
    119         0.252051923645839
    364         0.256258896525109
    372         0.25625889640749 
    376         0.255689442235299
    377         0.255689442235299
    402         0.25356702983926 
    403         0.25356702983926 
    435         0.250579669527435
    436         0.250579669527435
    438         0.250620243194824


    sqlite> select d.idx, max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) as mds  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx having mds > 0.255  ;

    sqlite> select d.idx, max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) as mds  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx having mds > 0.255  ;
    idx         mds              
    ----------  -----------------
    364         0.256258896525109
    372         0.25625889640749 
    376         0.255689442235299
    377         0.255689442235299
    912         0.256639970217134
    913         0.256639970217134
    1100        0.259075682699121
    1101        0.259075682699121
    1132        0.258564938347323
    1133        0.258564938347323
    2456        0.255183839891695
    ...
    5472        0.256080675965338
    5616        0.256080675965338
    5760        0.256080675965338
    5904        0.256080675965338
    8545        0.256874095728416
    8562        0.256781381772678
    9136        0.25735507284098 
    9170        0.256821185116763
    9204        0.256818434540021
    9238        0.256818568269607
    9980        0.255273145259131
    10424       0.256093619945864
    10968       0.255974403689378
    sqlite> 


Nov 18 2013 : Same Process export : DAE then WRL
--------------------------------------------------
Prep the DB::

    daedb.py --daepath g4_00.dae
    vrml2file.py --save --noshape g4_00.wrl 

Point comparison::

    sqlite> select d.idx, max(abs(d.x - w.x)), max(abs(d.y - w.y)), max(abs(d.z - w.z))  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx ;
    ...
    12217       0.0489782299046055   0.495534300804138    0.0                
    12218       0.0521936156255833   0.490957915782928    0.0                
    12219       0.0487635113167926   0.494483592337929    0.0                
    12220       0.0493128095640714   0.493383262306452    0.0                
    12221       0.0464650988578796   0.250274777412415    0.0                
    12222       0.0406980668885808   0.454132347600535    0.0                
    12223       0.0394991636276245   0.499330341815948    0.0                
    12224       0.0418918146169744   0.46747952979058     0.0                
    12225       0.0464650988578796   0.250274777412415    0.0                
    12226       0.0406980668867618   0.454132347600535    0.0                
    12227       0.0394991636276245   0.499330341815948    0.0                
    12228       0.0418918146169744   0.46747952979058     0.0                
    12229       0.0545820657571312   0.42653064802289     0.0490875244140625 
    sqlite> 
    sqlite> 


    sqlite> select d.idx, max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) as mds  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx having mds > 0.255  ;
    idx         mds              
    ----------  -----------------
    364         0.256258896525109
    372         0.25625889640749 
    376         0.255689442235299
    377         0.255689442235299
    912         0.256639970217134
    913         0.256639970217134
    1100        0.259075682699121
    1101        0.259075682699121
    1132        0.258564938347323
    1133        0.258564938347323
    ...
    5597        0.256311818725851
    5741        0.256311818725851
    5885        0.256311818725851
    8545        0.256874095728416
    8562        0.256781381772678
    9136        0.25735507284098 
    9170        0.256821185116763
    9204        0.256818434540021
    9238        0.256818568269607
    10424       0.256093619945864
    10968       0.255974403689378
    sqlite> 


Other order leads to the same level of agreement, ie just XY rounding issue.


Nov 19 : recompile N Geant4 with patched VRML2
--------------------------------------------------

Rebuild libVRML::

    g4-
    g4-vrml-deploy
    g4-vrml-make

Perform export again::

    [blyth@belle7 ~]$ export_all.sh

::

    [blyth@belle7 ~]$ cd /data1/env/local/env/geant4/geometry/gdml/20131119-1348/
    [blyth@belle7 20131119-1348]$ ls -l *.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 13:51 g4_00.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 13:51 g4_01.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 13:51 g4_02.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 13:52 g4_03.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 13:52 g4_04.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 13:52 g4_05.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 13:52 g4_06.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 13:52 g4_07.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 13:52 g4_08.wrl
    [blyth@belle7 20131119-1348]$ 
    [blyth@belle7 20131119-1348]$ 
    [blyth@belle7 20131119-1348]$ ls -l *.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 13:51 g4_00.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 13:51 g4_01.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 13:51 g4_02.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 13:52 g4_03.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 13:52 g4_04.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 13:52 g4_05.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 13:53 g4_06.dae
    [blyth@belle7 20131119-1348]$ 
    [blyth@belle7 20131119-1348]$ ls -l *.gdml
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 19 13:52 g4_00.gdml
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 19 13:52 g4_01.gdml
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 19 13:52 g4_02.gdml
    [blyth@belle7 20131119-1348]$ 


g4dae-prep::

    [blyth@belle7 20131119-1348]$ vrml2file.py --save --noshape g4_00.wrl 
    2013-11-19 13:58:07,683 env.geant4.geometry.vrml2.vrml2file INFO     /home/blyth/env/bin/vrml2file.py --save --noshape g4_00.wrl
    2013-11-19 13:58:07,683 env.geant4.geometry.vrml2.vrml2file INFO     parse
    ...
    [blyth@belle7 20131119-1348]$ python-
    [blyth@belle7 20131119-1348]$ python- source
    [blyth@belle7 20131119-1348]$ daedb.py --daepath g4_00.dae

g4dae-cf::

    [blyth@belle7 20131119-1348]$ g4dae-cf
    -- Loading resources from cf.sql
    seq  name             file                                                      
    ---  ---------------  ----------------------------------------------------------
    0    main                                                                       
    2    dae              /data1/env/local/env/geant4/geometry/gdml/20131119-1348/g4
    3    wrl              /data1/env/local/env/geant4/geometry/gdml/20131119-1348/g4

    SQLite version 3.8.0.2 2013-09-03 17:11:13
    Enter ".help" for instructions
    Enter SQL statements terminated with a ";"
    sqlite>  select d.idx, max(abs(d.x - w.x)), max(abs(d.y - w.y)), max(abs(d.z - w.z))  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx ;
    ...
    12221       0.00441282987594604  0.0112730264663696   0.0                
    12222       0.00356368305438082  0.0273382695158944   0.0                
    12223       0.00419008731842041  0.0190313458442688   0.0                
    12224       0.00595974788302556  0.0325204702094197   0.0                
    12225       0.00447291135787964  0.0112730264663696   0.0                
    12226       0.00262916892279463  0.0273382695158944   0.0                
    12227       0.00346958637237549  0.0190313458442688   0.0                
    12228       0.00599601340582012  0.0325204702094197   0.0                
    12229       0.00739222402626183  0.0321993082761765   0.00032806396484375
    sqlite> 




::

    sqlite> select d.idx, max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) as mds  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx having mds > 0.025 ; 
    sqlite> 
    sqlite> select d.idx, max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) as mds  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx having mds > 0.01 ; 
    sqlite> 
    sqlite> select d.idx, max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) as mds  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx having mds > 0.005 ; 
    sqlite> 
    sqlite> select d.idx, max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) as mds  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx having mds > 0.001 ; 

    --
    -- maximum squared offset between DAE and WRL vertices for each volume
    --
    --    DAE using pycollada monkey patch matrix diddling 
    --    WRL using g4-vrml-make to fix world coordinate Y 1mm rounding issue 
    --

    idx         mds                
    ----------  -------------------
    1           0.00328436747986416
    2           0.00222253675903199
    3           0.00106225017796422
    4           0.00106450298543162
    8           0.00118477510230729
    9           0.00118477510230729
    18          0.00118477510230729
    19          0.00118477510230729
    47          0.00100555585161196
    49          0.00107641548805947
    50          0.0010764154880389 
    59          0.00107641548805947
    60          0.0010764154880389 
    69          0.0010763202976155 
    77          0.0010763202976155 
    ...
    12205       0.00103526764915049
    12211       0.00103526764914272
    12213       0.00103526764915049
    12219       0.00103526764914272
    12224       0.00106595149374566
    12228       0.00106595149374566
    12229       0.00106225017796422
    sqlite> 


::

    In [45]: math.pow(0.00106595, 0.5)
    Out[45]: 0.0326488897207853            #  .03 mm is good enough


Nov 19 : Move Matrix fix from pycollada monkey patch to G4DAEWrite
---------------------------------------------------------------------

No surpises from /data1/env/local/dyb/external/build/LCG/clhep/2.0.4.2/CLHEP/Vector/Vector/Rotation.icc::

    278 inline HepRotation HepRotation::inverse() const {
    279   return HepRotation( rxx, ryx, rzx,
    280               rxy, ryy, rzy,
    281               rxz, ryz, rzz );
    282 }

::

    [blyth@belle7 DAE]$ pwd
    /home/blyth/e/geant4/geometry/DAE

    [blyth@belle7 DAE]$ cat make.sh 
    #!/bin/bash -l
    main(){
      local arg=$1
      dae-
      if [ "$arg" == "clean" ]; then 
         dae-make clean
         dae-make && dae-install
      else
         dae-make && dae-install
      fi
    }
    main $*

    [blyth@belle7 DAE]$ ./make.sh 
    Making dependency for file src/G4DAEWriteStructure.cc ...
    ...

re-compare::

    [blyth@belle7 ~]$ cd /data1/env/local/env/geant4/geometry/gdml/20131119-1632/
    [blyth@belle7 20131119-1632]$ ll *.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 16:35 g4_00.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 16:35 g4_01.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 16:35 g4_02.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 16:35 g4_03.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 16:35 g4_04.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 16:35 g4_05.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 16:35 g4_06.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 16:36 g4_07.wrl
    -rw-rw-r-- 1 blyth blyth 103914464 Nov 19 16:36 g4_08.wrl
    [blyth@belle7 20131119-1632]$ ll *.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 16:34 g4_00.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 16:35 g4_01.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 16:35 g4_02.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 16:35 g4_03.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 16:36 g4_04.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 16:36 g4_05.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 19 16:36 g4_06.dae
    [blyth@belle7 20131119-1632]$ ll *.gdml
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 19 16:35 g4_00.gdml
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 19 16:35 g4_01.gdml
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 19 16:35 g4_02.gdml
    [blyth@belle7 20131119-1632]$ 

    [blyth@belle7 20131119-1632]$ g4dae-
    [blyth@belle7 20131119-1632]$ g4dae-prep
    2013-11-19 16:38:57,992 env.geant4.geometry.vrml2.vrml2file INFO     /home/blyth/env/bin/vrml2file.py --save --noshape g4_00.wrl
    ...
    2013-11-19 16:40:26,073 env.geant4.geometry.vrml2.vrml2file INFO     skip extend
    Traceback (most recent call last):
      File "/home/blyth/env/bin/daedb.py", line 2, in <module>
        from env.geant4.geometry.collada.daedb import main
      File "/data1/env/system/python/Python-2.5.1/lib/python2.5/site-packages/env/geant4/geometry/collada/daedb.py", line 39, in <module>
        from daenode import DAENode, parse_args
      File "/data1/env/system/python/Python-2.5.1/lib/python2.5/site-packages/env/geant4/geometry/collada/daenode.py", line 259, in <module>
        from monkey_matrix_load import _monkey_matrix_load
      File "/data1/env/system/python/Python-2.5.1/lib/python2.5/site-packages/env/geant4/geometry/collada/monkey_matrix_load.py", line 21, in <module>
        assert 0, "NOV 18 2013 : NO LONGER REQUIRED NOW THAT THE INVROT IS DONE TO THE SOURCE DAE " 
    AssertionError: NOV 18 2013 : NO LONGER REQUIRED NOW THAT THE INVROT IS DONE TO THE SOURCE DAE 
    [blyth@belle7 20131119-1632]$ 



DAE WRL comparison with invrot done in DAE writing
---------------------------------------------------

After moving the matrix invrot to before DAE writing agreement is worse ? But still good enough. 

::

    sqlite> select d.idx, max(abs(d.x - w.x)), max(abs(d.y - w.y)), max(abs(d.z - w.z))  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx ;
    ...
    12223       0.00390625           0.0625               0.0                
    12224       0.0068359375         0.0                  0.0                
    12225       0.0048828125         0.0625               0.0                
    12226       0.0029296875         0.0                  0.0                
    12227       0.00390625           0.0625               0.0                
    12228       0.005859375          0.0                  0.0                
    12229       0.0078125            0.0625               0.0                   
    sqlite> 
             --- huh looks like Y is being rounded to 0.25 mm somewhere

::

    In [47]: math.pow(0.0625 , 0.5)
    Out[47]: 0.25

Many are RPC stripts that are +-1mm in Y 

::

    sqlite> select d.idx, max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) as mds  from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id group by d.idx having mds > 0.005 ; 
    idx         mds               
    ----------  ------------------
    116         0.0156595706939697
    120         0.015658512711525 
    284         0.0156528949737549     http://belle7.nuu.edu.tw/dae/tree/284.html
    288         0.0156470686197281
    447         0.0156452655792236
    448         0.0156452655792236
    451         0.0156443119049072
    452         0.0156452655792236
    455         0.0156432539224625
    456         0.0156442075967789
    494         0.0156452655792236
    498         0.0156442075967789
    657         0.0156443119049072
    658         0.0156481266021729
    661         0.0156452655792236
    662         0.0156481266021729
    665         0.0156442075967789
    666         0.0156470686197281
    704         0.0156481266021729
    708         0.0156518369913101
    825         0.0156481266021729
    826         0.0156775861978531
    829         0.0156595706939697
    830         0.0156595706939697
    834         0.0156775861978531
    867         0.0156528949737549
    868         0.0156681537628174
    871         0.0156528949737549
    872         0.0156595706939697
    875         0.0156518369913101
    876         0.015658512711525 
    914         0.0156528949737549
    918         0.0156470686197281
    1040        0.0156775861978531
    1044        0.0157042890787125    http://belle7.nuu.edu.tw/dae/tree/1044.html
    1124        0.0156910419464111
    1128        0.0156775861978531
    1250        0.0157042890787125
    1254        0.0157386213541031
    1502        0.0156481266021729
    1506        0.0156470686197281
    1712        0.0156452655792236
    1716        0.0156470686197281
    1875        0.0156518369913101
    1876        0.0156681537628174
    1879        0.0156595706939697
    1880        0.0156595706939697
    1883        0.015658512711525 
    1884        0.015658512711525 
    1922        0.0156595706939697
    1926        0.015658512711525 
    2043        0.0156775861978531
    2044        0.0157053470611572
    2047        0.0156786441802979
    2048        0.0156786441802979
    2051        0.0156775861978531
    2052        0.0157042890787125
    2090        0.0156786441802979
    2094        0.0156775861978531
    2132        0.0156786441802979
    2136        0.0156775861978531
    2253        0.0156775861978531
    2254        0.0157053470611572
    2257        0.0156786441802979
    2258        0.0157053470611572
    2261        0.0156775861978531
    2262        0.0157042890787125
    2300        0.0157053470611572
    2304        0.0157042890787125
    2360        0.0156488418579102
    2395        0.0156861692667007
    2396        0.0156861692667007
    2420        0.0157205015420914
    2422        0.0157205015420914
    sqlite> 

::

    In [48]: math.pow(0.0156595706939697,0.5)
    Out[48]: 0.12513820637187389


::

    sqlite> select idx, id, x, y, z from dae.point where idx=116 ;
    idx         id          x                  y             z                
    ----------  ----------  -----------------  ------------  -----------------
    116         0           -7321.64794921875  -806950.4375  -1338.71520996094
    116         1           -8443.369140625    -805198.8125  -1338.71520996094
    116         2           -8662.0576171875   -805338.875   -1351.45520019531
    116         3           -7540.33642578125  -807090.5     -1351.45520019531
    116         4           -7321.73046875     -806950.5     -1336.71765136719
    116         5           -8443.4521484375   -805198.875   -1336.71765136719
    116         6           -8662.140625       -805338.9375  -1349.45764160156
    116         7           -7540.4189453125   -807090.5625  -1349.45764160156
    sqlite> 
    sqlite> select idx, id, x, y, z from wrl.point where idx=116 ;
    idx         id          x                  y            z            
    ----------  ----------  -----------------  -----------  -------------
    116         0           -7321.64404296875  -806950.375  -1338.7109375
    116         1           -8443.3671875      -805198.75   -1338.7109375
    116         2           -8662.0546875      -805338.812  -1351.4509277
    116         3           -7540.33251953125  -807090.375  -1351.4509277
    116         4           -7321.7265625      -806950.437  -1336.7132568
    116         5           -8443.44921875     -805198.812  -1336.7132568
    116         6           -8662.1376953125   -805338.875  -1349.4532470
    116         7           -7540.4150390625   -807090.437  -1349.4532470
    sqlite> 


DAE Viz
---------

After 40 min import into meshlab, see that the PMT rotations look correct 
following the invrot fix.

Need to check SVN future of vcglib to see if its fixed. Otherwise need to 
profile. Suspect some trivial cacheing (map of id against parsed geometries, effects).
Especially geometries : as that repeats approx 250 times.
 

Face Check
------------

World::

    In [52]: import collada
    In [53]: dae = collada.Collada("/usr/local/env/geant4/geometry/gdml/20131119-1632/g4_00.dae")
    In [56]: top = dae.scene.nodes[0]
    In [59]: geom = list(top.objects('geometry'))
    In [60]: len(geom)
    Out[60]: 12230
    n [61]: geom[0]
    Out[61]: <BoundGeometry id=WorldBox0xcaa0198, 1 primitives>

    In [62]: w = geom[0]
    In [63]: w.primitives()
    Out[63]: <generator object primitives at 0x2519940>

    In [64]: list(w.primitives())
    Out[64]: [<BoundPolylist length=6>]

    In [65]: bpl = list(w.primitives())[0]

    In [92]: poly = list(bpl.polygons())

    In [93]: poly
    Out[93]: 
    [<Polygon vertices=4>,
     <Polygon vertices=4>,
     <Polygon vertices=4>,
     <Polygon vertices=4>,
     <Polygon vertices=4>,
     <Polygon vertices=4>]

    In [99]: poly[0].indices
    Out[99]: array([0, 3, 2, 1])

    In [100]: poly[1].indices
    Out[100]: array([4, 7, 3, 0])

    In [101]: poly[2].indices
    Out[101]: array([7, 6, 2, 3])

    In [102]: poly[3].indices
    Out[102]: array([6, 5, 1, 2])


::

    simon:20131119-1632 blyth$ head -100 g4_00.wrl 
    ...
    #---------- SOLID: Universe.0
    ...
                           coordIndex [
                                    0, 3, 2, 1, -1,
                                    4, 7, 3, 0, -1,
                                    7, 6, 2, 3, -1,
                                    6, 5, 1, 2, -1,
                                    5, 4, 0, 1, -1,
                                    4, 5, 6, 7, -1,
                            ]

near rock::

    In [108]: list(geom[1].primitives())[0]
    Out[108]: <BoundPolylist length=11>

    In [109]: bpl = list(geom[1].primitives())[0]

    In [110]: bpl.pol
    bpl.polygons   bpl.polyindex  

    In [110]: poly = list(bpl.polygons())

    In [111]: poly[0].indices
    Out[111]: array([0, 1, 2, 3])

    In [112]: poly[1].indices
    Out[112]: array([4, 5, 0])

    In [113]: poly[2].indices
    Out[113]: array([0, 3, 4])

::

    #---------- SOLID: /dd/Structure/Sites/db-rock.1000
                            }
                            coordIndex [
                                    0, 1, 2, 3, -1,
                                    4, 5, 0, -1,
                                    0, 3, 4, -1,
                                    6, 4, 3, -1,
                                    3, 2, 6, -1,
                                    7, 6, 2, -1,
                                    2, 1, 7, -1,
                                    5, 7, 1, -1,
                                    1, 0, 5, -1,
                                    5, 4, 6, -1,
                                    6, 7, 5, -1,
                            ]

::

    In [124]: map(lambda _:numpy.append(_.indices, -1), poly)         
    Out[124]: 
    [array([ 0,  1,  2,  3, -1]),
     array([ 4,  5,  0, -1]),
     array([ 0,  3,  4, -1]),
     array([ 6,  4,  3, -1]),
     array([ 3,  2,  6, -1]),
     array([ 7,  6,  2, -1]),
     array([ 2,  1,  7, -1]),
     array([ 5,  7,  1, -1]),
     array([ 1,  0,  5, -1]),
     array([ 5,  4,  6, -1]),
     array([ 6,  7,  5, -1])]

    In [125]: numpy.concatenate(map(lambda _:numpy.append(_.indices, -1), poly))  
    Out[125]: 
    array([ 0,  1,  2,  3, -1,  4,  5,  0, -1,  0,  3,  4, -1,  6,  4,  3, -1,
            3,  2,  6, -1,  7,  6,  2, -1,  2,  1,  7, -1,  5,  7,  1, -1,  1,
            0,  5, -1,  5,  4,  6, -1,  6,  7,  5, -1])


    In [138]: wrl
    Out[138]: 
    array([ 0,  1,  2,  3, -1,  4,  5,  0, -1,  0,  3,  4, -1,  6,  4,  3, -1,
            3,  2,  6, -1,  7,  6,  2, -1,  2,  1,  7, -1,  5,  7,  1, -1,  1,
            0,  5, -1,  5,  4,  6, -1,  6,  7,  5, -1,  0])

    In [139]: wrl =  numpy.fromstring(s[:-2], dtype=numpy.int, sep=',')

    In [140]: wrl
    Out[140]: 
    array([ 0,  1,  2,  3, -1,  4,  5,  0, -1,  0,  3,  4, -1,  6,  4,  3, -1,
            3,  2,  6, -1,  7,  6,  2, -1,  2,  1,  7, -1,  5,  7,  1, -1,  1,
            0,  5, -1,  5,  4,  6, -1,  6,  7,  5, -1])

    In [143]: numpy.array_equal(wrl, dae)
    Out[143]: True


All well and good, but want to stuff the WRL faces into DB table::

    In [144]: s = r"""
       .....:  0  3  2  1 -1  7  6  2  3 -1  6  5  1  2 -1  4  5  6  7 -1 11 13 10  8 -1
       .....:  12 11  8  9 -1 16 17  4 -1  3  0 15 -1 16  4  7 -1  3 15 14 -1 14 16  7 -1
       .....:   7  3 14 -1 18 15  0 -1  1  5  4 -1  4 17 19 -1 18  0  1 -1  1  4 19 -1 19
       .....:  18  1 -1 15 18  9 -1  8 10 14 -1 15  9  8 -1  8 14 15 -1 16 14 10 -1 10 13
       .....:  16 -1 18 19 12 -1 12  9 18 -1 17 16 13 -1 12 19 17 -1 17 13 11 -1 11 12 17
       .....:  -1 """

    In [147]: a = numpy.fromstring(s, dtype=numpy.int, sep=' ')

    In [148]: a
    Out[148]: 
    array([ 0,  3,  2,  1, -1,  7,  6,  2,  3, -1,  6,  5,  1,  2, -1,  4,  5,
            6,  7, -1, 11, 13, 10,  8, -1, 12, 11,  8,  9, -1, 16, 17,  4, -1,
            3,  0, 15, -1, 16,  4,  7, -1,  3, 15, 14, -1, 14, 16,  7, -1,  7,
            3, 14, -1, 18, 15,  0, -1,  1,  5,  4, -1,  4, 17, 19, -1, 18,  0,
            1, -1,  1,  4, 19, -1, 19, 18,  1, -1, 15, 18,  9, -1,  8, 10, 14,
           -1, 15,  9,  8, -1,  8, 14, 15, -1, 16, 14, 10, -1, 10, 13, 16, -1,
           18, 19, 12, -1, 12,  9, 18, -1, 17, 16, 13, -1, 12, 19, 17, -1, 17,
           13, 11, -1, 11, 12, 17, -1])


    In [172]: numpy.split(a,  numpy.where(a==-1)[0] + 1 )
    Out[172]: 
    [array([ 0,  3,  2,  1, -1]),
     array([ 7,  6,  2,  3, -1]),
     array([ 6,  5,  1,  2, -1]),
     array([ 4,  5,  6,  7, -1]),
     array([11, 13, 10,  8, -1]),
     array([12, 11,  8,  9, -1]),
     array([16, 17,  4, -1]),
     array([ 3,  0, 15, -1]),
     array([16,  4,  7, -1]),
     array([ 3, 15, 14, -1]),
     array([14, 16,  7, -1]),
     array([ 7,  3, 14, -1]),
     array([18, 15,  0, -1]),
     array([ 1,  5,  4, -1]),
     array([ 4, 17, 19, -1]),
     array([18,  0,  1, -1]),
     array([ 1,  4, 19, -1]),
     array([19, 18,  1, -1]),
     array([15, 18,  9, -1]),
     array([ 8, 10, 14, -1]),
     array([15,  9,  8, -1]),
     array([ 8, 14, 15, -1]),
     array([16, 14, 10, -1]),
     array([10, 13, 16, -1]),
     array([18, 19, 12, -1]),
     array([12,  9, 18, -1]),
     array([17, 16, 13, -1]),
     array([12, 19, 17, -1]),
     array([17, 13, 11, -1]),
     array([11, 12, 17, -1]),
     array([], dtype=float64)]

::

    sqlite> .schema face
    CREATE TABLE face (idx int,v0 int,v1 int,v2 int,v3 int,vx text,id int,nv int);
    sqlite> .w 10 10 10 10 10 10 10 100
    sqlite> select idx,id,nv,v0,v1,v2,v3,vx from face limit 20 ;
    idx         id          nv          v0          v1          v2          v3          vx                                                                                                  
    ----------  ----------  ----------  ----------  ----------  ----------  ----------  ---------------------------------------------------------------------------------------------       
    0           0           4           0           3           2           1           0,3,2,1                                                                                             
    0           1           4           4           7           3           0           4,7,3,0                                                                                             
    0           2           4           7           6           2           3           7,6,2,3                                                                                             
    0           3           4           6           5           1           2           6,5,1,2                                                                                             
    0           4           4           5           4           0           1           5,4,0,1                                                                                             
    0           5           4           4           5           6           7           4,5,6,7                                                                                             
    1           0           4           0           1           2           3           0,1,2,3                                                                                             
    1           1           3           4           5           0           -1          4,5,0,-1                                                                                            
    1           2           3           0           3           4           -1          0,3,4,-1                                                                                            
    1           3           3           6           4           3           -1          6,4,3,-1                                                                                            
    1           4           3           3           2           6           -1          3,2,6,-1                                                                                            
    1           5           3           7           6           2           -1          7,6,2,-1                                                                                            
    1           6           3           2           1           7           -1          2,1,7,-1                                                                                            
    1           7           3           5           7           1           -1          5,7,1,-1                                                                                            
    1           8           3           1           0           5           -1          1,0,5,-1                                                                                            
    1           9           3           5           4           6           -1          5,4,6,-1                                                                                            
    1           10          3           6           7           5           -1          6,7,5,-1                                                                                            
    2           0           4           0           3           2           1           0,3,2,1                                                                                             
    2           1           4           7           6           2           3           7,6,2,3                                                                                             
    2           2           4           6           5           1           2           6,5,1,2                                                                                             




Investigate DAE/WRL code path differences
--------------------------------------------------------


visualization/management/src/G4VSceneHandler.cc::

    859 G4int G4VSceneHandler::GetNoOfSides(const G4VisAttributes* pVisAttribs)
    860 {
    861   // No. of sides (lines segments per circle) is normally determined
    862   // by the view parameters, but it can be overriddden by the
    863   // ForceLineSegmentsPerCircle in the vis attributes.
    864   G4int lineSegmentsPerCircle = fpViewer->GetViewParameters().GetNoOfSides();
    865   if (pVisAttribs) {
    866     if (pVisAttribs->IsForceLineSegmentsPerCircle())
    867       lineSegmentsPerCircle = pVisAttribs->GetForcedLineSegmentsPerCircle();
    868     const G4int nSegmentsMin = 12;
    869     if (lineSegmentsPerCircle < nSegmentsMin) {
    870       lineSegmentsPerCircle = nSegmentsMin;
    871       G4cout <<
    872     "G4VSceneHandler::GetNoOfSides: attempt to set the"
    873     "\nnumber of line segements per circle < " << nSegmentsMin
    874          << "; forced to " << lineSegmentsPerCircle << G4endl;
    875     }
    876   }
    877   return lineSegmentsPerCircle;
    878 }


VRML2 specialisation of above with visualization/VRML/include/G4VRML2SceneHandler.hh::


* visualization/modeling/include/G4ModelingParameters.hh


Polysmry for debug
--------------------

The polysmry txt file allows rapid name/nvertex/nface checking by sorting and uniqing::

    [blyth@belle7 VDGX_20131121-1957]$ sort g4_00.wrl.txt | uniq > g4_00.wrl.txt.su
    [blyth@belle7 VDGX_20131121-1957]$ sort g4_00.dae.txt | uniq > g4_00.dae.txt.su
    [blyth@belle7 VDGX_20131121-1957]$ ll *.su
    -rw-rw-r-- 1 blyth blyth 850219 Nov 21 20:19 g4_00.wrl.txt.su
    -rw-rw-r-- 1 blyth blyth 850197 Nov 21 20:19 g4_00.dae.txt.su
    [blyth@belle7 VDGX_20131121-1957]$ diff g4_00.wrl.txt.su g4_00.dae.txt.su
    5643d5642
    < n Universe.0 v 8 f 6 


Always Create Polyhedron
---------------------------

Switching to CreatePolyhedron rather than GetPolyhedron allows the difference to 
be apparent in the same nuwa.py run::

    [blyth@belle7 VDGX_20131121-2043]$ cat g4_00.dae.txt | sort | uniq > g4_00.dae.txt.su
    [blyth@belle7 VDGX_20131121-2043]$ cat g4_00.wrl.txt  | sort | uniq > g4_00.wrl.txt.su
    [blyth@belle7 VDGX_20131121-2043]$ 
    [blyth@belle7 VDGX_20131121-2043]$ diff g4_00.dae.txt.su g4_00.wrl.txt.su
    27c27
    < n /dd/Geometry/AdDetails/lvOcrGdsTfbInLsoOfl#pvOcrGdsInLsoOfl.1000 v 49 f 94 
    ---
    > n /dd/Geometry/AdDetails/lvOcrGdsTfbInLsoOfl#pvOcrGdsInLsoOfl.1000 v 98 f 188 
    40c40
    < n /dd/Geometry/AD/lvADE#pvOflTnkContainer.1002 v 364 f 644 
    ---
    > n /dd/Geometry/AD/lvADE#pvOflTnkContainer.1002 v 366 f 648 
    47c47
    < n /dd/Geometry/AD/lvLSO#pvIAV.1000 v 170 f 264 
    ---
    > n /dd/Geometry/AD/lvLSO#pvIAV.1000 v 148 f 192 
    58c58
    < n /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:1#IavTopRibRot.1 v 27 f 53 
    ---
    > n /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:1#IavTopRibRot.1 v 16 f 28 
    60,64c60,64
    < n /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:3#IavTopRibRot.3 v 24 f 47 
    < n /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:4#IavTopRibRot.4 v 19 f 34 
    < n /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:5#IavTopRibRot.5 v 24 f 47 
    < n /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:6#IavTopRibRot.6 v 19 f 34 
    < n /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:7#IavTopRibRot.7 v 27 f 53 
    ---
    > n /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:3#IavTopRibRot.3 v 16 f 28 
    > n /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:4#IavTopRibRot.4 v 16 f 28 
    > n /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:5#IavTopRibRot.5 v 16 f 28 
    > n /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:6#IavTopRibRot.6 v 16 f 28 
    > n /dd/Geometry/AD/lvLSO#pvIavTopRibs#IavRibs:7#IavTopRibRot.7 v 16 f 28 
    77c77
    ...

Is the difference the same as that between runs ? Not so simple it seems::

    [blyth@belle7 gdml]$ grep NearSiteRock\#pvNearHallTop */*.txt 
    DVGX_20131121-2053/g4_00.dae.txt:n /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000 v 16 f 12 
    DVGX_20131121-2053/g4_00.wrl.txt:n /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000 v 20 f 30 
    VDGX_20131121-2043/g4_00.dae.txt:n /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000 v 16 f 12 
    VDGX_20131121-2043/g4_00.wrl.txt:n /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000 v 16 f 12 


[Dec 3, 2013] Back to Face checking after MeshLab Visualisation Interlude
---------------------------------------------------------------------------

Following fixing the omission of simtab clearing previously the horrendous 
former performance of chunked writing is explained. 
For DAE, straightforward write to DB once at the end, takes 5 min::

    [blyth@belle7 ~]$ daedb.py --daepath '$LOCAL_BASE/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae'       
    2013-12-03 15:34:54,665 env.geant4.geometry.collada.daenode INFO     /home/blyth/env/bin/daedb.py
    ...
    2013-12-03 15:37:03,314 env.geant4.geometry.collada.daedb INFO     perform final DB insert for inode 12229 insertsize 0 
    2013-12-03 15:37:03,315 env.geant4.geometry.collada.daedb INFO     writing tables to /data1/env/local/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae.db 
    2013-12-03 15:37:03,315 env.geant4.geometry.collada.daedb INFO     writing geom_t to /data1/env/local/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae.db 
    2013-12-03 15:37:03,824 env.geant4.geometry.collada.daedb INFO     writing point_t to /data1/env/local/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae.db 
    2013-12-03 15:38:20,287 env.geant4.geometry.collada.daedb INFO     writing face_t to /data1/env/local/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae.db 
    2013-12-03 15:40:05,296 env.geant4.geometry.collada.daedb INFO     completed writing to /data1/env/local/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae.db 

WRL::

    [blyth@belle7 daeserver]$ vrml2file.py --save  '$LOCAL_BASE/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.wrl'
    2013-12-03 16:00:44,574 env.geant4.geometry.vrml2.vrml2file INFO     /home/blyth/env/bin/vrml2file.py --save $LOCAL_BASE/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.wrl
    2013-12-03 16:00:44,575 env.geant4.geometry.vrml2.vrml2file INFO     parse
    2013-12-03 16:02:06,873 env.geant4.geometry.vrml2.vrml2file INFO     gathering geometry, using idoffset 0 idlabel True insertsize 0 
    2013-12-03 16:03:24,562 env.geant4.geometry.vrml2.vrml2file INFO     final insert
    2013-12-03 16:03:24,562 env.geant4.geometry.vrml2.vrml2file INFO     start persisting 
    2013-12-03 16:06:36,969 env.geant4.geometry.vrml2.vrml2file INFO     completed persisting
    2013-12-03 16:06:36,973 env.geant4.geometry.vrml2.vrml2file INFO     final insert done
    2013-12-03 16:06:36,981 env.geant4.geometry.vrml2.vrml2file INFO     skip extend
    [blyth@belle7 daeserver]$ 


Basic DAE queries::

    [blyth@belle7 ~]$ echo "select count(*) from geom ;" | sqlite3 $LOCAL_BASE/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae.db 
    12230

    [blyth@belle7 ~]$ sqlite3 /data1/env/local/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae.db 

    sqlite> select count(*) from point ; 
    1248274
    sqlite> select count(*) from geom ; 
    12230

    sqlite> select count(*) from face ; 
    1810829
    sqlite> select count(*) from face where nv=3 ; 
    1168654        
    sqlite> select count(*) from face where nv=4 ; 
    642175         
    sqlite> select 1168654+642175  ;
    1810829        
    sqlite> select sum(nv) from face ; 
    6074662        

    sqlite> .w 5 5 5 5 5 10 5 5 
    sqlite> select * from face limit 10 ; 
    idx    v0     v1     v2     v3     vx          id     nv   
    -----  -----  -----  -----  -----  ----------  -----  -----
    0      0      3      2      1      0,3,2,1     0      4    
    0      4      7      3      0      4,7,3,0     1      4    
    0      7      6      2      3      7,6,2,3     2      4    
    0      6      5      1      2      6,5,1,2     3      4    
    0      5      4      0      1      5,4,0,1     4      4    
    0      4      5      6      7      4,5,6,7     5      4    
    1      0      1      2      3      0,1,2,3     0      4    
    1      4      5      0      -1     4,5,0,-1    1      3    
    1      0      3      4      -1     0,3,4,-1    2      3    
    1      6      4      3      -1     6,4,3,-1    3      3    
    sqlite> 
    sqlite> select count(distinct(idx)) from face ; 
    12230


Basic WRL queries::

    sqlite> select count(*) from geom ; 
    12230
    sqlite> select count(*) from shape ; 
    12230
    sqlite> select count(*) from point ;  
    1246046
    sqlite> select count(*) from face ; 
    1806787


Disagree as was using always recreate poly approach.




With Poly reuse : quick polysmry is matching 
----------------------------------------------

Format the DAE metadata into a poly summary file::

    [blyth@belle7 DVGX_20131203-1719]$ pwd
    /data1/env/local/env/geant4/geometry/gdml/DVGX_20131203-1719

    [blyth@belle7 DVGX_20131203-1719]$ daemeta.py -p g4_00.dae -f "%(n_polysmry)s" > g4_00.dae.meta.txt
    [blyth@belle7 DVGX_20131203-1719]$ ll
    total 115916
    drwxrwxr-x 7 blyth blyth      4096 Dec  3 17:19 ..
    -rw-rw-r-- 1 blyth blyth    850213 Dec  3 17:22 g4_00.dae.txt
    -rw-rw-r-- 1 blyth blyth   7785762 Dec  3 17:22 g4_00.dae
    -rw-rw-r-- 1 blyth blyth   1383330 Dec  3 17:22 g4_00.wrl.txt
    -rw-rw-r-- 1 blyth blyth 103033949 Dec  3 17:22 g4_00.wrl
    -rw-rw-r-- 1 blyth blyth   4111332 Dec  3 17:22 g4_00.gdml
    drwxrwxr-x 2 blyth blyth      4096 Dec  3 17:35 .
    -rw-rw-r-- 1 blyth blyth   1383310 Dec  3 17:36 g4_00.dae.meta.txt


Compare the polysummary from the DAE metadata with the .txt summary during VRML2 export::

    [blyth@belle7 DVGX_20131203-1719]$ diff g4_00.wrl.txt g4_00.dae.meta.txt    # just the world volume is different 
    1c1
    < n Universe.0 v 8 f 6 
    ---
    > -

Following switching OFF always recreate poly get polysummary agreement. 
Polysummary just compares names and vertex and face counts::

    [blyth@belle7 DVGX_20131203-1719]$ head -10  g4_00.wrl.txt
    n Universe.0 v 8 f 6 
    n /dd/Structure/Sites/db-rock.1000 v 8 f 11 
    n /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000 v 16 f 12 
    n /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover.1000 v 32 f 60 
    n /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1.1 v 8 f 6 
    n /dd/Geometry/RPC/lvRPCMod#pvRPCFoam.1000 v 8 f 6 
    n /dd/Geometry/RPC/lvRPCFoam#pvBarCham14Array#pvBarCham14ArrayOne:1#pvBarCham14Unit.1 v 8 f 6 
    n /dd/Geometry/RPC/lvRPCBarCham14#pvRPCGasgap14.1000 v 8 f 6 
    n /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit.1 v 8 f 6 
    n /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit.2 v 8 f 6 
    [blyth@belle7 DVGX_20131203-1719]$ 

    [blyth@belle7 DVGX_20131203-1719]$ head -10 g4_00.dae.meta.txt
    -
    n /dd/Structure/Sites/db-rock.1000 v 8 f 11 
    n /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000 v 16 f 12 
    n /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover.1000 v 32 f 60 
    n /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1.1 v 8 f 6 
    n /dd/Geometry/RPC/lvRPCMod#pvRPCFoam.1000 v 8 f 6 
    n /dd/Geometry/RPC/lvRPCFoam#pvBarCham14Array#pvBarCham14ArrayOne:1#pvBarCham14Unit.1 v 8 f 6 
    n /dd/Geometry/RPC/lvRPCBarCham14#pvRPCGasgap14.1000 v 8 f 6 
    n /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit.1 v 8 f 6 
    n /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit.2 v 8 f 6 
    [blyth@belle7 DVGX_20131203-1719]$ 

::

    sqlite> select idx, count(*) as nface from wrl.face group by idx limit 10 ; 
    idx         nface     
    ----------  ----------
    0           6         
    1           11        
    2           12        
    3           60        
    4           6         
    5           6         
    6           6         
    7           6         
    8           6         
    9           6         

    sqlite> select idx, count(*) as nface from dae.face group by idx limit 10 ;    
    ## HUH THIS IS INCONSISTENT WITH DAE METADATA POLYSMRY, PyCollada post processing perhaps ? 
    idx         nface     
    ----------  ----------
    0           6         
    1           11        
    2           30        **
    3           64        **  
    4           6         
    5           6         
    6           6         
    7           6         
    8           6         
    9           6         


* Hmm is the VRML2, also recreating poly ? 


With poly reuse
-----------------


Turns out that there was extra poly creation for debugging here.

::

    [blyth@belle7 DVGX_20131203-1719]$ daedb.py --daepath g4_00.dae
    [blyth@belle7 DVGX_20131203-1719]$ vrml2file.py --save g4_00.wrl


::

    [blyth@belle7 DVGX_20131203-1719]$ g4dae-cf- 
    attach database "g4_00.dae.db" as dae ;
    attach database "g4_00.wrl.db" as wrl ;
    .databases
    .mode column
    .header on 

    -- sqlite3 -init cf.sql
    --
    [blyth@belle7 DVGX_20131203-1719]$ g4dae-cf- > cf.sql 
    [blyth@belle7 DVGX_20131203-1719]$ sqlite3 -init cf.sql
    Loading resources from cf.sql
    seq  name             file                                                      
    ---  ---------------  ----------------------------------------------------------
    0    main                                                                       
    2    dae              /data1/env/local/env/geant4/geometry/gdml/DVGX_20131203-17
    3    wrl              /data1/env/local/env/geant4/geometry/gdml/DVGX_20131203-17
    SQLite version 3.3.6
    Enter ".help" for instructions
    sqlite> 


::

    sqlite> select count(*) from dae.geom ; 
    12230     
    sqlite> select count(*) from wrl.geom ; 
    12230     

    sqlite> select count(*) from dae.point ; 
    1252843   
    sqlite> select count(*) from wrl.point ; 
    1252587   

    sqlite> select count(*) from dae.face ; 
    1816177   
    sqlite> select count(*) from wrl.face ; 
    1811191   




::

    sqlite> select count(*) from dae.geom d join wrl.geom w on w.idx = d.idx where d.nvertex != w.nvertex ; 
    1208      
    sqlite> select count(*) from dae.geom d join wrl.geom w on w.idx = d.idx where d.nvertex = w.nvertex ; 
    11022     
    sqlite> select 1208+11022 ; 
    12230     
    sqlite> select 1208./(1208+11022) ; 
    0.0987735077677841

    sqlite> select count(*) from dae.geom d join wrl.geom w on w.idx = d.idx where d.nface != w.nface ; 
    1210      





DGV_20131204-1746 : after removing all extra poly creation
-------------------------------------------------------------

Getting precisely the same geometry in DAE and VRML, as are using exactly the 
same instances of G4Polyhedron. Same points, faces. 
They get created on the first DAE export and then are reused for the VRML export.

Its kinda cheating but it works.

::

    [blyth@belle7 DGV_20131204-1746]$ export-
    [blyth@belle7 DGV_20131204-1746]$ export-cf
    /data1/env/local/env/geant4/geometry/export/DGV_20131204-1746
    export-cf is a function
    export-cf () 
    { 
        [ -n "$G4DAE_EXPORT_DIR" ] && cd_func $G4DAE_EXPORT_DIR;
        pwd;
        type $FUNCNAME;
        local base=g4_00;
        local dae=$base.dae;
        local wrl=$base.wrl;
        local sql=$base.sql;
        [ ! -f "$dae.db" ] && daedb.py --daepath $dae;
        [ ! -f "$wrl.db" ] && vrml2file.py --save $wrl;
        [ ! -f "$sql" ] && export-cf-sql- $base >$sql;
        cat $sql;
        alias | /usr/bin/which --tty-only --read-alias --show-dot --show-tilde sqlite3;
        sqlite3 -version;
        sqlite3 -init $sql
    }
    2013-12-04 18:08:21,025 env.geant4.geometry.collada.daenode INFO     /home/blyth/env/bin/daedb.py

    /data1/env/local/env/bin/sqlite3
    -- Loading resources from /home/blyth/.sqliterc

    3.8.0.2 2013-09-03 17:11:13 7dd4968f235d6e1ca9547cda9cf3bd570e1609ef
    -- Loading resources from g4_00.sql
    seq  name             file                                                      
    ---  ---------------  ----------------------------------------------------------
    0    main                                                                       
    2    dae              /data1/env/local/env/geant4/geometry/export/DGV_20131204-1
    3    wrl              /data1/env/local/env/geant4/geometry/export/DGV_20131204-1

    SQLite version 3.8.0.2 2013-09-03 17:11:13
    Enter ".help" for instructions
    Enter SQL statements terminated with a ";"
    sqlite> 
    sqlite> select count(*) from dae.geom ; 
    12230     
    sqlite> select count(*) from wrl.geom ;
    12230     
    sqlite> select count(*) from dae.geom d join wrl.geom w on w.idx = d.idx where d.nvertex != w.nvertex ;
    0         
    sqlite> select count(*) from dae.geom d join wrl.geom w on w.idx = d.idx where d.nface != w.nface ;
    0         

    sqlite> select count(*) from dae.point ;
    1264049   
    sqlite> select count(*) from wrl.point ;
    1264049   

    sqlite> select count(*) from wrl.face ;
    1824947   
    sqlite> select count(*) from dae.face ;
    1824947   

    sqlite> select count(*) from dae.face d join wrl.face w on w.idx = d.idx and w.id = d.id where w.vx = d.vx ; 
    1824947   


    sqlite> select count(*) from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id ;
    1264049   
    sqlite> select max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id ;

    sqlite> select max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id ;
    max((d.x-w
    ----------
    0.01573862


::

    In [4]: math.sqrt(0.01573862)
    Out[4]: 0.12545365678209622


Offsets up to about 0.125 mm. Most of imprecision comes from the numerically large y values.::

    sqlite> select d.idx as idx, d.id as id, d.x as dx, w.x as wx, d.x - w.x as xx, d.y as dy, w.y as wy, d.y - w.y as yy, d.z as dz, w.z as wz, d.z - w.z as zz, round((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z),5) as d2 from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id where d2 > 0.01 ;
    idx         id     dx               wx               xx               dy                    wy           yy          dz                 wz                 zz                d2        
    ----------  -----  ---------------  ---------------  ---------------  --------------------  -----------  ----------  -----------------  -----------------  ----------------  ----------
    116         3      -7540.336425781  -7540.332519531  -0.00390625      -807090.5             -807090.375  -0.125      -1351.45520019531  -1351.45092773438  -0.0042724609375  0.01566   
    116         7      -7540.418945312  -7540.415039062  -0.00390625      -807090.5625          -807090.437  -0.125      -1349.45764160156  -1349.45324707031  -0.00439453125    0.01566   
    120         3      -8415.08984375   -8415.0859375    -0.00390625      -807650.6875          -807650.562  -0.125      -1402.41516113281  -1402.41088867187  -0.0042724609375  0.01566   
    120         7      -8415.171875     -8415.16796875   -0.00390625      -807650.75            -807650.625  -0.125      -1400.41760253906  -1400.41333007812  -0.0042724609375  0.01566   
    284         7      -14277.37988281  -14277.37695312  -0.0029296875    -811404.875           -811404.75   -0.125      -1349.45764160156  -1349.45324707031  -0.00439453125    0.01565   
    288         7      -15152.1328125   -15152.13085937  -0.001953125     -811965.0625          -811964.937  -0.125      -1400.41760253906  -1400.41333007812  -0.0042724609375  0.01565   
    447         3      -13943.66210937  -13943.66113281  -0.0009765625    -806453.125           -806453.0    -0.125      -1404.90283203125  -1404.8984375      -0.00439453125    0.01565   
    448         5      -13313.0234375   -13313.02441406  0.0009765625     -803567.4375          -803567.312  -0.125      -1298.49768066406  -1298.49328613281  -0.00439453125    0.01565   
    448         6      -15062.53027343  -15062.53027343  0.0              -804687.8125          -804687.687  -0.125      -1400.41760253906  -1400.41333007812  -0.0042724609375  0.01564   
    451         7      -12847.3671875   -12847.3671875   0.0              -805739.1875          -805739.062  -0.125      -1336.71765136719  -1336.71325683594  -0.00439453125    0.01564   
    ...
    2257        7      -19758.29882812  -19758.3046875   0.005859375      -798290.125           -798290.0    -0.125      -1228.91760253906  -1228.91320800781  -0.00439453125    0.01568   
    2258        3      -19976.90429687  -19976.91015625  0.005859375      -798430.125           -798430.0    -0.125      -1243.65515136719  -1243.65087890625  -0.0042724609375  0.01568   
    2258        4      -19758.29882812  -19758.3046875   0.005859375      -798290.125           -798290.0    -0.125      -1228.91760253906  -1228.91320800781  -0.00439453125    0.01568   
    2258        6      -21098.70898437  -21098.71679687  0.0078125        -796678.5625          -796678.437  -0.125      -1241.65759277344  -1241.65319824219  -0.00439453125    0.01571   
    2258        7      -19976.98828125  -19976.99414062  0.005859375      -798430.1875          -798430.062  -0.125      -1241.65759277344  -1241.65319824219  -0.00439453125    0.01568   
    2261        7      -20633.05273437  -20633.05859375  0.005859375      -798850.3125          -798850.187  -0.125      -1279.87756347656  -1279.87329101563  -0.0042724609375  0.01568   
    2262        3      -20851.65625     -20851.6640625   0.0078125        -798990.3125          -798990.187  -0.125      -1294.61511230469  -1294.61083984375  -0.0042724609375  0.0157    
    2262        4      -20633.05078125  -20633.05859375  0.0078125        -798850.3125          -798850.187  -0.125      -1279.87756347656  -1279.87329101563  -0.0042724609375  0.0157    
    2262        6      -21973.4609375   -21973.46875     0.0078125        -797238.75            -797238.625  -0.125      -1292.61755371094  -1292.61328125     -0.0042724609375  0.0157    
    2262        7      -20851.74023437  -20851.74609375  0.005859375      -798990.375           -798990.25   -0.125      -1292.61755371094  -1292.61328125     -0.0042724609375  0.01568   
    2300        7      -21661.2265625   -21661.234375    0.0078125        -799508.75            -799508.625  -0.125      -1241.65759277344  -1241.65319824219  -0.00439453125    0.01571   
    2304        7      -22535.97851562  -22535.98632812  0.0078125        -800068.9375          -800068.812  -0.125      -1292.61755371094  -1292.61328125     -0.0042724609375  0.0157    
    2360        16     -16265.6796875   -16265.68457031  0.0048828125     -791837.6875          -791837.562  -0.125      -1730.63610839844  -1730.63610839844  0.0               0.01565   
    2360        20     -16265.6796875   -16265.68457031  0.0048828125     -791837.6875          -791837.562  -0.125      -1718.63610839844  -1718.63610839844  0.0               0.01565   
    2395        2      -20896.93164062  -20896.93945312  0.0078125        -794803.75            -794803.625  -0.125      -1436.3701171875   -1436.37048339844  0.0003662109375   0.01569   
    2395        5      -20896.93164062  -20896.93945312  0.0078125        -794803.75            -794803.625  -0.125      -1448.3701171875   -1448.37048339844  0.0003662109375   0.01569   
    2396        2      -20896.93164062  -20896.93945312  0.0078125        -794803.75            -794803.625  -0.125      -1718.3701171875   -1718.37048339844  0.0003662109375   0.01569   
    2396        5      -20896.93164062  -20896.93945312  0.0078125        -794803.75            -794803.625  -0.125      -1730.3701171875   -1730.37048339844  0.0003662109375   0.01569   
    2420        3      -23126.91015625  -23126.91992187  0.009765625      -796671.1875          -796671.062  -0.125      -1700.3701171875   -1700.37048339844  0.0003662109375   0.01572   
    2420        7      -23126.91015625  -23126.91992187  0.009765625      -796671.1875          -796671.062  -0.125      -1466.3701171875   -1466.37048339844  0.0003662109375   0.01572   
    2422        3      -25244.84179687  -25244.8515625   0.009765625      -798027.5             -798027.375  -0.125      -1700.3701171875   -1700.37048339844  0.0003662109375   0.01572   
    2422        7      -25244.84179687  -25244.8515625   0.009765625      -798027.5             -798027.375  -0.125      -1466.3701171875   -1466.37048339844  0.0003662109375   0.01572   
    sqlite> 
    sqlite> select count(*) from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id where (d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z) > 0.01 ;
    135       


    sqlite> select count(*) as N, round((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z), 5) as d2 from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id  group by d2 ; 
    N           d2        
    ----------  ----------
    597025      0.0       
    97493       1.0e-05   
    208077      2.0e-05   
    22731       3.0e-05   
    1189        4.0e-05   
    4274        5.0e-05   
    3895        6.0e-05   
    114         7.0e-05   
    1896        8.0e-05   
    31          9.0e-05   
    417         0.0001    
    578         0.00011   
    2           0.00012   
    20          0.00014   
    1           0.00015   
    17          0.00016   
    1           0.00019   
    1           0.00251   
    1           0.00254   
    1           0.00264   
    1           0.00267   
    240227      0.00391   
    57126       0.00392   
    16431       0.00393   
    6797        0.00394   
    1280        0.00395   
    1172        0.00396   
    1367        0.00397   
    275         0.00398   
    875         0.00399   
    265         0.004     
    307         0.00402   
    12          0.00404   
    10          0.00406   
    5           0.00407   
    18          0.01564   
    40          0.01565   
    25          0.01566   
    2           0.01567   
    24          0.01568   
    5           0.01569   
    12          0.0157    
    4           0.01571   
    4           0.01572   
    1           0.01574   
    sqlite> 




VGD_20131204-1806 : VRML first
----------------------------------


::

    [blyth@belle7 VGD_20131204-1806]$ export-cf
    /data1/env/local/env/geant4/geometry/export/VGD_20131204-1806
    export-cf is a function
    ...

    3.8.0.2 2013-09-03 17:11:13 7dd4968f235d6e1ca9547cda9cf3bd570e1609ef
    -- Loading resources from g4_00.sql
    seq  name             file                                                      
    ---  ---------------  ----------------------------------------------------------
    0    main                                                                       
    2    dae              /data1/env/local/env/geant4/geometry/export/VGD_20131204-1
    3    wrl              /data1/env/local/env/geant4/geometry/export/VGD_20131204-1

    SQLite version 3.8.0.2 2013-09-03 17:11:13
    Enter ".help" for instructions
    Enter SQL statements terminated with a ";"
    sqlite> 

    sqlite> select count(*) from dae.geom ;
    12230     
    sqlite> select count(*) from wrl.geom ;
    12230     
    sqlite> select count(*) from dae.geom d join wrl.geom w on w.idx = d.idx where d.nvertex != w.nvertex ;
    0         
    sqlite> select count(*) from dae.geom d join wrl.geom w on w.idx = d.idx where d.nface != w.nface ;
    0         

    sqlite> select count(*) from dae.point ;
    1246046   
    sqlite> select count(*) from wrl.point ;
    1246046   

    sqlite> select count(*) from wrl.face ;
    1806787   
    sqlite> select count(*) from dae.face ;
    1806787   

    sqlite> select count(*) from dae.face d join wrl.face w on w.idx = d.idx and w.id = d.id where w.vx = d.vx ; 
    1806787   
    sqlite> select count(*) from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id ;
    1246046   

    sqlite> select max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z)) from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id ;
    max((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z))
    --------------------------------------------------------------------
    0.0157386213541031                                                  


    sqlite> select d.idx as idx, d.id as id, d.x as dx, w.x as wx, d.x - w.x as xx, d.y as dy, w.y as wy, d.y - w.y as yy, d.z as dz, w.z as wz, d.z - w.z as zz, round((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z),5) as      d2 from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id where d2 > 0.01 ;
    idx         id          dx                 wx                 xx           dy          wy           yy          dz                 wz                 zz                d2        
    ----------  ----------  -----------------  -----------------  -----------  ----------  -----------  ----------  -----------------  -----------------  ----------------  ----------
    116         3           -7540.33642578125  -7540.33251953125  -0.00390625  -807090.5   -807090.375  -0.125      -1351.45520019531  -1351.45092773438  -0.0042724609375  0.01566   
    116         7           -7540.4189453125   -7540.4150390625   -0.00390625  -807090.56  -807090.437  -0.125      -1349.45764160156  -1349.45324707031  -0.00439453125    0.01566   
    120         3           -8415.08984375     -8415.0859375      -0.00390625  -807650.68  -807650.562  -0.125      -1402.41516113281  -1402.41088867187  -0.0042724609375  0.01566   
    120         7           -8415.171875       -8415.16796875     -0.00390625  -807650.75  -807650.625  -0.125      -1400.41760253906  -1400.41333007812  -0.0042724609375  0.01566   
    284         7           -14277.3798828125  -14277.376953125   -0.00292968  -811404.87  -811404.75   -0.125      -1349.45764160156  -1349.45324707031  -0.00439453125    0.01565   
    288         7           -15152.1328125     -15152.130859375   -0.00195312  -811965.06  -811964.937  -0.125      -1400.41760253906  -1400.41333007812  -0.0042724609375  0.01565   
    447         3           -13943.662109375   -13943.6611328125  -0.00097656  -806453.12  -806453.0    -0.125      -1404.90283203125  -1404.8984375      -0.00439453125    0.01565   
    448         5           -13313.0234375     -13313.0244140625  0.000976562  -803567.43  -803567.312  -0.125      -1298.49768066406  -1298.49328613281  -0.00439453125    0.01565   
    448         6           -15062.5302734375  -15062.5302734375  0.0          -804687.81  -804687.687  -0.125      -1400.41760253906  -1400.41333007812  -0.0042724609375  0.01564   
    451         7           -12847.3671875     -12847.3671875     0.0          -805739.18  -805739.062  -0.125      -1336.71765136719  -1336.71325683594  -0.00439453125    0.01564   
    452         3           -13065.9736328125  -13065.97265625    -0.00097656  -805879.18  -805879.062  -0.125      -1351.45520019531  -1351.45092773438  -0.0042724609375  0.01564   
    452         4           -12847.3671875     -12847.3671875     0.0          -805739.18  -805739.062  -0.125      -1336.71765136719  -1336.71325683594  -0.00439453125    0.01564   
    452         6           -14187.77734375    -14187.77734375    0.0          -804127.62  -804127.5    -0.125      -1349.45764160156  -1349.45324707031  -0.00439453125    0.01564   
    452         7           -13066.0556640625  -13066.0546875     -0.00097656  -805879.25  -805879.125  -0.125      -1349.45764160156  -1349.45324707031  -0.00439453125    0.01565   
    455         7           -13722.1201171875  -13722.1201171875  0.0          -806299.37  -806299.25   -0.125      -1387.67761230469  -1387.67333984375  -0.0042724609375  0.01564   
    456         3           -13940.7265625     -13940.7255859375  -0.00097656  -806439.37  -806439.25   -0.125      -1402.41516113281  -1402.41088867187  -0.0042724609375  0.01564   
    456         4           -13722.12109375    -13722.1201171875  -0.00097656  -806299.37  -806299.25   -0.125      -1387.67761230469  -1387.67333984375  -0.0042724609375  0.01564   
    456         6           -15062.5302734375  -15062.5302734375  0.0          -804687.81  -804687.687  -0.125      -1400.41760253906  -1400.41333007812  -0.0042724609375  0.01564   
    456         7           -13940.80859375    -13940.8076171875  -0.00097656  -806439.43  -806439.312  -0.125      -1400.41760253906  -1400.41333007812  -0.0042724609375  0.01564   
    494         7           -14750.296875      -14750.2958984375  -0.00097656  -806957.81  -806957.687  -0.125      -1349.45764160156  -1349.45324707031  -0.00439453125    0.01565   
    498         7           -15625.0498046875  -15625.048828125   -0.00097656  -807518.0   -807517.875  -0.125      -1400.41760253906  -1400.41333007812  -0.0042724609375  0.01564   
    657         3           -14416.5791015625  -14416.5791015625  0.0          -802006.06  -802005.937  -0.125      -1404.90283203125  -1404.8984375      -0.00439453125    0.01564   
    ...
    2262        3           -20851.65625       -20851.6640625     0.0078125    -798990.31  -798990.187  -0.125      -1294.61511230469  -1294.61083984375  -0.0042724609375  0.0157    
    2262        4           -20633.05078125    -20633.05859375    0.0078125    -798850.31  -798850.187  -0.125      -1279.87756347656  -1279.87329101563  -0.0042724609375  0.0157    
    2262        6           -21973.4609375     -21973.46875       0.0078125    -797238.75  -797238.625  -0.125      -1292.61755371094  -1292.61328125     -0.0042724609375  0.0157    
    2262        7           -20851.740234375   -20851.74609375    0.005859375  -798990.37  -798990.25   -0.125      -1292.61755371094  -1292.61328125     -0.0042724609375  0.01568   
    2300        7           -21661.2265625     -21661.234375      0.0078125    -799508.75  -799508.625  -0.125      -1241.65759277344  -1241.65319824219  -0.00439453125    0.01571   
    2304        7           -22535.978515625   -22535.986328125   0.0078125    -800068.93  -800068.812  -0.125      -1292.61755371094  -1292.61328125     -0.0042724609375  0.0157    
    2372        8           -16214.8525390625  -16214.859375      0.006835937  -793106.87  -793106.75   -0.125      -1722.33911132813  -1722.33947753906  0.0003662109375   0.01567   
    2372        11          -16221.58984375    -16221.595703125   0.005859375  -793111.18  -793111.062  -0.125      -1722.33911132813  -1722.33947753906  0.0003662109375   0.01566   
    2372        12          -16214.8525390625  -16214.859375      0.006835937  -793106.87  -793106.75   -0.125      -1650.33911132813  -1650.33947753906  0.0003662109375   0.01567   
    2372        15          -16221.58984375    -16221.595703125   0.005859375  -793111.18  -793111.062  -0.125      -1650.33911132813  -1650.33947753906  0.0003662109375   0.01566   
    2395        2           -20896.931640625   -20896.939453125   0.0078125    -794803.75  -794803.625  -0.125      -1436.3701171875   -1436.37048339844  0.0003662109375   0.01569   
    2395        5           -20896.931640625   -20896.939453125   0.0078125    -794803.75  -794803.625  -0.125      -1448.3701171875   -1448.37048339844  0.0003662109375   0.01569   
    2396        2           -20896.931640625   -20896.939453125   0.0078125    -794803.75  -794803.625  -0.125      -1718.3701171875   -1718.37048339844  0.0003662109375   0.01569   
    2396        5           -20896.931640625   -20896.939453125   0.0078125    -794803.75  -794803.625  -0.125      -1730.3701171875   -1730.37048339844  0.0003662109375   0.01569   
    2420        3           -23126.91015625    -23126.919921875   0.009765625  -796671.18  -796671.062  -0.125      -1700.3701171875   -1700.37048339844  0.0003662109375   0.01572   
    2420        7           -23126.91015625    -23126.919921875   0.009765625  -796671.18  -796671.062  -0.125      -1466.3701171875   -1466.37048339844  0.0003662109375   0.01572   
    2422        3           -25244.841796875   -25244.8515625     0.009765625  -798027.5   -798027.375  -0.125      -1700.3701171875   -1700.37048339844  0.0003662109375   0.01572   
    2422        7           -25244.841796875   -25244.8515625     0.009765625  -798027.5   -798027.375  -0.125      -1466.3701171875   -1466.37048339844  0.0003662109375   0.01572   
    sqlite> 


    sqlite> select count(*) as N, round((d.x-w.x)*(d.x-w.x) + (d.y-w.y)*(d.y-w.y) + (d.z-w.z)*(d.z-w.z), 5) as d2 from dae.point d join wrl.point w on d.idx = w.idx and d.id = w.id  group by d2 ; 
    N           d2        
    ----------  ----------
    587943      0.0       
    98421       1.0e-05   
    204538      2.0e-05   
    22716       3.0e-05   
    1125        4.0e-05   
    4301        5.0e-05   
    3868        6.0e-05   
    109         7.0e-05   
    1885        8.0e-05   
    31          9.0e-05   
    413         0.0001    
    578         0.00011   
    22          0.00014   
    1           0.00015   
    17          0.00016   
    5           0.00019   
    2           0.00035   
    2           0.00046   
    2           0.0007    
    2           0.0008    
    1           0.00251   
    1           0.00254   
    1           0.00264   
    1           0.00267   
    235705      0.00391   
    57199       0.00392   
    15199       0.00393   
    6351        0.00394   
    1127        0.00395   
    1163        0.00396   
    1418        0.00397   
    265         0.00398   
    876         0.00399   
    271         0.004     
    311         0.00402   
    20          0.00404   
    10          0.00406   
    5           0.00407   
    4           0.00409   
    18          0.01564   
    38          0.01565   
    27          0.01566   
    4           0.01567   
    24          0.01568   
    5           0.01569   
    12          0.0157    
    4           0.01571   
    4           0.01572   
    1           0.01574   



Crossed Order Comparison
--------------------------

Compare D from VGD and DGV::

    [blyth@belle7 export]$ export-cf-find- > cf.sql
    [blyth@belle7 export]$ sqlite3 -init cf.sql
    -- Loading resources from cf.sql
    seq  name             file                                                      
    ---  ---------------  ----------------------------------------------------------
    0    main                                                                       
    2    db1              /data1/env/local/env/geant4/geometry/export/./VGD_20131204
    3    db2              /data1/env/local/env/geant4/geometry/export/./DGV_20131204

    SQLite version 3.8.0.2 2013-09-03 17:11:13
    Enter ".help" for instructions
    Enter SQL statements terminated with a ";"
    sqlite> 
    [blyth@belle7 export]$ sqlite3 -version
    -- Loading resources from /home/blyth/.sqliterc

    3.8.0.2 2013-09-03 17:11:13 7dd4968f235d6e1ca9547cda9cf3bd570e1609ef
    [blyth@belle7 export]$ 
    [blyth@belle7 export]$ sqlite3 -init cf.sql
    SQLite version 3.8.0.2 2013-09-03 17:11:13
    Enter ".help" for instructions
    Enter SQL statements terminated with a ";"
    sqlite> select count(*) from db1.geom ; 
    12230     
    sqlite> select count(*) from db2.geom ; 
    12230     
    sqlite> select count(*) from db1.point ; 
    1246046   
    sqlite> select count(*) from db2.point ; 
    1264049   

    sqlite> select count(*) from db1.geom l inner join db2.geom r on l.idx = r.idx ;  
    12230     
    sqlite> select count(*) from db1.geom l inner join db2.geom r on l.idx = r.idx where l.geoid = r.geoid ;  
    12230     


::

    sqlite> select l.geoid, count(*) as N from db1.geom l inner join db2.geom r on l.idx = r.idx where l.nvertex != r.nvertex or l.nface != r.nface group by l.geoid ; 
    geoid                    N         
    -----------------------  ----------
    AmCCo60AcrylicContainer  6         
    AmCCo60Cavity            6         
    IavTopRib                16        
    LsoOflTnk                2         
    OavTopRib                16        
    OcrCalLso                2         
    OcrCalLsoPrt             2         
    OcrGdsInLsoOfl           2         
    OcrGdsLsoInOil           2         
    OcrGdsLsoPrt             2         
    OcrGdsPrt                2         
    OcrGdsTfbInLso           2         
    OcrGdsTfbInLsoOfl        2         
    OflTnkContainer          2         
    SstBotRib                16        
    SstTopCirRibBase         16        
    SstTopHub                2         
    SstTopRadiusRib          16        
    amcco60-source-assy      6         
    headon-pmt-assy          12        
    headon-pmt-mount         12        
    led-source-assy          6         
    led-source-shell         6         
    lso                      2         
    near-radslab-box-9       1         
    near_hall_top_dwarf      1         
    near_pentagon_iron_box   144       
    near_pool_dead_box       1         
    near_pool_liner_box      1         
    near_pool_ows_box        1         
    near_top_cover_box       1         
    pmt-hemi                 672       
    pmt-hemi-vac             672       
    source-assy              6         
    source-shell             6         
    wall-led-assy            6         
    weight-shell             36        
    sqlite> 


Have a look at the matched tables::

    sqlite> drop table matched ;
    sqlite> create table matched as select l.idx,l.nvertex,l.nface,l.geoid, l.name from db1.geom l inner join db2.geom r on l.idx = r.idx where l.nvertex = r.nvertex and l.nface = r.nface ;
    sqlite> select count(*) from matched ; 
    10524     

    sqlite> select count(*) from matched m, db1.point l, db2.point r on m.idx = l.idx and m.idx = r.idx and l.id = r.id ;      -- points from the matched volumes 
    750531    

Simple way of establishing volume match by vertex and face count is incorrect for many volumes
with large point offsets. The coordinate offsets are meaningless when are not talking abot the same points.::

    sqlite> select max((l.x-r.x)*(l.x-r.x) + (l.y-r.y)*(l.y-r.y) + (l.z-r.z)*(l.z-r.z)) from matched m, db1.point l, db2.point r on m.idx = l.idx and m.idx = r.idx and l.id = r.id ;  
    max((l.x-r.x)*(l.x-r.x) + (l.y-r.y)*(l.y-r.y) + (l.z-r.z)*(l.z-r.z))
    --------------------------------------------------------------------
    1968663.87335779                                                    
    sqlite> 

    sqlite> select m.geoid, round(max((l.x-r.x)*(l.x-r.x) + (l.y-r.y)*(l.y-r.y) + (l.z-r.z)*(l.z-r.z)),6) as mx, round(min((l.x-r.x)*(l.x-r.x) + (l.y-r.y)*(l.y-r.y) + (l.z-r.z)*(l.z-r.z)),6) as mi, count(*) as N from matched m, db1.point l, db2.point r on m.idx = l.idx and m.idx = r.idx and l.id = r.id group by m.geoid  ;  
    geoid                           mx               mi               N              
    ------------------------------  ---------------  ---------------  ---------------
    AcrylicCylinder                 0.0              0.0              576            
    AdPmtCollar                     0.0              0.0              36864          
    AmCCo60SourceAcrylic            0.0              0.0              300            
    GdLSCalibTubAbvLidTub3          0.0              0.0              192            
    ...
    GdLSCalibTubAbvLidTub4          0.0              0.0              192            
    GdLSCalibTubAbvLidTub5          0.0              0.0              192            
    GdLSInCalibTubAbvLid            0.0              0.0              100            
    GdsOfl                          0.003907         0.0              200            
    GdsOflTnk                       1306770.525574   0.0              1552           
    Ge68AirBot                      0.0              0.0              300            
    ...
    SupportSpoolInteriorGdLS        0.0              0.0              100            
    SupportSpoolInteriorLS          0.0              0.0              100            
    SupportSpoolLS                  0.0              0.0              100            
    TopESRCutHols                   0.004002         0.0              1156           
    TopRefCutHols                   4223.694238      0.0              592            
    TopRefGapCutHols                9023.086114      0.0              592            
    WallLedDiffuserBall             0.0              0.0              1602           
    Weight                          0.0              0.0              1800           
    ...
    near-radslab-box-6              0.0              0.0              8              
    near-radslab-box-7              0.0              0.0              8              
    near-radslab-box-8              0.0              0.0              8              
    near_diagonal_angle_iron        1968663.873358   0.0              1440           
    near_diagonal_square_iron       0.0              0.0              576            
    ...
    pmt-rib-1                       0.0              0.0              6912           
    pmt-rib-2                       0.0              0.0              6912           
    pmt-rib-3                       0.0              0.0              6912           
    pmt-tee                         111473.942777    0.0              5760           
    pmt-top-ring                    0.0              0.0              27648          
    sst                             0.0              0.0              100            
    support_rib1_box                0.0              0.0              64             
    support_rib5_box                0.0              0.0              32             
    table_panel_box                 0.047119         0.0              116            
    top_corner_cable                0.0              0.0              32             
    top_corner_cable_tray           0.036831         0.0              64             
    top_short_cable                 0.0              0.0              16             
    top_short_cable_tray            0.012612         0.0              32             
    turntable                       172.265908       0.0              2436           
    verti_cable                     0.0              0.0              64             
    verti_cable_tray                0.171421         0.0              128            
    wall-led-rod                    0.0              0.0              300            
    sqlite> 

Lots of huge offsets indicate invalid comparisons, ie not the same point.

Need a better way to establish a volume match, which is independent of the point coordinates.
Use face matching.::

    sqlite> .w 10 100
    sqlite> select idx, group_concat(vx,":") from db1.face group by idx limit 10 ; 
    idx         group_concat(vx,":")                                                                                
    ----------  ---------------------------------------------------------------------------------------------       
    0           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    1           0,1,2,3:4,5,0,-1:0,3,4,-1:6,4,3,-1:3,2,6,-1:7,6,2,-1:2,1,7,-1:5,7,1,-1:1,0,5,-1:5,4,6,-1:6,7,5,-1   
    2           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7:8,11,10,9:12,15,11,8:15,14,10,11:14,13,9,10:13,12,8,
    3           5,1,0,-1:0,3,5,-1:4,2,1,-1:1,5,4,-1:8,9,1,-1:1,2,8,-1:11,12,2,-1:2,4,11,-1:12,8,2,-1:14,11,4,-1:4,5,
    4           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    5           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    6           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    7           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    8           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    9           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    sqlite> 
    sqlite> create table face1 as select idx, group_concat(vx,":") as vxa from db1.face group by idx ;
    sqlite> create table face2 as select idx, group_concat(vx,":") as vxa from db2.face group by idx ;
    sqlite> 


::

    sqlite> select count(*) from face1 ;  
    12230     
    sqlite> select count(*) from face2 ;
    12230     

    sqlite> select * from face1 limit 10 ; 
    idx         vxa                                                                                                 
    ----------  ---------------------------------------------------------------------------------------------       
    0           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    1           0,1,2,3:4,5,0,-1:0,3,4,-1:6,4,3,-1:3,2,6,-1:7,6,2,-1:2,1,7,-1:5,7,1,-1:1,0,5,-1:5,4,6,-1:6,7,5,-1   
    2           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7:8,11,10,9:12,15,11,8:15,14,10,11:14,13,9,10:13,12,8,
    3           5,1,0,-1:0,3,5,-1:4,2,1,-1:1,5,4,-1:8,9,1,-1:1,2,8,-1:11,12,2,-1:2,4,11,-1:12,8,2,-1:14,11,4,-1:4,5,
    4           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    5           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    6           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    7           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    8           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    9           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    sqlite> select * from face2 limit 10 ; 
    idx         vxa                                                                                                 
    ----------  ---------------------------------------------------------------------------------------------       
    0           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    1           0,1,2,3:4,5,0,-1:0,3,4,-1:6,4,3,-1:3,2,6,-1:7,6,2,-1:2,1,7,-1:5,7,1,-1:1,0,5,-1:5,4,6,-1:6,7,5,-1   
    2           0,3,2,1:7,6,2,3:6,5,1,2:4,5,6,7:11,13,10,8:12,11,8,9:16,17,4,-1:3,0,15,-1:16,4,7,-1:3,15,14,-1:14,16
    3           3,1,0,-1:0,2,3,-1:6,7,0,-1:0,1,8,-1:8,6,0,-1:10,11,1,-1:1,3,10,-1:11,8,1,-1:13,10,3,-1:12,13,3,-1:12
    4           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    5           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    6           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    7           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    8           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    9           0,3,2,1:4,7,3,0:7,6,2,3:6,5,1,2:5,4,0,1:4,5,6,7                                                     
    sqlite> 

    sqlite> select count(*) from face1 l, face2 r on l.idx = r.idx and l.vxa = r.vxa ; 
    10146     

    sqlite> create table fmatched as select l.idx from face1 l, face2 r on l.idx = r.idx and l.vxa = r.vxa ; 
    sqlite> select count(*) from fmatched ; 
    10146     

    sqlite> select count(*) from matched ; 
    10524     

    sqlite> select round(max((l.x-r.x)*(l.x-r.x) + (l.y-r.y)*(l.y-r.y) + (l.z-r.z)*(l.z-r.z)),6) from fmatched m, db1.point l, db2.point r on m.idx = l.idx and m.idx = r.idx and l.id = r.id ;  
    round(max(
    ----------
    0.336924  


    sqlite> select round((l.x-r.x)*(l.x-r.x) + (l.y-r.y)*(l.y-r.y) + (l.z-r.z)*(l.z-r.z),3) as d2, count(*) as N from fmatched m, db1.point l, db2.point r on m.idx = l.idx and m.idx = r.idx and l.id = r.id group by d2 order by d2 ;  
    d2          N                                                                                                   
    ----------  ---------------------------------------------------------------------------------------------       
    0.0         702500                                                                                              
    0.001       11183                                                                                               
    0.002       1242                                                                                                
    0.003       4                                                                                                   
    0.004       17982                                                                                               
    0.005       2742                                                                                                
    0.006       96                                                                                                  
    0.007       9                                                                                                   
    0.008       139                                                                                                 
    0.009       83                                                                                                  
    0.01        27                                                                                                  
    0.011       2                                                                                                   
    0.013       9                                                                                                   
    0.014       10                                                                                                  
    0.015       7                                                                                                   
    0.016       56                                                                                                  
    0.017       161                                                                                                 
    0.018       7                                                                                                   
    0.02        256                                                                                                 
    0.021       92                                                                                                  
    0.023       44                                                                                                  
    0.024       84                                                                                                  
    0.025       64                                                                                                  
    0.026       64                                                                                                  
    0.027       19                                                                                                  
    0.028       13                                                                                                  
    0.029       64                                                                                                  
    0.032       4                                                                                                   
    0.033       4                                                                                                   
    0.036       4                                                                                                   
    0.037       21                                                                                                  
    0.038       179                                                                                                 
    0.039       156                                                                                                 
    0.04        20                                                                                                  
    0.041       20                                                                                                  
    0.042       2                                                                                                   
    0.047       12                                                                                                  
    0.063       64                                                                                                  
    0.065       62                                                                                                  
    0.071       8                                                                                                   
    0.074       64                                                                                                  
    0.16        4                                                                                                   
    0.17        16                                                                                                  
    0.171       16                                                                                                  
    0.174       64                                                                                                  
    0.176       60                                                                                                  
    0.177       4                                                                                                   
    0.283       64                                                                                                  
    0.285       56                                                                                                  
    0.286       88                                                                                                  
    0.3         24                                                                                                  
    0.301       8                                                                                                   
    0.312       32                                                                                                  
    0.335       10                                                                                                  
    0.336       30                                                                                                  
    0.337       104            


::

    In [6]: math.sqrt(0.337)
    Out[6]: 0.58051701094799968


Conclusion is that when doing cross comparison, the majority 83% of 
volumes have points at most a fraction of a mm off::

    In [7]: 10146./12230
    Out[7]: 0.82959934587080952

For the toxic volumes, there is no face/point correspondence so cannot compare this way.
Could look at bbox howewever.

::

    qlite> create table ftoxic as select l.idx from face1 l, face2 r on l.idx = r.idx and l.vxa != r.vxa ; 
    sqlite> select count(*) from ftoxic ; 
    2084      

    sqlite> select 2084 + 10146  ;
    12230     


    sqlite> .w 10 10 10 10 10 
    sqlite> select t.idx, round(min(l.x),10),round(min(r.x),10), round(max(l.x),10), round(max(r.x),10) from ftoxic t, db1.point l, db2.point r on t.idx = l.idx and t.idx = r.idx and l.id = r.id group by t.idx ; 
    idx         round(min(  round(min(  round(max(  round(max(
    ----------  ----------  ----------  ----------  ----------
    2           -31088.125  -31088.125  5406.43164  5405.88867
    3           -23431.509  -23431.705  -9608.4257  -9608.4902
    2381        -17382.134  -17382.134  -16887.052  -16887.052
    2382        -17382.134  -17382.134  -16887.052  -16887.052
    2383        -21719.052  -21719.052  -21223.970  -21223.970
    2384        -21719.052  -21719.052  -21223.970  -21223.970
    2385        -20968.531  -20968.570  -19566.027  -19566.027
    2386        -20968.560  -20968.531  -19566.027  -19566.027



    sqlite> select t.idx, round(min(l.x)-min(r.x),5) as dmix, round(max(l.x)-max(r.x),5) as dmax, round(min(l.y)-min(r.y),5) as dmiy, round(max(l.y)-max(r.y),5) as dmay, round(min(l.z)-min(r.z),5) as dmiz, round(max(l.z)-max(r.z),5) as dmaz  from ftoxic t, db1.point l, db2.point r on t.idx = l.idx and t.idx = r.idx and l.id = r.id group by t.idx ; 
    idx         dmix        dmax        dmiy        dmay        dmiz        dmaz      
    ----------  ----------  ----------  ----------  ----------  ----------  ----------
    2           0.0         0.54297     -0.875      0.0         -0.29004    0.0       
    3           0.19531     0.06445     0.1875      -0.0625     0.0         0.0       
    2381        0.0         0.0         0.0625      0.0         0.01953     0.0       
    2382        0.0         0.0         -0.0625     0.0         0.0         -0.01953  
    2383        0.0         0.0         0.0625      0.0         0.01953     0.0       
    2384        0.0         0.0         -0.0625     0.0         0.0         -0.01953  
    2385        0.03906     0.0         0.0         0.0         0.01953     0.0       
    2386        -0.0293     0.0         0.0         0.0         0.0         -0.01953  
    2387        0.03906     0.0         0.0         0.0         0.01953     0.0       
    2388        -0.0293     0.0         0.0         0.0         0.0         -0.01953  
    2389        0.0         0.0         0.0         0.0         0.0         0.0       
    2390        0.0         0.0         0.0         0.0         0.0         0.0       
    2391        0.0         0.0         0.0         0.0         0.0         0.0       
    2392        0.0         0.0         0.0         0.0         0.0         0.0       
    2393        0.0         0.0         0.0         0.0         0.0         0.0       
    2394        0.0         0.0         0.0         0.0         0.0         0.0       


    sqlite> create table toxbox as select t.idx, round(min(l.x)-min(r.x),5) as dmix, round(max(l.x)-max(r.x),5) as dmax, round(min(l.y)-min(r.y),5) as dmiy, round(max(l.y)-max(r.y),5) as dmay, round(min(l.z)-min(r.z),5) as dmiz, round(max(l.z)-max(r.z),5) as dmaz  from ftoxic t, db1.point l, db2.point r on t.idx = l.idx and t.idx = r.idx and l.id = r.id group by t.idx ; 

    sqlite> select count(*) from toxbox ; 
    2084      


Limited number with xy problems::


    sqlite> select * from toxbox where abs(dmix) > 0.1 or abs(dmax) > 0.1 or abs(dmiy) > 0.1 or abs(dmiy) > 0.1  ; 
    t.idx       dmix        dmax        dmiy        dmay        dmiz        dmaz      
    ----------  ----------  ----------  ----------  ----------  ----------  ----------
    2           0.0         0.54297     -0.875      0.0         -0.29004    0.0       
    3           0.19531     0.06445     0.1875      -0.0625     0.0         0.0       
    3148        -0.21875    4241.53809  -0.125      0.375       0.0         0.0       
    3149        0.21289     -4168.5293  0.125       -0.375      0.0         0.0       
    3150        -0.21289    -0.32031    -0.125      0.375       0.0         0.0       
    4465        3.375       -7.69531    9.4375      -6.5        0.0         0.0       
    4466        0.0         -4.31445    9.125       -10.0625    0.0         0.0       
    4467        6.55078     -9.47852    3.375       -7.6875     0.0         0.0       
    4468        10.07227    -9.08789    0.0         -4.3125     0.0         0.0       
    4469        7.69531     -3.375      6.5         -9.4375     0.0         0.0       
    4470        4.31445     0.0         10.0625     -9.125      0.0         0.0       
    4471        9.47852     -6.55078    7.6875      -3.375      0.0         0.0       
    4472        9.08789     -10.07227   4.3125      0.0         0.0         0.0       
    6125        3.37402     -7.69629    9.4375      -6.5        0.0         0.0       
    6126        0.0         -4.31543    9.125       -10.0625    0.0         0.0       
    6127        6.55078     -9.47852    3.375       -7.6875     0.0         0.0       
    6128        10.07422    -9.08887    0.0         -4.3125     0.0         0.0       
    6129        7.69629     -3.37402    6.5         -9.4375     0.0         0.0       
    6130        4.31543     0.0         10.0625     -9.125      0.0         0.0       
    6131        9.47852     -6.55078    7.6875      -3.375      0.0         0.0       
    6132        9.08887     -10.07422   4.3125      0.0         0.0         0.0       
    12229       0.20703     -4241.5712  0.125       -0.1875     0.0         0.0       
    sqlite> 


Many more with z differences::

    sqlite> select * from toxbox where abs(dmix) > 0.1 or abs(dmax) > 0.1 or abs(dmiy) > 0.1 or abs(dmiy) > 0.1 or abs(dmiz) > 0.1 or abs(dmaz) > 0.1 ; 
    t.idx       dmix        dmax        dmiy        dmay        dmiz        dmaz      
    ----------  ----------  ----------  ----------  ----------  ----------  ----------
    2           0.0         0.54297     -0.875      0.0         -0.29004    0.0       
    3           0.19531     0.06445     0.1875      -0.0625     0.0         0.0       
    3148        -0.21875    4241.53809  -0.125      0.375       0.0         0.0       
    3149        0.21289     -4168.5293  0.125       -0.375      0.0         0.0       
    3150        -0.21289    -0.32031    -0.125      0.375       0.0         0.0       
    3165        0.0         0.0         0.0         0.0         -3.06982    0.0       
    4448        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    4449        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    4450        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    4451        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    4452        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    4453        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    4454        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    4455        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    4465        3.375       -7.69531    9.4375      -6.5        0.0         0.0       
    4466        0.0         -4.31445    9.125       -10.0625    0.0         0.0       
    4467        6.55078     -9.47852    3.375       -7.6875     0.0         0.0       
    4468        10.07227    -9.08789    0.0         -4.3125     0.0         0.0       
    4469        7.69531     -3.375      6.5         -9.4375     0.0         0.0       
    4470        4.31445     0.0         10.0625     -9.125      0.0         0.0       
    4471        9.47852     -6.55078    7.6875      -3.375      0.0         0.0       
    4472        9.08789     -10.07227   4.3125      0.0         0.0         0.0       
    4511        0.0         0.0         0.0         0.0         26.21826    0.0       
    4514        0.0         0.0         0.0         0.0         -27.15381   0.0       
    4515        0.0         0.0         0.0         0.0         28.07471    0.0       
    4516        0.0         0.0         0.0         0.0         -28.12061   0.0       
    4517        0.0         0.0         0.0         0.0         3.88184     0.0       
    4520        0.0         0.0         0.0         0.0         -4.82471    0.0       
    4540        0.0         0.0         0.0         0.0         -72.38525   0.00098   
    4551        0.0         0.0         0.0         0.0         -72.38525   10.0354   
    4566        -0.00195    0.0         0.0         0.0         -72.38525   82.76147  
    4567        0.0         0.00195     0.0         0.0         7.09619     -0.00098  
    4606        0.0         0.00391     0.0         0.0625      4.99463     165.05713 
    4628        0.0         0.0         0.0         0.0         -72.38525   0.00098   
    4639        0.0         0.0         0.0         0.0         -72.38525   10.0354   
    4654        -0.00195    0.0         0.0         0.0         -72.38525   82.76147  
    4655        0.0         0.00195     0.0         0.0         7.09619     -0.00098  
    4710        0.0         0.0         0.0         0.0         -72.38525   0.00098   
    4721        0.0         0.0         0.0         0.0         -72.38525   10.0354   
    4736        -0.00195    0.0         0.0         0.0         -72.38525   82.76147  
    4737        0.0         0.00195     0.0         0.0         7.09619     -0.00098  
    4825        0.0         0.0         0.0         0.0         -3.06982    0.0       
    6108        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    6109        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    6110        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    6111        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    6112        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    6113        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    6114        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    6115        0.0         0.0         0.0         0.0         -25.57861   345.57861 
    6125        3.37402     -7.69629    9.4375      -6.5        0.0         0.0       
    6126        0.0         -4.31543    9.125       -10.0625    0.0         0.0       
    6127        6.55078     -9.47852    3.375       -7.6875     0.0         0.0       
    6128        10.07422    -9.08887    0.0         -4.3125     0.0         0.0       
    6129        7.69629     -3.37402    6.5         -9.4375     0.0         0.0       
    6130        4.31543     0.0         10.0625     -9.125      0.0         0.0       
    6131        9.47852     -6.55078    7.6875      -3.375      0.0         0.0       
    6132        9.08887     -10.07422   4.3125      0.0         0.0         0.0       
    6171        0.0         0.0         0.0         0.0         26.21826    0.0       
    6174        0.0         0.0         0.0         0.0         -27.15381   0.0       
    6175        0.0         0.0         0.0         0.0         28.07471    0.0       
    6176        0.0         0.0         0.0         0.0         -28.12061   0.0       
    6177        0.0         0.0         0.0         0.0         3.88184     0.0       
    6180        0.0         0.0         0.0         0.0         -4.82471    0.0       
    6200        -0.00098    0.0         0.0         0.0         -72.38525   0.00098   
    6211        0.00098     0.0         0.0         0.0         -72.38525   10.0354   
    6226        0.0         0.0         0.0         0.0         -72.38525   82.76147  
    6227        0.0         0.0         0.0         0.0         7.09619     -0.00098  
    6266        0.0         0.00391     0.0         0.0625      4.99463     165.05713 
    6288        -0.00098    0.0         0.0         0.0         -72.38525   0.00098   
    6299        0.00098     0.0         0.0         0.0         -72.38525   10.0354   
    6314        0.0         0.0         0.0         0.0         -72.38525   82.76147  
    6315        0.0         0.0         0.0         0.0         7.09619     -0.00098  
    6370        -0.00098    0.0         0.0         0.0         -72.38525   0.00098   
    6381        0.00098     0.0         0.0         0.0         -72.38525   10.0354   
    6396        0.0         0.0         0.0         0.0         -72.38525   82.76147  
    6397        0.0         0.0         0.0         0.0         7.09619     -0.00098  
    12229       0.20703     -4241.5712  0.125       -0.1875     0.0         0.0       
    sqlite> 



