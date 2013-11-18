DAE cf WRL
============


.. contents:: :local:

Questions
----------

#. why are AD PMTs in the iPad and meshlab renders all pointing in same direction, and not towards center ?

   * that render was without the ``monkey_matrix_load`` fix ? 

#. VRML2 Y is being rounded to the nearest mm, and X often to nearest 0.1 mm

#. why do the boolean volume vertices depending on which export is done first DAE or WRL  ?

   * this causes between process exports to differ (for boolean volumes), if they differ in export ordering 


Create DAE DB
---------------

Make sure the DB are not preexisting, othewise will add duplicates::

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


Consistent counts
--------------------

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


Compare G4DAE and VRML2 geometry handling code
------------------------------------------------

#. comparing VRML2 and G4DAE code for vertices : looks identical,

   * maybe some parameters : dont think so, all seem at defaults
   * precision issue 
   
.. sidebar:: Promising explanation but seemingly not the case 

   DAE creation so far uses expedient of running from a Geant4 geometry created from an exported GDML file, for development speed. 
   **BUT** that compounds precision issues.  The polyhedron creation algorithm appears sensitive to precise geometry especially
   when you have subtraction/union solids.
   Checked this by testing DAE creation direct from original in memory model, not the one loaded from the GDML. This 
   allows to compare apples-to-apples rather than comparison against 2nd generation geometry filtered thru GDML precision.
   
   The results of that comparison are precisely the same, perhaps some parameter tweaks in VRML2 ?


BooleanProcessor
----------------

``graphics_reps/src/BooleanProcessor.src`` 





G4Polyhedron::SetNumberOfRotationSteps ?
--------------------------------------------

Given that the differences are all in subtraction/union solids it seems unlikely to be 
a difference in such a parameter.  To determine perhaps could add some ``extra`` metadata
to the exported DAE with param values ? 


::

    [blyth@belle7 source]$ find . -exec grep -H G4Polyhedron:: {} \;
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:      G4Polyhedron::SetNumberOfRotationSteps
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:      G4Polyhedron::SetNumberOfRotationSteps(fpMP->GetNoOfSides());
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:    G4Polyhedron::ResetNumberOfRotationSteps();
    ./visualization/management/src/G4VSceneHandler.cc:    G4Polyhedron::SetNumberOfRotationSteps (GetNoOfSides (fpVisAttribs));
    ./visualization/management/src/G4VSceneHandler.cc:    G4Polyhedron::ResetNumberOfRotationSteps ();
    ./geometry/solids/specific/src/G4TwistedTubs.cc:    G4int(G4Polyhedron::GetNumberOfRotationSteps() * dA / twopi) + 2;
    ./geometry/solids/specific/src/G4TwistedTubs.cc:    G4int(G4Polyhedron::GetNumberOfRotationSteps() * fPhiTwist / twopi) + 2;
    ./geometry/solids/specific/src/G4VTwistedFaceted.cc:    G4int(G4Polyhedron::GetNumberOfRotationSteps() * fPhiTwist / twopi) + 2;
    ./geometry/solids/specific/src/G4Polycone.cc:          G4int(G4Polyhedron::GetNumberOfRotationSteps()
    ./geometry/solids/specific/History:  G4Polyhedron::GetNumberOfRotationSteps().
    ./graphics_reps/include/HepPolyhedron.h://    G4Polyhedron::SetNumberOfRotationSteps
    ./graphics_reps/include/HepPolyhedron.h://    G4Polyhedron::ResetNumberOfRotationSteps ();
    ./graphics_reps/src/G4Polyhedron.cc:G4Polyhedron::G4Polyhedron ():
    ./graphics_reps/src/G4Polyhedron.cc:G4Polyhedron::~G4Polyhedron () {}
    ./graphics_reps/src/G4Polyhedron.cc:G4Polyhedron::G4Polyhedron (const HepPolyhedron& from)
    ./graphics_reps/History:- Added G4Polyhedron::Transform and G4Polyhedron::InvertFacets (Evgeni
    [blyth@belle7 source]$ 


``graphics_reps/include/HepPolyhedron.h``::

    105 //   GetNumberOfRotationSteps()   - get number of steps for whole circle;
    106 //   SetNumberOfRotationSteps (n) - set number of steps for whole circle;
    107 //   ResetNumberOfRotationSteps() - reset number of steps for whole circle
    108 //                            to default value;
    109 //   IsErrorBooleanProcess()- true if there has been an error during the
    110 //                            processing of a Boolean operation.
    ...
    168 #ifndef HEP_POLYHEDRON_HH
    169 #define HEP_POLYHEDRON_HH
    170 
    171 #include <CLHEP/Geometry/Point3D.h>
    172 #include <CLHEP/Geometry/Normal3D.h>
    173 
    174 #ifndef DEFAULT_NUMBER_OF_STEPS
    175 #define DEFAULT_NUMBER_OF_STEPS 24
    176 #endif


``LCG/geant4.9.2.p01/source/visualization/management/src/G4VSceneHandler.cc``::

    421 void G4VSceneHandler::RequestPrimitives (const G4VSolid& solid) {
    422   BeginPrimitives (*fpObjectTransformation);
    423   G4NURBS* pNURBS = 0;
    424   G4Polyhedron* pPolyhedron = 0;
    425   switch (fpViewer -> GetViewParameters () . GetRepStyle ()) {
    426   case G4ViewParameters::nurbs:
    427     pNURBS = solid.CreateNURBS ();
    428     if (pNURBS) {
    429       pNURBS -> SetVisAttributes (fpVisAttribs);
    430       AddPrimitive (*pNURBS);
    431       delete pNURBS;
    432       break;
    433     }
    434     else {
    435       G4VisManager::Verbosity verbosity =
    436     G4VisManager::GetInstance()->GetVerbosity();
    437       if (verbosity >= G4VisManager::errors) {
    438     G4cout <<
    439       "ERROR: G4VSceneHandler::RequestPrimitives"
    440       "\n  NURBS not available for "
    441            << solid.GetName () << G4endl;
    442     G4cout << "Trying polyhedron." << G4endl;
    443       }
    444     }
    445     // Dropping through to polyhedron...
    446   case G4ViewParameters::polyhedron:
    447   default:
    448     G4Polyhedron::SetNumberOfRotationSteps (GetNoOfSides (fpVisAttribs));
    449     pPolyhedron = solid.GetPolyhedron ();
    450     G4Polyhedron::ResetNumberOfRotationSteps ();
    451     if (pPolyhedron) {
    452       pPolyhedron -> SetVisAttributes (fpVisAttribs);
    453       AddPrimitive (*pPolyhedron);
    454     }
    455     else {
    456       G4VisManager::Verbosity verbosity =
    457     G4VisManager::GetInstance()->GetVerbosity();
    458       if (verbosity >= G4VisManager::errors) {
    459     G4cout <<
    460       "ERROR: G4VSceneHandler::RequestPrimitives"
    461       "\n  Polyhedron not available for " << solid.GetName () <<
    462       ".\n  This means it cannot be visualized on most systems."
    463       "\n  Contact the Visualization Coordinator." << G4endl;
    464       }
    465     }
    466     break;
    467   }
    468   EndPrimitives ();
    469 }



::

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




Geant4
-------


geometry/solids/Boolean/src/G4UnionSolid.cc::

    453 G4Polyhedron*
    454 G4UnionSolid::CreatePolyhedron () const
    455 {
    456   G4Polyhedron* pA = fPtrSolidA->GetPolyhedron();
    457   G4Polyhedron* pB = fPtrSolidB->GetPolyhedron();
    458   if (pA && pB) {
    459     G4Polyhedron* resultant = new G4Polyhedron (pA->add(*pB));
    460     return resultant;
    461   } else {
    462     std::ostringstream oss;
    463     oss << GetName() <<
    464       ": one of the Boolean components has no corresponding polyhedron.";
    465     G4Exception("G4UnionSolid::CreatePolyhedron",
    466         "", JustWarning, oss.str().c_str());
    467     return 0;
    468   }
    469 }

geometry/solids/Boolean/src/G4SubtractionSolid.cc::

    466 G4Polyhedron*
    467 G4SubtractionSolid::CreatePolyhedron () const
    468 {
    469   G4Polyhedron* pA = fPtrSolidA->GetPolyhedron();
    470   G4Polyhedron* pB = fPtrSolidB->GetPolyhedron();
    471   if (pA && pB)
    472   {
    473     G4Polyhedron* resultant = new G4Polyhedron (pA->subtract(*pB));
    474     return resultant;
    475   }
    476   else
    477   {
    478     std::ostringstream oss;
    479     oss << "Solid - " << GetName()
    480         << " - one of the Boolean components has no" << G4endl
    481         << " corresponding polyhedron. Returning NULL !";
    482     G4Exception("G4SubtractionSolid::CreatePolyhedron()", "InvalidSetup",
    483                 JustWarning, oss.str().c_str());
    484     return 0;
    485   }
    486 }


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

But its real slow at reading DAE. I can adhoc parse WRL quicker. 
WRL import hidden in x3d

* /usr/local/env/graphics/meshlab/meshlab/src/meshlabplugins/io_x3d/io_x3d.h


Also no VRML support in 

* http://assimp.sourceforge.net/main_features_formats.html

* :google:`open source 3D VRML import`

  * https://helixtoolkit.codeplex.com/discussions/403825


Nov 18 2013 : Same Process export : WRL then DAE
--------------------------------------------------

Prep the DB::

    daedb.py --daepath g4_00.dae
    vrml2file.py --save --noshape g4_00.wrl 

Make point comparison::

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


max squared offset per volume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    2465        0.255183839891695
    2478        0.255032854157919
    2487        0.255032936805749
    2526        0.256215984512191
    2527        0.256215984512191
    2638        0.25531043502309 
    2647        0.25531043502309 
    2707        0.255310435022713
    2708        0.255310435022713
    3308        0.256080675965532
    3452        0.256080675965532
    3596        0.256080675965532
    3740        0.256080675965532
    3884        0.256080675965532
    4028        0.256080675965532
    4172        0.256080675965532
    4316        0.256080675965532
    4790        0.257234604314828
    4791        0.257234604314828
    4794        0.257234604314828
    4796        0.257234604314828
    4896        0.256080675965338
    5040        0.256080675965338
    5184        0.256080675965338
    5328        0.256080675965338
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
    2456        0.255183839891695
    2465        0.255183839891695
    2478        0.255046293993105
    2487        0.255048144647476
    2611        0.255303779209358
    2638        0.25531043502309 
    2647        0.25531043502309 
    2707        0.255310435022713
    2708        0.255310435022713
    2814        0.255934838260422
    2858        0.25591536352458 
    3289        0.256226811362892
    3433        0.256226811362892
    3577        0.256226811362892
    3721        0.256226811362892
    3865        0.256226811362892
    4009        0.256226811362892
    4153        0.256226811362892
    4297        0.256226811362892
    4790        0.257234604314828
    4791        0.257234604314828
    4794        0.257234604314828
    4796        0.257234604314828
    4877        0.256311818725851
    5021        0.256311818725851
    5165        0.256311818725851
    5309        0.256311818725851
    5453        0.256311818725851
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


