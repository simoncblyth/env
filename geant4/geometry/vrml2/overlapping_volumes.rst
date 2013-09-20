Fully Overlapping volumes, due to imprecise export
==================================================

.. contents:: :local:

executive summary
------------------

The large y dimension values: `-805788` are resulting in a loss of precision
due to use of default streaming in the exported VRML2 geometries. 
Six mother volumes end up exactly overlapping their six daughters, they are very close
concentric tubs.

G4Point3D streaming
--------------------

global/HEPGeometry/include/G4Point3D.hh::

     33 #include "globals.hh"
     34 #include <CLHEP/Geometry/Point3D.h>
     36 typedef HepGeom::Point3D<G4double> G4Point3D;

external/build/LCG/geant4.9.2.p01/source/visualization/VRML/src/G4VRML2SceneHandlerFunc.icc::

    209         G4Point3D point = polyhedron.GetVertex(i);
    211         point.transform( *fpObjectTransformation );
    214         fDest <<                   point.x() << " ";
    215         fDest <<                   point.y() << " ";
    216         fDest <<                   point.z() << "," << "\n";
 
::

     std::ofstream     fDest ;


Probable Fix
---------------

external/build/LCG/geant4.9.2.p01/source/visualization/VRML/src/G4VRML2FileSceneHandler.cc::

    041 #include <iomanip>   // SCB 
    ...
    190     
    191     G4cerr << "Using setprecision(5) and fixed floating point notation for veracity of output [SCB PATCH] " << G4endl;
    192     fDest << std::setprecision(5) << std::fixed ; // SCB
    193 
    194 }


distinct volume count discrepancy
---------------------------------

Before including volume name metadata line in *src* name::

    sqlite> select count(distinct(src)) from shape ; 
    12223

After including the the name line, are 6 more distinct::

    simon:export blyth$ echo "select count(distinct(src)) from shape ;" | sqlite3 -noheader g4_00.db 
    12229       

#. This suggests six absolute position duplicated shapes with different volume names

confirmation of shape overlapping
----------------------------------

Hashing the shape with name excluded confirms issue.
The dodgy dozen, six pairs of volumes are precisely co-located::

    sqlite> select hash, group_concat(name), group_concat(id)  from shape group by hash having count(*) > 1 ;
    hash                              group_concat(name)                                                                                                                           group_concat(id)
    --------------------------------  ---------------------------------------------------------------------------------------------                                                ----------------
    036f14cfb2e7bbe62226d213bd3e7780  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  6400,6401       
    2043a400a35f062979ddfa73254cac9d  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  6318,6319       
    547dd4e8ad4c711815456951753d8fa9  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  4570,4571       
    b7e229d741481e47f3c06236dbc2961d  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  6230,6231       
    be270355bc36384aa290479074aaec4e  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  4658,4659       
    c35f0b07cfa25126ec1b156aca3364d8  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  4740,4741       
    sqlite> 


check detdesc source xml
--------------------------

* :dybsvn:`dybgaudi/trunk/Detector/XmlDetDesc/DDDB/CalibrationSources/sources.xml`

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Detector/XmlDetDesc/DDDB/CalibrationSources/sources.xml?annotate=blame&rev=19722 4 year old XML from Dan

::

    [blyth@cms01 DDDB]$ find . -name '*.xml' -exec grep -H /dd/Geometry/CalibrationSources/lvMainSSTube {} \;
    ./CalibrationSources/sources.xml:             logvol="/dd/Geometry/CalibrationSources/lvMainSSTube">
    [blyth@cms01 DDDB]$ find . -name '*.xml' -exec grep -H /dd/Geometry/CalibrationSources/lvMainSSCavity {} \;
    ./CalibrationSources/sources.xml:             logvol="/dd/Geometry/CalibrationSources/lvMainSSCavity">

::

    443   <!-- AmCCo60 Source SS Tube Structure -->
    444   <logvol name="lvMainSSTube" material="StainlessSteel">
    445     <tubs name="MainSSTube" 
    446           outerRadius="SSTubeOuterRadius"      ### 7.94*mm
    447           sizeZ="SSTubeHeight-2*SSSpringPinOffSet"/>   ## 29.72*mm   29.72-(2*.815)=28.09
    448 
    449     <physvol name="pvMainSSCavity" 
    450              logvol="/dd/Geometry/CalibrationSources/lvMainSSCavity">
    451       <posXYZ x="0*m" y="0*m" z="0*m" />
    452     </physvol>
    453 
    454   </logvol>
    455 
    456   <logvol name="lvMainSSCavity" material="Air">
    457     <tubs name="MainSSCavity" 
    458           sizeZ="AmCCo60AcrylicHeight"            ### 23.6*mm
    459           outerRadius="AmCCo60AcrylicRadius" />   ### 6.55*mm
    460 
    461     <physvol name="pvAmCCo60SourceAcrylic"
    462              logvol="/dd/Geometry/CalibrationSources/lvAmCCo60SourceAcrylic">
    463       <posXYZ x="0*m" y="0*m" z="0*m" />
    464     </physvol>
    465 
    466   </logvol>


parameters.xml::

     68 <parameter name="AmCCo60SourceAcrylicRadius" value="10.035*mm"/>
     69 <parameter name="AmCCo60SourceAcrylicHeight" value="49.8*mm"/>
     70 <parameter name="AmCCo60AcrylicHeight" value="23.6*mm"/>              ### MATCHES
     71 <parameter name="AmCCo60AcrylicRadius" value="6.55*mm"/>
     72 <parameter name="SSTubeOuterRadius" value="7.94*mm"/>
     73 <parameter name="SSTubeHeight" value="29.72*mm"/>
     74 <parameter name="AcrylicSideCavityRadius" value="6.35*mm"/>
     75 <parameter name="AcrylicSideCavityHeight" value="3.8*mm"/>
     76 <parameter name="Co60AlRadius" value="2.5*mm"/>
     77 <parameter name="Co60AlHeight" value="5.0*mm"/>
     78 <parameter name="Co60SourceRadius" value="1.0*mm"/>
     79 <parameter name="Co60SourceHeight" value="2.0*mm"/>
     80 <parameter name="Co60OffSet" value="1.5*mm"/>
     81 <parameter name="AmCContainerOuterRadius" value="3.25*mm"/>
     82 <parameter name="AmCContainerHeight" value="2.0*mm"/>
     83 <parameter name="AmCOffSet" value="2.3*mm"/>
     84 <parameter name="AmCSourceRadius" value="2.5*mm"/>
     85 <parameter name="AmCSourceHeight" value="1.0*mm"/>
     86 <parameter name="AmCSourceCupHeight" value="1.3*mm"/>
     .. 
     88 <parameter name="SSSpringPinRadius" value=".815*mm"/>
     89 <parameter name="SSSpringPinOffSet" value="3.05*mm"/>



correlate extents of exported shapes with expectations 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    sqlite> select sid,npo,ax,ay,az,dx,dy,dz,name from xshape where name like '/dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000' or name like '/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000' ;
    sid         npo         ax          ay          az                 dx                dy          dz                name                                                            
    ----------  ----------  ----------  ----------  -----------------  ----------------  ----------  ----------------  ----------------------------------------------------------------
    4570        50          -18063.584  -799502.16  -4157.12000000001  13.0999999999985  13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
    4571        50          -18063.584  -799502.16  -4157.12000000001  13.0999999999985  13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAc
    4658        50          -17296.992  -798390.84  -4157.12000000001  13.0              13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
    4659        50          -17296.992  -798390.84  -4157.12000000001  13.0              13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAc
    4740        50          -19070.092  -800961.16  -4157.12000000001  13.0              13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
    4741        50          -19070.092  -800961.16  -4157.12000000001  13.0              13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAc
    6230        50          -14944.676  -804323.16  -4157.12000000001  13.1000000000004  13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
    6231        50          -14944.676  -804323.16  -4157.12000000001  13.1000000000004  13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAc
    6318        50          -14178.092  -803212.0   -4157.12000000001  13.0              14.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
    6319        50          -14178.092  -803212.0   -4157.12000000001  13.0              14.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAc
    6400        50          -15951.18   -805782.2   -4157.12000000001  13.1000000000004  13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
    6401        50          -15951.18   -805782.2   -4157.12000000001  13.1000000000004  13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAc
    sqlite> 

::

    [blyth@belle7 export]$ shapedb.py -ce 100 4570 > $(nginx-htdocs)/wrl/4570.wrl
    2013-09-18 16:54:27,945 env.geant4.geometry.export.shapecnf INFO     /home/blyth/env/bin/shapedb.py -ce 100 4570
    2013-09-18 16:54:27,945 env.geant4.geometry.export.shapedb INFO     opening /usr/lib/python2.4/site-packages/env/geant4/geometry/export/g4_01.db 
    2013-09-18 16:54:27,946 env.geant4.geometry.export.shapedb INFO     Operate on 1 shapes, selected by args : [4570] 
    2013-09-18 16:54:27,966 env.geant4.geometry.export.shapedb INFO     opts.center selected, will translate all 1 shapes such that centroid of all is at origin, original coordinate centroid at (-18063.583999999995, -799502.16000000003, -4157.1200000000053) 
    2013-09-18 16:54:27,966 env.geant4.geometry.export.shapedb INFO     #        sid        npo          ax          ay          az          dx          dy          dz 
    2013-09-18 16:54:27,986 env.geant4.geometry.export.shapedb INFO     #       4570         50   -18063.58  -799502.16    -4157.12       13.10       13.00       23.60  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000 
    2013-09-18 16:54:27,987 env.geant4.geometry.export.shapedb INFO     select src_head||x'0A'||group_concat(x'09'||x'09'||x'09'||x'09'||x'09'||(1*(x-(-18063.584)))||' '||(1*(y-(-799502.16)))||' '||(1*(z-(-4157.12)))||',',x'0A')||x'0A'||src_tail from point join shape on shape.id = point.sid where sid in (4570) group by sid ;

* http://belle7.nuu.edu.tw/wrl/4570.wrl


.. image:: 4570.png

Grotty 50 point (24+24+1+1) representation of a Tubs.
Numerical imprecision of the export is the cause of the grottiness.

::

    [blyth@cms01 DDDB]$ find . -name '*.xml' -exec grep -H AmCCo60AcrylicRadius {} \;
    ./CalibrationSources/sources.xml:          outerRadius="AmCCo60AcrylicRadius" />
    ./CalibrationSources/sources.xml:          outerRadius="AmCCo60AcrylicRadius"
    ./CalibrationSources/sources.xml:          sizeZ="2*AmCCo60AcrylicRadius"/>
    ./CalibrationSources/sources.xml:          outerRadius="AmCCo60AcrylicRadius"
    ./CalibrationSources/sources.xml:          outerRadius="AmCCo60AcrylicRadius" />
    ./CalibrationSources/parameters.xml:<parameter name="AmCCo60AcrylicRadius" value="6.55*mm"/>
    [blyth@cms01 DDDB]$ 
    [blyth@cms01 DDDB]$ find . -name '*.xml' -exec grep -H SSTubeOuterRadius {} \;
    ./CalibrationSources/sources.xml:            outerRadius="SSTubeOuterRadius"
    ./CalibrationSources/sources.xml:          outerRadius="SSTubeOuterRadius"
    ./CalibrationSources/sources.xml:          outerRadius="SSTubeOuterRadius"
    ./CalibrationSources/sources.xml:          outerRadius="SSTubeOuterRadius"
    ./CalibrationSources/parameters.xml:<parameter name="SSTubeOuterRadius" value="7.94*mm"/>
    [blyth@cms01 DDDB]$ 



initial hypothesis
~~~~~~~~~~~~~~~~~~~

The closeness of the two tubs with difference of radii `7.94-6.55=1.39 mm` 
(representing the thin stainless steel source shell) triggered 
some overlap (vertex moving) code in the export that results 
in the merging of the volumes.

This merging got done twice resulting in exactly the same shape on both occasions. 

* NOPE: tis simple imprecise export formatting 


Look for others with those names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All six of those names are overlapped::

    sqlite> select * from xshape where name like '/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000' ;
    sid     npo     sumx    ax      minx    maxx    dx      sumy    ay      miny    maxy    dy      sumz    az      minz    maxz    dz      name                                                                                                
    ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ---------------------------------------------------------------------------------------------       
    4571    50      -90317  -18063  -18070  -18057  13.099  -39975  -79950  -79950  -79949  13.0    -20785  -4157.  -4168.  -4145.  23.600  /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000                          
    4659    50      -86484  -17296  -17303  -17290  13.0    -39919  -79839  -79839  -79838  13.0    -20785  -4157.  -4168.  -4145.  23.600  /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000                          
    4741    50      -95350  -19070  -19076  -19063  13.0    -40048  -80096  -80096  -80095  13.0    -20785  -4157.  -4168.  -4145.  23.600  /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000                          
    6231    50      -74723  -14944  -14951  -14938  13.100  -40216  -80432  -80433  -80431  13.0    -20785  -4157.  -4168.  -4145.  23.600  /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000                          
    6319    50      -70890  -14178  -14184  -14171  13.0    -40160  -80321  -80321  -80320  14.0    -20785  -4157.  -4168.  -4145.  23.600  /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000                          
    6401    50      -79755  -15951  -15957  -15944  13.100  -40289  -80578  -80578  -80577  13.0    -20785  -4157.  -4168.  -4145.  23.600  /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000                          
    sqlite> 
    sqlite> select * from xshape where name like '/dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000' ;
    sid     npo     sumx    ax      minx    maxx    dx      sumy    ay      miny    maxy    dy      sumz    az      minz    maxz    dz      name                                                                                                
    ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ---------------------------------------------------------------------------------------------       
    4570    50      -90317  -18063  -18070  -18057  13.099  -39975  -79950  -79950  -79949  13.0    -20785  -4157.  -4168.  -4145.  23.600  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000                                    
    4658    50      -86484  -17296  -17303  -17290  13.0    -39919  -79839  -79839  -79838  13.0    -20785  -4157.  -4168.  -4145.  23.600  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000                                    
    4740    50      -95350  -19070  -19076  -19063  13.0    -40048  -80096  -80096  -80095  13.0    -20785  -4157.  -4168.  -4145.  23.600  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000                                    
    6230    50      -74723  -14944  -14951  -14938  13.100  -40216  -80432  -80433  -80431  13.0    -20785  -4157.  -4168.  -4145.  23.600  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000                                    
    6318    50      -70890  -14178  -14184  -14171  13.0    -40160  -80321  -80321  -80320  14.0    -20785  -4157.  -4168.  -4145.  23.600  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000                                    
    6400    50      -79755  -15951  -15957  -15944  13.100  -40289  -80578  -80578  -80577  13.0    -20785  -4157.  -4168.  -4145.  23.600  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000                                    
    sqlite> 

Visualize, they are distorted small cylinders, widely spaces at same z : making them difficult to see all together : look like dots::

    [blyth@belle7 export]$ shapedb.py -k '/dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000'  > dupe.wrl
    2013-09-13 12:15:19,433 env.geant4.geometry.export.shapecnf INFO     /home/blyth/env/bin/shapedb.py -k /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
    2013-09-13 12:15:19,433 env.geant4.geometry.export.shapedb INFO     opening /usr/lib/python2.4/site-packages/env/geant4/geometry/export/g4_01.db 
    2013-09-13 12:15:19,458 env.geant4.geometry.export.shapedb INFO     Operate on 6 shapes, selected by opts.around "None" opts.like "/dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000" query  
    2013-09-13 12:15:19,458 env.geant4.geometry.export.shapedb INFO     #        sid        npo          ax          ay          az          dx          dy          dz 
    2013-09-13 12:15:19,464 env.geant4.geometry.export.shapedb INFO     #       4570         50   -18063.58  -799502.16    -4157.12       13.10       13.00       23.60  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000 
    2013-09-13 12:15:19,465 env.geant4.geometry.export.shapedb INFO     #       4658         50   -17296.99  -798390.84    -4157.12       13.00       13.00       23.60  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000 
    2013-09-13 12:15:19,467 env.geant4.geometry.export.shapedb INFO     #       4740         50   -19070.09  -800961.16    -4157.12       13.00       13.00       23.60  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000 
    2013-09-13 12:15:19,467 env.geant4.geometry.export.shapedb INFO     #       6230         50   -14944.68  -804323.16    -4157.12       13.10       13.00       23.60  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000 
    2013-09-13 12:15:19,467 env.geant4.geometry.export.shapedb INFO     #       6318         50   -14178.09  -803212.00    -4157.12       13.00       14.00       23.60  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000 
    2013-09-13 12:15:19,476 env.geant4.geometry.export.shapedb INFO     #       6400         50   -15951.18  -805782.20    -4157.12       13.10       13.00       23.60  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000 
    2013-09-13 12:15:19,476 env.geant4.geometry.export.shapedb INFO     select src_head||x'0A'||group_concat(x'09'||x'09'||x'09'||x'09'||x'09'||x||' '||y||' '||z||',',x'0A')||x'0A'||src_tail from point join shape on shape.id = point.sid where sid in (4570,4658,4740,6230,6318,6400) group by sid ;

    [blyth@belle7 export]$ nginx- ; cp dupe.wrl $(nginx-htdocs)/wrl/


Check the viscinity of 4570::

    [blyth@belle7 wrl]$ shapedb.py -ca  -18063.58,-799502.16,-4157.12,1000 > $(nginx-htdocs)/wrl/around_dupe.wrl
    2013-09-13 12:52:23,362 env.geant4.geometry.export.shapecnf INFO     /home/blyth/env/bin/shapedb.py -ca -18063.58,-799502.16,-4157.12,1000
    2013-09-13 12:52:23,362 env.geant4.geometry.export.shapedb INFO     opening /usr/lib/python2.4/site-packages/env/geant4/geometry/export/g4_01.db 
    2013-09-13 12:52:23,389 env.geant4.geometry.export.shapedb INFO     Operate on 151 shapes, selected by opts.around "-18063.58,-799502.16,-4157.12,1000" opts.like "None" query  
    2013-09-13 12:52:23,411 env.geant4.geometry.export.shapedb INFO     opts.center selected, will translate all 151 shapes such that centroid of all is at origin, original coordinate centroid at (-17853.515780398648, -799347.31567694328, -4392.8840961445603) 
    2013-09-13 12:52:23,412 env.geant4.geometry.export.shapedb INFO     #        sid        npo          ax          ay          az          dx          dy          dz 
    2013-09-13 12:52:23,418 env.geant4.geometry.export.shapedb INFO     #       4351        100   -18289.83  -800004.46    -4867.75       60.80       61.00      165.00  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAd2inPmt:1#pvHeadonPmtAssy.1 
    2013-09-13 12:52:23,419 env.geant4.geometry.export.shapedb INFO     #       4352         50   -18289.83  -800004.48    -4909.00       51.90       51.00      112.00  /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtGlass.1000 
    2013-09-13 12:52:23,419 env.geant4.geometry.export.shapedb INFO     #       4353         50   -18289.84  -800004.36    -4909.00       45.90       46.00      106.00  /dd/Geometry/PMT/lvHeadonPmtGlass#pvHeadonPmtVacuum.1000 
    2013-09-13 12:52:23,419 env.geant4.geometry.export.shapedb INFO     #       4354         50   -18289.84  -800004.36    -4961.50       45.90       46.00        1.00  /dd/Geometry/PMT/lvHeadonPmtVacuum#pvHeadonPmtCathode.1000 
    2013-09-13 12:52:23,419 env.geant4.geometry.export.shapedb INFO     #       4355         50   -18289.84  -800004.36    -4908.50       45.90       46.00      105.00  /dd/Geometry/PMT/lvHeadonPmtVacuum#pvHeadonPmtBehindCathode.1001 
    2013-09-13 12:52:23,419 env.geant4.geometry.export.shapedb INFO     #       4356         50   -18289.83  -800004.44    -4826.50       60.80       61.00       53.00  /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtBase.1001 
    2013-09-13 12:52:23,419 env.geant4.geometry.export.shapedb INFO     #       4357         96   -18289.84  -800004.42    -4735.00       73.50       73.00      200.00  /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAd2inPmt:1#pvHeadonPmtMount.1 
    2013-09-13 12:52:23,420 env.geant4.geometry.export.shapedb INFO     #       4425        296   -18118.36  -799755.84    -4988.00     4494.30     4495.00       20.00  /dd/Geometry/AD/lvOIL#pvTopReflector.1429 
    2013-09-13 12:52:23,420 env.geant4.geometry.export.shapedb INFO     #       4426        296   -18118.36  -799755.85    -4988.00     4444.30     4445.00        0.20  /dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap.1000 
    2013-09-13 12:52:23,420 env.geant4.geometry.export.shapedb INFO     #       4427        578   -18379.17  -799831.91    -4987.95     4440.30     4441.00        0.10  /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR.1000 


In Instant Reality Player use `File > Open Location` and enter the URL

* http://belle7.nuu.edu.tw/wrl/around_dupe.wrl



Scene Attribute edits via web interface
----------------------------------------

* S4425 pick large extent volumes to be easy to spot

* http://localhost:35668/Node.html?node=S4425
* http://localhost:35668/setFieldValue?node=S4425&field=diffuseColor&value=0+1+0
* http://localhost:35668/setFieldValue?node=265938224&field=diffuseColor&value=0+1+0
* http://localhost:35668/setFieldValue?node=265938224&field=5&value=TRUE&link=referer

Toggle the bbox for a volume from commandline, unfortunately need to use internal node id, not my name::

    simon:export blyth$ curl "http://localhost:35668/setFieldValue?node=265938224&field=5&value=TRUE&link=referer"
    simon:export blyth$ curl "http://localhost:35668/setFieldValue?node=265938224&field=5&value=FALSE&link=referer"
    simon:export blyth$ curl "http://localhost:35668/setFieldValue?node=265938224&field=5&value=TRUE&link=referer"
    simon:export blyth$ 

Works with external names too::

    simon:export blyth$ curl "http://localhost:35668/setFieldValue?node=S4425&field=5&value=FALSE&link=referer"
    simon:export blyth$ curl "http://localhost:35668/setFieldValue?node=S4425&field=5&value=TRUE&link=referer"




Volume selection with the dupes and context volumes
----------------------------------------------------

* http://belle7.nuu.edu.tw/wrl/adcalib2.wrl

::

    [blyth@belle7 wrl]$ shapedb.py -cq "select sid from xshape where (name like '/dd/Geometry/AD/%' and dx > 1000 and dy > 1000 ) or (name like '/dd/Geometry/CalibrationSources/%' ) ; " > adcalib2.wrl
    2013-09-17 12:20:53,485 env.geant4.geometry.export.shapecnf INFO     /home/blyth/env/bin/shapedb.py -cq select sid from xshape where (name like '/dd/Geometry/AD/%' and dx > 1000 and dy > 1000 ) or (name like '/dd/Geometry/CalibrationSources/%' ) ; 
    2013-09-17 12:20:53,485 env.geant4.geometry.export.shapedb INFO     opening /usr/lib/python2.4/site-packages/env/geant4/geometry/export/g4_01.db 
    2013-09-17 12:20:53,524 env.geant4.geometry.export.shapedb INFO     Operate on 418 shapes, selected by opts.query "select sid from xshape where (name like '/dd/Geometry/AD/%' and dx > 1000 and dy > 1000 ) or (name like '/dd/Geometry/CalibrationSources/%' ) ; " 
    2013-09-17 12:20:53,548 env.geant4.geometry.export.shapedb INFO     opts.center selected, will translate all 418 shapes such that centroid of all is at origin, original coordinate centroid at (-16704.83929964064, -802106.49254530168, -4649.1902775441531) 

    [blyth@belle7 wrl]$ du -hs adcalib2.wrl
    2.3M    adcalib2.wrl



highlight the dupes
~~~~~~~~~~~~~~~~~~~~

::

    sqlite> select sid,npo,ax,ay,az,dx,dy,dz,name from xshape where name like '/dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000' ;
    sid         npo         ax          ay          az                 dx                dy          dz                name                                                            
    ----------  ----------  ----------  ----------  -----------------  ----------------  ----------  ----------------  ----------------------------------------------------------------
    4570        50          -18063.584  -799502.16  -4157.12000000001  13.0999999999985  13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
    4658        50          -17296.992  -798390.84  -4157.12000000001  13.0              13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
    4740        50          -19070.092  -800961.16  -4157.12000000001  13.0              13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
    6230        50          -14944.676  -804323.16  -4157.12000000001  13.1000000000004  13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
    6318        50          -14178.092  -803212.0   -4157.12000000001  13.0              14.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
    6400        50          -15951.18   -805782.2   -4157.12000000001  13.1000000000004  13.0        23.6000000000004  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000

::

    simon:instant_reality_player blyth$ eai-edit -emissiveColor 1,0,0 -transparency 0 4570 4658 4740 6230 6318 6400 
    eai-edit is a function
    eai-edit () 
    { 
        type $FUNCNAME;
        jcli-;
        eai-cd;
        javac -cp $(jcli-jar):$(eai-jar) SceneEdit.java && java -cp $(jcli-jar):$(eai-jar):. SceneEdit $*
    }
    Browser.Name = "Avalon"
    Browser.Version = "V2.3.0 build: R-25322 Jul 18 2013 Mac OS X ppc"
    Browser.CurrentSpeed = 1.0
    Browser.CurrentFrameRate = 23.016539
    Browser.WorldURL = "http://belle7.nuu.edu.tw/wrl/adcalib.wrl"
    applyEdit to node : M4570 type : [Material] : org.instantreality.vrml.eai.net.Node@2f8b5a
    changeFloat transparency from 0.7 to 0.0
    applyEdit to node : M4658 type : [Material] : org.instantreality.vrml.eai.net.Node@7a1767
    changeFloat transparency from 0.7 to 0.0
    applyEdit to node : M4740 type : [Material] : org.instantreality.vrml.eai.net.Node@968fda
    changeFloat transparency from 0.7 to 0.0
    applyEdit to node : M6230 type : [Material] : org.instantreality.vrml.eai.net.Node@be41ec
    changeFloat transparency from 0.7 to 0.0
    applyEdit to node : M6318 type : [Material] : org.instantreality.vrml.eai.net.Node@7ec9f7
    changeFloat transparency from 0.7 to 0.0
    applyEdit to node : M6400 type : [Material] : org.instantreality.vrml.eai.net.Node@fd918a
    changeFloat transparency from 0.7 to 0.0



.. image:: six-dupes.png



first degenerate pair
~~~~~~~~~~~~~~~~~~~~~~~

The large y dimension values: `-805788` are resulting in a loss of precision
due to use of default streaming.


::

    sqlite> select substr(src,0,600) from shape where id = 6401 ;
    #---------- SOLID: /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000
            Shape {
                    appearance Appearance {
                            material Material {
                                    diffuseColor 1 1 1
                                    transparency 0.7
                            }
                    }
                    geometry IndexedFaceSet {
                            coord Coordinate {
                                    point [
                                            -15954.9 -805788 -4145.32,
                                            -15953.4 -805788 -4145.32,
                                            -15951.7 -805789 -4145.32,
                                            -15950 -805789 -4145.32,
                                            -15948.4 -805788 -4145.32,
                                            -15946.9 -805787 -4145.32,
                                            -15945.8 -805786 -4145.32,
                                            -15945 -805784 -4145.32,
                                            -15944.6 -805783 -4145.32,
                                            -15944.7 -805781 -4145.32,
                                            -15945.3 -8
    sqlite> 
    sqlite> 
    sqlite> 
    sqlite> select substr(src,0,600) from shape where id = 6400 ;
    #---------- SOLID: /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
            Shape {
                    appearance Appearance {
                            material Material {
                                    diffuseColor 1 1 1
                                    transparency 0.7
                            }
                    }
                    geometry IndexedFaceSet {
                            coord Coordinate {
                                    point [
                                            -15954.9 -805788 -4145.32,
                                            -15953.4 -805788 -4145.32,
                                            -15951.7 -805789 -4145.32,
                                            -15950 -805789 -4145.32,
                                            -15948.4 -805788 -4145.32,
                                            -15946.9 -805787 -4145.32,
                                            -15945.8 -805786 -4145.32,
                                            -15945 -805784 -4145.32,
                                            -15944.6 -805783 -4145.32,
                                            -15944.7 -805781 -4145.32,
                                            -15945.3 -805779 -414

