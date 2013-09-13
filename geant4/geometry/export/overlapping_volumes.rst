Fully Overlapping volumes, dodgy dozen
=======================================

distinct volume count discrepancy
---------------------------------

Before including volume name metadata line in *src* name::

    sqlite> select count(distinct(src)) from shape ; 
    12223

After including the the name line, are 6 more distinct::

    simon:export blyth$ echo "select count(distinct(src)) from shape ;" | sqlite3 -noheader g4_00.db 
    12229       


Observe:

#. 6 more after including the volume name comment metadata first line suggests a small number of absolute position duplicated shapes with different volume names
#. confirmed that assertion using `shape.hash` digest that excludes the name metadata 


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


All size of those names are overlapped::

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



* http://belle7.nuu.edu.tw/wrl/dupe.wrl


first degenerate pair
~~~~~~~~~~~~~~~~~~~~~~~

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




