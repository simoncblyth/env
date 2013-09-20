Explore ShapeDB
=====================

.. contents:: :local:

Overview
---------

SQLite DB of shapes created by parsing a Geant4 exported VRML2 file.
Allows examination of geometry at level of the vertices of their polyhedronal representations.

Create Geometry DB from .wrl
-----------------------------

::

    cd ~/e/geant4/geometry/export
    ./vrml2file.py --help
    ./vrml2file.py -c g4_00.wrl     

Open Geometry DB
------------------

::

    simon:export blyth$ sqlite3 g4_00.db
    -- Loading resources from /Users/blyth/.sqliterc

    SQLite version 3.7.14.1 2012-10-04 19:37:12
    Enter ".help" for instructions
    Enter SQL statements terminated with a ";"
    sqlite> 


Find the big volumes
---------------------

Volumes that extend more than a cubic meter::

    sqlite> select sid,npo,ax,ay,az,dx,dy,dz,name from xshape where dx > 1000 and dy > 1000 and dz > 1000  ;
    sid         npo         ax          ay          az          dx          dy          dz          name                                                                                                
    ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------  ---------------------------------------------------------------------------------------------       
    1           8           -16519.99   -802110.0   3892.9      69139.8     69140.0     37994.2     /dd/Structure/Sites/db-rock.1000                                                                    
    2           16          -11482.888  -808975.25  2639.855    36494.56    45091.0     15000.29    /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000                                                
    3147        8           -16519.992  -802110.0   -7260.0     17916.63    19696.0     10300.0     /dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot.1001                                                
    3148        34          -15749.992  -802774.5   -7110.0052  13823.3     15602.0     10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead.1000                                                
    3149        50          -16048.293  -803091.86  -7067.9982  13644.59    15422.0     9916.0      /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner.1000                                               
    3150        53          -16292.764  -799918.11  -7361.3286  13644.47    15424.0     9912.0      /dd/Geometry/Pool/lvNearPoolLiner#pvNearPoolOWS.1000                                                
    3151        50          -16085.48   -802990.16  -6565.9996  11506.8     13286.0     8912.0      /dd/Geometry/Pool/lvNearPoolOWS#pvNearPoolCurtain.1000                                              
    3152        53          -16422.016  -800242.98  -6830.8607  11506.7     13286.0     8908.0      /dd/Geometry/Pool/lvNearPoolCurtain#pvNearPoolIWS.1000                                              
    3153        50          -18079.452  -799699.44  -6605.0     5492.9      5493.0      6010.0      /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE1.1000                                                     
    3154        50          -18079.456  -799699.4   -7100.0     4993.6      4993.0      5000.0      /dd/Geometry/AD/lvADE#pvSST.1000                                                                    
    3155        50          -18079.456  -799699.36  -7092.5     4969.7      4969.0      4955.0      /dd/Geometry/AD/lvSST#pvOIL.1000                                                                    
    3156        148         -18079.546  -799699.43  -5790.9070  4074.8      4075.0      4094.71     /dd/Geometry/AD/lvOIL#pvOAV.1000                                                                    
    3157        168         -18079.366  -799699.44  -5674.1325  3958.9      3959.0      4076.53     /dd/Geometry/AD/lvOAV#pvLSO.1000                                                                    
    3158        148         -18079.460  -799699.57  -6066.1087  3126.0      3126.0      3174.49     /dd/Geometry/AD/lvLSO#pvIAV.1000                                                                    
    3159        146         -18079.450  -799699.35  -6062.9378  3096.1      3096.0      3159.39     /dd/Geometry/AD/lvIAV#pvGDS.1000                                                                    
    4813        50          -14960.548  -804520.56  -6605.0     5492.9      5493.0      6010.0      /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE2.1001                                                     
    4814        50          -14960.544  -804520.6   -7100.0     4993.6      4993.0      5000.0      /dd/Geometry/AD/lvADE#pvSST.1000                                                                    
    4815        50          -14960.544  -804520.64  -7092.5     4969.7      4969.0      4955.0      /dd/Geometry/AD/lvSST#pvOIL.1000                                                                    
    4816        148         -14960.643  -804520.58  -5790.9070  4074.8      4075.0      4094.71     /dd/Geometry/AD/lvOIL#pvOAV.1000                                                                    
    4817        168         -14960.458  -804520.54  -5674.1325  3958.9      3959.0      4076.53     /dd/Geometry/AD/lvOAV#pvLSO.1000                                                                    
    4818        148         -14960.551  -804520.68  -6066.1087  3125.9      3126.0      3174.49     /dd/Geometry/AD/lvLSO#pvIAV.1000                                                                    
    4819        146         -14960.543  -804520.48  -6062.9378  3096.1      3096.0      3159.39     /dd/Geometry/AD/lvIAV#pvGDS.1000                                                                    
    12221       8           -20946.875  -795267.0   -7260.0     3513.9      2420.0      10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab1.1001                         
    12222       8           -23132.875  -798523.0   -7260.0     1184.0      4218.0      10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab2.1002                         
    12223       8           -20844.05   -804907.5   -7260.0     5678.7      8551.0      10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab3.1003                         
    12224       8           -15958.85   -809612.25  -7260.0     4217.5      1184.0      10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab4.1004                         
    12225       8           -12093.125  -808953.0   -7260.0     3513.9      2420.0      10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab5.1005                         
    12226       8           -9907.1275  -805697.0   -7260.0     1183.96     4218.0      10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab6.1006                         
    12227       8           -12195.94   -799312.5   -7260.0     5678.71     8551.0      10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab7.1007                         
    12228       8           -17081.15   -794607.75  -7260.0     4217.5      1184.0      10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab8.1008                         
    sqlite> 


Ordering by decreasing extent in x 
----------------------------------

::

    sqlite> select dx, dy, dz, name from xshape join shape on xshape.sid == shape.id order by dx desc limit 100 ;
    dx          dy          dz          name                                                                                                
    ----------  ----------  ----------  ---------------------------------------------------------------------------------------------       
    69139.8     69140.0     37994.2     /dd/Structure/Sites/db-rock.1000                                                                    
    36494.56    45091.0     15000.29    /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000                                                
    20239.92    21988.0     293.51      /dd/Geometry/Sites/lvNearHallTop#pvNearRPCRoof.1003                                                 
    19770.52    21602.0     294.0       /dd/Geometry/Sites/lvNearHallTop#pvNearRPCSptRoof.1004                                              
    17916.63    19696.0     10300.0     /dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot.1001                                                
    13823.3     15602.0     10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead.1000                                                
    13823.18    15602.0     300.0       /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab9.1009                         
    13823.07    15602.0     44.0        /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover.1000                                       
    ...
    4759.2      7326.0      40.0        /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearLongEdgeUnistruts:2#pvNearLongQuadEdgeUnistrus
    4759.2      7326.0      40.0        /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearLongEdgeUnistruts:2#pvNearLongQuadEdgeUnistrus
    4759.2      7326.0      40.0        /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearLongEdgeUnistruts:2#pvNearLongQuadEdgeUnistrus
    4494.3      4495.0      20.0        /dd/Geometry/AD/lvOIL#pvTopReflector.1429                                                           
    4494.3      4495.0      20.0        /dd/Geometry/AD/lvOIL#pvBotReflector.1430                                                           
    4494.3      4495.0      20.0        /dd/Geometry/AD/lvOIL#pvTopReflector.1429                                                           
    4494.3      4495.0      20.0        /dd/Geometry/AD/lvOIL#pvBotReflector.1430                                                           
    4444.3      4445.0      0.20000000  /dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap.1000                                              
    4444.3      4445.0      0.20000000  /dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap.1000                                              
    4444.3      4445.0      0.20000000  /dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap.1000                                              
    4444.3      4445.0      0.20000000  /dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap.1000                                              
    4440.3      4441.0      0.10000000  /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR.1000                                                    
    4440.3      4441.0      0.09999999  /dd/Geometry/AdDetails/lvBotRefGap#pvBotESR.1000                                                    
    4440.3      4441.0      0.10000000  /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR.1000                                                    
    4440.3      4441.0      0.09999999  /dd/Geometry/AdDetails/lvBotRefGap#pvBotESR.1000                                                    
    4217.5      1184.0      10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab8.1008                         
    4217.5      1184.0      10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab4.1004                         
    4074.8      4075.0      4094.71     /dd/Geometry/AD/lvOIL#pvOAV.1000                                                                    
    4074.8      4075.0      4094.71     /dd/Geometry/AD/lvOIL#pvOAV.1000                                                                    
    3958.9      3959.0      4076.53     /dd/Geometry/AD/lvOAV#pvLSO.1000                                                                    
    3958.9      3959.0      4076.53     /dd/Geometry/AD/lvOAV#pvLSO.1000                                                                    
    3919.6      878.0       40.0        /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:1#pvNearQuadCornerUnistrus:2#pvC
    3919.6      878.0       40.0        /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:1#pvNearQuadCornerUnistrus:2#pvC


shape counting
----------------

* NB names are not unique

::

    sqlite> select count(distinct(name)) from shape ; 
    count(distinct(name))                                                                               
    ---------------------------------------------------------------------------------------------       
    5642            

::

    sqlite> select name, count(*) as N from shape where name like '/dd/Geometry/PMT/%' group by name ;
    name                                                                                                  N         
    ---------------------------------------------------------------------------------------------         ----------
    /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtBase.1001                                                 12        
    /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtGlass.1000                                                12        
    /dd/Geometry/PMT/lvHeadonPmtGlass#pvHeadonPmtVacuum.1000                                              12        
    /dd/Geometry/PMT/lvHeadonPmtVacuum#pvHeadonPmtBehindCathode.1001                                      12        
    /dd/Geometry/PMT/lvHeadonPmtVacuum#pvHeadonPmtCathode.1000                                            12        
    /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000                                                       672       
    /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom.1001                                                 672       
    /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000                                                672       
    /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode.1002                                                 672       
    sqlite> 


::

    sqlite> select id,min(ax),max(ax),max(ax)-min(ax),min(ay),max(ay),max(ay)-min(ay),min(az),max(az),max(az)-min(az),name from shape join xshape on shape.id = xshape.sid where name like '/dd/Geometry/PMT/%' group by name ;
    id          min(ax)     max(ax)     max(ax)-mi  min(ay)     max(ay)     max(ay)-min(ay)   min(az)     max(az)     max(az)-min(az)  name                                                 
    ----------  ----------  ----------  ----------  ----------  ----------  ----------------  ----------  ----------  ---------------  -----------------------------------------------------
    4356        -20227.02   -15170.928  5056.092    -806121.56  -799597.32  6524.24000000011  -9358.5     -4826.5     4532.0           /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtBase.1001
    4352        -20227.024  -15170.92   5056.104    -806121.52  -799597.12  6524.40000000002  -9276.0     -4909.0     4367.0           /dd/Geometry/PMT/lvHeadonPmtAssy#pvHeadonPmtGlass.100
    4353        -20227.02   -15170.936  5056.084    -806121.52  -799597.24  6524.28000000003  -9276.0     -4909.0     4367.0           /dd/Geometry/PMT/lvHeadonPmtGlass#pvHeadonPmtVacuum.1
    4355        -20227.02   -15170.936  5056.084    -806121.52  -799597.24  6524.28000000003  -9276.5     -4908.5     4368.0           /dd/Geometry/PMT/lvHeadonPmtVacuum#pvHeadonPmtBehindC
    4354        -20227.02   -15170.936  5056.084    -806121.52  -799597.24  6524.28000000003  -9223.5     -4961.5     4262.0           /dd/Geometry/PMT/lvHeadonPmtVacuum#pvHeadonPmtCathode
    7596        -22949.841  -10090.079  12859.7621  -809429.31  -794790.79  14638.5207100591  -11698.076  -2602.3031  9095.7734319526  /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000      
    7598        -22961.933  -10077.982  12883.9512  -809441.32  -794778.75  14662.5743801653  -11710.447  -2602.3     9108.1479338842  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom.1001
    7597        -22877.517  -10162.404  12715.1128  -809356.95  -794863.15  14493.7987551867  -11624.127  -2602.3036  9021.8237136929  /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.100
    7599        -23051.25   -9988.6614  13062.5886  -809530.7   -794689.46  14841.24          -11801.8    -2602.3     9199.5           /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode.1002
    sqlite> 





shape join xshape
------------------

::

    sqlite> select id, npo, name, hash from shape join xshape on shape.id = xshape.sid where xshape.npo > 400 ;
    id          npo         name                                                                                                  hash                            
    ----------  ----------  ---------------------------------------------------------------------------------------------         --------------------------------
    3201        482         /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000                                                9974386e5bd0e4966627df7e927b7a38
    3207        482         /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000                                                50bf1403c1514e09064082b2941f47bf
    3213        482         /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000                                                de08e1c595795a77403997c09ebd5e11
    3219        482         /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000                                                8e0a67ba1785b3f5c6c50bd197198767
    3225        482         /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000                                                ff0de4e2f48c5f0e79bedaf5049505c5


::

    sqlite> select distinct(name) from shape join xshape on shape.id = xshape.sid where xshape.npo > 365 ;
    name                                                                                                
    ---------------------------------------------------------------------------------------------       
    /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000                                              
    /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR.1000                                                    
    /dd/Geometry/CalibrationBox/lvDomeInterior#pvLedSourceAssyInAcu.1003                                
    /dd/Geometry/AD/lvADE#pvOflTnkContainer.1002                                                        
    /dd/Geometry/OverflowTanks/lvOflTnkCnrSpace#pvGdsOflTnk.1002                                        
    sqlite> 


::

    sqlite> select name, npo, count(*) as N from shape join xshape on shape.id = xshape.sid group by name order by npo desc limit 100 ;
    name                                                                                                  npo         N                                                                                                   
    ---------------------------------------------------------------------------------------------         ----------  ---------------------------------------------------------------------------------------------       
    /dd/Geometry/OverflowTanks/lvOflTnkCnrSpace#pvGdsOflTnk.1002                                          776         2                                                                                                   
    /dd/Geometry/CalibrationBox/lvDomeInterior#pvLedSourceAssyInAcu.1003                                  629         6                                                                                                   
    /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR.1000                                                      578         2                                                                                                   
    /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000                                                482         672                                                                                                 
    /dd/Geometry/AD/lvADE#pvOflTnkContainer.1002                                                          366         2                                                                                                   
    /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUni  362         2                                                                                                   
    /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:10#pvAdPmtUn  362         2                                                                                                   
    /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:11#pvAdPmtUn  362         2                                                                                                   
    /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:12#pvAdPmtUn  362         2                                                                                                   
    /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:13#pvAdPmtUn  362         2                                                                                                   
    /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:14#pvAdPmtUn  362         2                                                                                                   
    /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:15#pvAdPmtUn  362         2                     



xshape
-------

Created via a `from point group by sid` query to give extents of every shape in a single statment::

  create table xshape as select sid, count(*) as npo, sum(x) as sumx, avg(x) as ax, min(x) as minx, max(x) as maxx, max(x)-min(x) as dx,... from point group by sid 

::

    sqlite> .schema xshape
    CREATE TABLE xshape(
      sid INT, npo, 
      sumx, ax, minx, maxx, dx,
      sumy, ay, miny, maxy, dy,
      sumz, az, minz, maxz, dz
    );



ranging the cetroids
~~~~~~~~~~~~~~~~~~~~~~~

::

    sqlite> select min(ax),max(ax),max(ax)-min(ax),min(ay),max(ay),max(ay)-min(ay),min(az),max(az),max(az)-min(az) from xshape ;
    min(ax)     max(ax)     max(ax)-min(ax)  min(ay)     max(ay)      max(ay)-min(ay)  min(az)     max(az)     max(az)-min(az)
    ----------  ----------  ---------------  ----------  -----------  ---------------  ----------  ----------  ---------------
    -25917.55   -7335.825   18581.725        -812374.5   -791974.375  20400.125        -12410.0    3892.9      16302.9        


primitive binning  
~~~~~~~~~~~~~~~~~~~~~~

::

    sqlite> select count(*) as N, min(ax),max(ax),(ax-(-25917.55))/18581.725,round(10*((ax-(-25917.55))/18581.725)) from xshape group by round(10*((ax-(-25917.55))/18581.725)) ;
    N           min(ax)     max(ax)     (ax-(-25917.55))/18581.725  round(10*((ax-(-25917.55))/18581.725))
    ----------  ----------  ----------  --------------------------  --------------------------------------
    37          -25917.55   -25041.75   0.0471323302868815          0.0                                   
    145         -24942.075  -23132.875  0.149860952091369           1.0                                   
    1157        -23123.533  -21279.475  0.249604113719259           2.0                                   
    1578        -21270.79   -19414.85   0.349951363503658           3.0                                   
    1819        -19407.912  -17556.925  0.44993804396524            4.0                                   
    2616        -17545.95   -15704.037  0.54965362473075            5.0                                   
    1843        -15696.7    -13841.35   0.649896605401275           6.0                                   
    1488        -13839.333  -11988.222  0.749625079659361           7.0                                   
    1298        -11980.75   -10123.123  0.849997836135594           8.0                                   
    211         -10114.547  -8275.785   0.949414814824781           9.0                                   
    37          -8249.255   -7335.825   1.0                         10.0     

::

    sqlite> select count(*) as N, min(ay),max(ay),(ay-(-812374.5))/20400.125,round(10*(ay-(-812374.5))/20400.125) from xshape group by round(10*(ay-(-812374.5))/20400.125) ;
    N           min(ay)     max(ay)     (ay-(-812374.5))/20400.125  round(10*(ay-(-812374.5))/20400.125)
    ----------  ----------  ----------  --------------------------  ------------------------------------
    37          -812374.5   -811397.5   0.0478918634076997          0.0                                 
    270         -811304.0   -809315.0   0.149974571234245           1.0                                 
    1336        -809308.5   -807282.0   0.249630823340543           2.0                                 
    1735        -807270.62  -805241.89  0.349635600689025           3.0                                 
    1777        -805233.8   -803196.12  0.4499175862893             4.0                                 
    1917        -803190.37  -801158.33  0.549808462803515           5.0                                 
    1763        -801152.5   -799115.62  0.649940870460353           6.0                                 
    1714        -799094.87  -797078.0   0.749823836863745           7.0                                 
    1331        -797073.75  -795037.0   0.849872243429881           8.0                                 
    308         -795032.5   -793052.62  0.947144931709977           9.0                                 
    41          -792951.5   -791974.37  1.0                         10.0                     

::

    sqlite> select count(*) as N, min(az),max(az),(az-(-12410.0))/16302.9,round(10*(az-(-12410.0))/16302.9) from xshape group by round(10*(az-(-12410.0))/16302.9) ;
    N           min(az)     max(az)     (az-(-12410.0))/16302.9  round(10*(az-(-12410.0))/16302.9)
    ----------  ----------  ----------  -----------------------  ---------------------------------
    508         -12410.0    -11622.925  0.0482782204393083       0.0                              
    1121        -11557.708  -10324.0    0.127952695532697        1.0                              
    1964        -9935.2825  -8342.4963  0.249495713780933        2.0                              
    1313        -8062.62    -6768.615   0.346035674634574        3.0                              
    2219        -6703.506   -5081.845   0.449500088941231        4.0                              
    1022        -5050.7824  -3618.5175  0.539258812849248        5.0                              
    937         -3269.2825  -2088.0     0.633138889400045        6.0                              
    3059        -1724.37    -1172.69    0.689282888320483        7.0                              
    84          669.904     746.664     0.807013721485135        8.0                              
    1           2639.855    2639.855    0.923139748142968        9.0                              
    1           3892.9      3892.9      1.0                      10.0                 




point
------

::

    sqlite> select count(*) from point ;
    count(*)  
    ----------
    1246038   

    sqlite> select min(x) as minx,max(x) as maxx,max(x)-min(x) as rngx,avg(x) as avgx,min(y) as miny,max(y) as maxy,max(y)-min(y) as rngy,avg(y) as avgy,min(z) as minz,max(z) as maxz,max(z)-min(z) as rngz,avg(z) as avgz from point ;
    min(x)      max(x)      max(x)-min(x)  avg(x)             min(y)      max(y)      max(y)-min(y)  avg(y)             min(z)      max(z)      max(z)-min(z)  avg(z)           
    ----------  ----------  -------------  -----------------  ----------  ----------  -------------  -----------------  ----------  ----------  -------------  -----------------
    -51089.9    18049.9     69139.8        -16532.7789121053  -836680.0   -767540.0   69140.0        -802114.819727007  -15104.2    22890.0     37994.2        -7055.12378466854



Primitive histogramming the 1.2M entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Map x to 0 to 1::

    sqlite> select min((x-(-51089.0))/69139.8), max((x-(-51089.0))/69139.8) from point ; 
    min((x-(-51089.0))/69139.8)  max((x-(-51089.0))/69139.8)
    ---------------------------  ---------------------------
    -1.30171044753016e-05        0.999986982895525  


::

    sqlite> select count(*) as N, min(x),max(x),round((x-(-51089.0))/69139.8*10,0) from point group by round((x-(-51089.0))/69139.8*10,0) ;
    N           min(x)      max(x)      round((x-(-51089.0))/69139.8*10,0)
    ----------  ----------  ----------  ----------------------------------
    2           -51089.9    -51089.9    0.0                               
    2           -31088.1    -31088.1    3.0                               
    194610      -26664.5    -19976.1    4.0                               
    859647      -19976.0    -13062.2    5.0                               
    191769      -13062.1    -6424.58    6.0                               
    4           -3828.12    -3827.51    7.0                               
    2           5406.46     5406.46     8.0                               
    2           18049.9     18049.9     10.0            





Simple binned querying 

* http://stackoverflow.com/questions/1764881/mysql-getting-data-for-histogram-plot

::

    SELECT b.*,count(*) as total FROM bins b 
    left outer join table1 a on a.value between b.min_value and b.max_value 
    group by b.min_value


xshape
--------

That 2 is the 2 ADs::

    sqlite> select sid,npo,count(*) as N,group_concat(round(ax,1)),group_concat(round(ay,1)),group_concat(round(az,1)),group_concat(round(dx,1)),group_concat(round(dy,1)),group_concat(round(dz,1)),name from xshape where name like '/dd/Geometry/AD/%' group by name ;
    sid         npo         N           group_concat(round(a  group_concat(round(a  group_concat(round(a  group_concat(round(d  group_concat(round(d  group_concat(round(d  name                                                                                                                                                  
    ----------  ----------  ----------  --------------------  --------------------  --------------------  --------------------  --------------------  --------------------  ---------------------------------------------------------------------------------------------                                                         
    6434        16          2           -20200.1,-17081.2     -801071.4,-805892.4   -7100.0,-7100.0       66.2,66.2             72.0,72.0             5000.0,5000.0         /dd/Geometry/AD/lvADE#pvAdVertiCableTray.1005                                                                                                         
    6193        50          2           -18014.5,-14895.6     -799605.4,-804426.4   -3955.5,-3955.5       798.1,798.1           798.0,798.0           689.0,689.0           /dd/Geometry/AD/lvADE#pvCenterCalibE.1001                                                                                                             
    6443        50          2           -17778.8,-14659.9     -801337.5,-806158.6   -4469.8,-4469.8       608.2,608.2           609.0,608.0           260.4,260.4           /dd/Geometry/AD/lvADE#pvElectricalDistributionBoxE.1007                                                                                               
    6352        50          2           -19021.1,-15902.1     -801064.4,-805885.4   -4105.5,-4105.5       798.1,798.1           798.0,798.0           989.0,989.0           /dd/Geometry/AD/lvADE#pvGCatCalibE.1004                                                                                                               
    6436        50          2           -18380.1,-15261.2     -798061.4,-802882.5   -4469.8,-4469.8       608.2,608.2           608.0,609.0           260.4,260.4           /dd/Geometry/AD/lvADE#pvGasDistributionBoxE.1006                                                                                                      
    6450        50          2           -17454.6,-14335.7     -797668.4,-802489.5   -4492.5,-4492.5       437.7,437.7           437.0,437.0           215.0,215.0           /dd/Geometry/AD/lvADE#pvMOClarityBoxE.1008                                                                                                            
    6270        50          2           -17248.0,-14129.1     -798494.1,-803315.3   -4105.5,-4105.5       798.1,798.1           798.0,798.0           989.0,989.0           /dd/Geometry/AD/lvADE#pvOffCenterCalibE.1003                                                                                                          
    6264        366         2           -18057.8,-14938.9     -799668.1,-804489.2   -4380.5,-4380.5       2027.4,2027.4         2027.0,2027.0         300.0,300.0           /dd/Geometry/AD/lvADE#pvOflTnkContainer.1002                                                                                                          
    4814        50          2           -18079.5,-14960.5     -799699.4,-804520.6   -7100.0,-7100.0       4993.6,4993.6         4993.0,4993.0         5000.0,5000.0         /dd/Geometry/AD/lvADE#pvSST.1000                                                                                                                      
    6457        50          2           -16620.4,-13501.5     -800706.0,-805527.0   -4451.5,-4451.5       1315.7,1315.7         1316.0,1316.0         297.0,297.0           /dd/Geometry/AD/lvADE#pvlvMOOverflowTankE1.1009                                                                                                       
    6465        50          2           -19538.5,-16419.6     -798693.0,-803514.0   -4451.5,-4451.5       1315.7,1315.7         1316.0,1316.0         297.0,297.0           /dd/Geometry/AD/lvADE#pvlvMOOverflowTankE2.1010                                                                                                       
    4819        146         2           -18079.5,-14960.5     -799699.4,-804520.5   -6062.9,-6062.9       3096.1,3096.1         3096.0,3096.0         3159.4,3159.4         /dd/Geometry/AD/lvIAV#pvGDS.1000                                                                                                                      
    4820        48          2           -18846.0,-15727.1     -800810.7,-805631.7   -5549.0,-5549.0       62.9,62.9             63.0,63.0             18.0,18.0             /dd/Geometry/AD/lvIAV#pvOcrGdsInIAV.1001                                                                                                              
    4822        192         2           -18079.5,-14960.5     -799699.5,-804520.5   -5345.0,-5345.0       299.7,299.7           299.0,299.0           30.0,30.0             /dd/Geometry/AD/lvLSO#pvCtrGdsOflBotClp.1002                                                                                                          
    4824        50          2           -18079.5,-14960.5     -799699.4,-804520.6   -5245.5,-5245.5       62.9,62.9             63.0,63.0             460.2,460.2           /dd/Geometry/AD/lvLSO#pvCtrGdsOflInLso.1004                                                                                                           



::

    sqlite> select sid,npo,count(*) as N,group_concat(round(ax,1)),group_concat(round(ay,1)),group_concat(round(az,1)),group_concat(round(dx,1)),group_concat(round(dy,1)),group_concat(round(dz,1)),name from xshape where name like '/dd/Geometry/AD/%' and dx > 1000 and dy > 1000 group by name ;
    sid         npo         N           group_concat(round(a  group_concat(round(a  group_concat(round(a  group_concat(round(d  group_concat(round(d  group_concat(round(d  name                                                                                                                                                  
    ----------  ----------  ----------  --------------------  --------------------  --------------------  --------------------  --------------------  --------------------  ---------------------------------------------------------------------------------------------                                                         
    6264        366         2           -18057.8,-14938.9     -799668.1,-804489.2   -4380.5,-4380.5       2027.4,2027.4         2027.0,2027.0         300.0,300.0           /dd/Geometry/AD/lvADE#pvOflTnkContainer.1002                                                                                                          
    4814        50          2           -18079.5,-14960.5     -799699.4,-804520.6   -7100.0,-7100.0       4993.6,4993.6         4993.0,4993.0         5000.0,5000.0         /dd/Geometry/AD/lvADE#pvSST.1000                                                                                                                      
    6457        50          2           -16620.4,-13501.5     -800706.0,-805527.0   -4451.5,-4451.5       1315.7,1315.7         1316.0,1316.0         297.0,297.0           /dd/Geometry/AD/lvADE#pvlvMOOverflowTankE1.1009                                                                                                       
    6465        50          2           -19538.5,-16419.6     -798693.0,-803514.0   -4451.5,-4451.5       1315.7,1315.7         1316.0,1316.0         297.0,297.0           /dd/Geometry/AD/lvADE#pvlvMOOverflowTankE2.1010                                                                                                       
    4819        146         2           -18079.5,-14960.5     -799699.4,-804520.5   -6062.9,-6062.9       3096.1,3096.1         3096.0,3096.0         3159.4,3159.4         /dd/Geometry/AD/lvIAV#pvGDS.1000                                                                                                                      
    4818        148         2           -18079.5,-14960.6     -799699.6,-804520.7   -6066.1,-6066.1       3126.0,3125.9         3126.0,3126.0         3174.5,3174.5         /dd/Geometry/AD/lvLSO#pvIAV.1000                                    



For a simple view of just the bigger bits of geometry::

    [blyth@belle7 export]$ shapedb.py -cq "select sid from xshape where name like '/dd/Geometry/AD/%' and dx > 1000 and dy > 1000" > ads.wrl
    2013-09-12 20:38:38,914 env.geant4.geometry.export.shapecnf INFO     /home/blyth/env/bin/shapedb.py -c -q select sid from xshape where name like '/dd/Geometry/AD/%' and dx > 1000 and dy > 1000
    2013-09-12 20:38:38,914 env.geant4.geometry.export.shapedb INFO     opening /usr/lib/python2.4/site-packages/env/geant4/geometry/export/g4_01.db 
    2013-09-12 20:38:38,938 env.geant4.geometry.export.shapedb INFO     Operate on 70 shapes, selected by opts.query "select sid from xshape where name like '/dd/Geometry/AD/%' and dx > 1000 and dy > 1000" 
    2013-09-12 20:38:38,958 env.geant4.geometry.export.shapedb INFO     opts.center selected, will translate all 70 shapes such that centroid of all is at origin, original coordinate centroid at (-16521.718058206105, -802112.48974236636, -5897.2523902671737) 









