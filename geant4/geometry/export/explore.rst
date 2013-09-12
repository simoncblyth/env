Explore Geometry DB
=====================

shape
------

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




