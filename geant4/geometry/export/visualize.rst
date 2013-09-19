Geometry Visualization
=======================

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
