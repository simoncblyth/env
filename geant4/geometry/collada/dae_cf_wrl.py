#!/usr/bin/env python
"""

::

    cat << EOS > dae_cf_wrl.sql 
    attach database "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.db" as dae ;
    attach database "$LOCAL_BASE/env/geant4/geometry/vrml2/g4_01.db" as wrl ;
    .databases
    EOS
    sqlite3 -init dae_cf_wrl.sql 

    sqlite> select count(*) from dae.geom ;
    12230                                                                                                                                                                                                                                                         
    sqlite> select count(*) from wrl.shape ;   # world volume was culled for this wrl export
    12229
    sqlite> select count(*) from wrl.xshape ;
    12229


::

    sqlite>  select sid, name from wrl.xshape limit 5000,10 ;
    sid         name                                                                                                                                                                                                    
    ----------  ---------------------------------------------------------------------------------------------                                                                                                           
    5001        /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode.1002                                                                                                                                                   
    5002        /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:24#pvAdPmtUnit#pvAdPmtCollar.1                                                                                  
    5003        /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:2#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt.2                                                                                         
    5004        /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000                                                                                                                                                         
    5005        /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode.1000                                                                                                                                                  
    5006        /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom.1001                                                                                                                                                   
    5007        /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode.1002                                                                                                                                                   
    5008        /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:2#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmtCollar.2                                                                                   
    5009        /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:2#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmt.2                                                                                         
    5010        /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum.1000                                                                                                                                                         
    sqlite> 
    sqlite> select idx, name from dae.geom limit 5000+1,10 ;
    idx         name                                                                                                                                                                                                    
    ----------  ---------------------------------------------------------------------------------------------                                                                                                           
    5001        __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiDynode0xa8d6e58.215                                                                                                                                      
    5002        __dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..24--pvAdPmtUnit--pvAdPmtCollar0xa8dce38.1                                                             
    5003        __dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..2--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8dcfb0.1                                                                    
    5004        __dd__Geometry__PMT__lvPmtHemi--pvPmtHemiVacuum0xa8d6ee8.216                                                                                                                                            
    5005        __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiCathode0xa8d6ab0.216                                                                                                                                     
    5006        __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiBottom0xa8d6db8.216                                                                                                                                      
    5007        __dd__Geometry__PMT__lvPmtHemiVacuum--pvPmtHemiDynode0xa8d6e58.216                                                                                                                                      
    5008        __dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..2--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmtCollar0xa8dd150.1                                                              
    5009        __dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..2--pvAdPmtInRing..2--pvAdPmtUnit--pvAdPmt0xa8dd268.1                                                                    
    5010        __dd__Geometry__PMT__lvPmtHemi--pvPmtHemiVacuum0xa8d6ee8.217                                                                                                                                            
    sqlite> 

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

    sqlite> select count(*) from wrl.xshape w inner join dae.geom d on w.sid = d.idx where w.npo != d.nvertex ;
    1688              # ouch 14% of volumes have different vertex counts  
    sqlite> select count(*) from wrl.xshape w inner join dae.geom d on w.sid = d.idx where w.npo = d.nvertex ;
    10541     

::

    sqlite> select d.idx, d.nvertex, w.npo, w.npo-d.nvertex, count(*) as N, group_concat(distinct(d.nvertex)) from wrl.xshape w inner join dae.geom d on w.sid = d.idx where w.npo != d.nvertex  group by w.npo ;
    idx         nvertex     npo         w.npo-d.nv  N           group_concat(distinct(d.nvertex))
    ----------  ----------  ----------  ----------  ----------  ---------------------------------------------------------------------------------------------
    3076        10          12          2           144         10
    4854        22          16          -6          17          20,22
    6164        16          33          17          16          16
    6132        48          34          -14         17          50,48
    6098        15          35          20          16          15
    3           34          40          6           1           34
    6175        98          49          -49         2           98
    12229       34          50          16          50          34,342
    3150        78          53          -25         1           78
    6124        192         96          -96         14          192
    6180        49          98          49          6           49
    6046        122         100         -22         12          122
    4817        170         168         -2          2           170
    6266        480         192         -288        6           288,480
    6398        150         194         44          6           150
    6397        342         233         -109        6           342
    4825        192         288         96          2           192
    6396        775         296         -479        6           775
    11409       334         338         4           672         334
    6381        780         357         -423        6           780
    6187        316         360         44          6           316
    11408       360         362         2           672         360
    6264        344         366         22          2           344
    6370        778         629         -149        6           778


::

    sqlite> select count(distinct(geoid)) from dae.geom ;    ## 34 shapes out of 249 are vertex count discrepant 
    249

    sqlite> select d.idx, group_concat(distinct(d.nvertex)) as dae_nvtx, group_concat(distinct(w.npo)) as wrl_npo, w.npo-d.nvertex, count(*) as N, d.geoid from wrl.xshape w inner join dae.geom d on w.sid = d.idx where w.npo != d.nvertex  group by d.geoid ;
    idx         dae_nvtx    wrl_npo     w.npo-d.nvertex  N           geoid                  
    ----------  ----------  ----------  ---------------  ----------  -----------------------
    6397        342         233         -109             6           AmCCo60AcrylicContainer
    6398        150         194         44               6           AmCCo60Cavity          
    4854        22          16          -6               16          IavTopRib              
    6266        480         192         -288             2           LsoOflTnk              
    6164        16          33          17               16          OavTopRib              
    6180        49          98          49               2           OcrCalLso              
    6177        288         192         -96              2           OcrCalLsoPrt           
    6176        49          98          49               2           OcrGdsInLsoOfl         
    6174        49          98          49               2           OcrGdsLsoInOil         
    6171        288         192         -96              2           OcrGdsLsoPrt           
    4825        192         288         96               2           OcrGdsPrt              
    6175        98          49          -49              2           OcrGdsTfbInLsoOfl      
    6264        344         366         22               2           OflTnkContainer        
    6098        15          35          20               16          SstBotRib              
    6132        48          34          -14              16          SstTopCirRibBase       
    6124        192         96          -96              2           SstTopHub              
    6396        775         296         -479             6           amcco60-source-assy    
    6046        122         100         -22              12          headon-pmt-assy        
    6052        192         96          -96              12          headon-pmt-mount       
    6370        778         629         -149             6           led-source-assy        
    6371        342         50          -292             6           led-source-shell       
    4817        170         168         -2               2           lso                    
    12229       34          50          16               1           near-radslab-box-9     
    2           20          16          -4               1           near_hall_top_dwarf    
    3076        10          12          2                144         near_pentagon_iron_box 
    3148        50          34          -16              1           near_pool_dead_box     
    3149        34          50          16               1           near_pool_liner_box    
    3150        78          53          -25              1           near_pool_ows_box      
    3           34          40          6                1           near_top_cover_box     
    11408       360         362         2                672         pmt-hemi               
    11409       334         338         4                672         pmt-hemi-vac           
    6381        780         357         -423             6           source-assy            
    6382        342         50          -292             6           source-shell           
    6187        316         360         44               6           wall-led-assy          
    6425        342         50          -292             36          weight-shell           




Possible sources of vertex count differences
----------------------------------------------

#. pycollada post processing ? duplicate removal ?


"""

import os, sqlite3

if __name__ == '__main__':
    pass
    db = sqlite3.connect(":memory:")
    dbs =  { 
                'wrl':"$LOCAL_BASE/env/geant4/geometry/vrml2/g4_01.db", 
                'dae':"$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.db",
           }
    for tag, path in dbs.items():
        path = os.path.expandvars(path)
        db.execute('attach database "%(path)s" as %(tag)s' % locals() )

    for _ in db.execute("select count(*) from dae.geom "):print _
    for _ in db.execute("select count(*) from wrl.xshape "):print _
