Export All
============

.. contents:: :local:

VRML2 sensitivity
--------------------

Using ``export_all.py`` which uses ``GiGaRunActionGDML`` (possibly to be renamed ``GiGaRunActionExport`` ) 
find vertex count and precision differences in the WRL between
exporting singly and when exported after DAE+GDML in the same process ?

Maybe:

#. missing vis initialistion
#. ordering effect ? is the DAE+GDML dump changing the geometry OR some parameters relevant to GetPolyhedron ? 

::

    [blyth@belle7 gdml]$ ll g4_00.* tmp/g4_00.*
    -rw-rw-r-- 1 blyth blyth  5126579 Nov 15 12:24 tmp/g4_00.dae
    -rw-rw-r-- 1 blyth blyth  4111332 Nov 15 12:24 tmp/g4_00.gdml
    -rw-rw-r-- 1 blyth blyth 85400082 Nov 15 13:06 tmp/g4_00.wrl
    -rw-rw-r-- 1 blyth blyth  4111332 Nov 15 14:38 g4_00.gdml
    -rw-rw-r-- 1 blyth blyth  5126579 Nov 15 14:38 g4_00.dae
    -rw-rw-r-- 1 blyth blyth 86458076 Nov 15 14:38 g4_00.wrl
    -rw-rw-r-- 1 blyth blyth   217259 Nov 15 14:53 tmp/g4_00.wrl.10k
    -rw-rw-r-- 1 blyth blyth   217194 Nov 15 14:53 g4_00.wrl.10k

    [blyth@belle7 gdml]$ du -hs g4_00.* tmp/g4_00.*
    5.0M    g4_00.dae
    4.0M    g4_00.gdml
    83M     g4_00.wrl
    220K    g4_00.wrl.10k
    5.0M    tmp/g4_00.dae
    4.0M    tmp/g4_00.gdml
    82M     tmp/g4_00.wrl
    220K    tmp/g4_00.wrl.10k

::

    [blyth@belle7 gdml]$ head -10000 tmp/g4_00.wrl > tmp/g4_00.wrl.10k 
    [blyth@belle7 gdml]$ head -10000 g4_00.wrl > g4_00.wrl.10k 
    [blyth@belle7 gdml]$ diff tmp/g4_00.wrl.10k  g4_00.wrl.10k 


Question

#. does the changed WRL better match the DAE, in vertex counts/offsets ?


Order dependance : interference between DAE and WRL exports
-------------------------------------------------------------

#. no effect on GDML, same no matter what the order
#. wrl alone matches wrl first 
#. dae alone matches gdml+dae first
#. looks like interference both ways, between dae and wrl 

   * dae exported after wrl is smaller than dae alone
   * wrl exported after dae is larger than wrl alone 

::

    [blyth@belle7 gdml]$ l wrl_gdml_dae/    # single process : wrl followed by gdml+dae
    total 92260
    -rw-rw-r-- 1 blyth blyth 85400082 Nov 15 17:24 g4_00.wrl
    -rw-rw-r-- 1 blyth blyth  4852486 Nov 15 17:24 g4_00.dae
    -rw-rw-r-- 1 blyth blyth  4111332 Nov 15 17:24 g4_00.gdml

    [blyth@belle7 gdml]$ l gdml_dae_wrl/    # single process : with gdml+dae followed by wrl 
    total 93780
    -rw-rw-r-- 1 blyth blyth  4111332 Nov 15 14:38 g4_00.gdml
    -rw-rw-r-- 1 blyth blyth  5126579 Nov 15 14:38 g4_00.dae
    -rw-rw-r-- 1 blyth blyth 86458076 Nov 15 14:38 g4_00.wrl

    [blyth@belle7 gdml]$ l tmp/             # wrl output separate from gdml+dae
    total 92748
    -rw-rw-r-- 1 blyth blyth 85400082 Nov 15 13:06 g4_00.wrl

    -rw-rw-r-- 1 blyth blyth  4111332 Nov 15 12:24 g4_00.gdml
    -rw-rw-r-- 1 blyth blyth  5126579 Nov 15 12:24 g4_00.dae


Comparing the interfered
------------------------------

::

    [blyth@belle7 gdml]$ daedb.py --daepath wrl_gdml_dae/g4_00.dae
    [blyth@belle7 gdml]$ daedb.py --daepath gdml_dae_wrl/g4_00.dae
    [blyth@belle7 gdml]$ vrml2file.py -cx gdml_dae_wrl/g4_00.wrl     ## this is taking >20 min ??? 
    

