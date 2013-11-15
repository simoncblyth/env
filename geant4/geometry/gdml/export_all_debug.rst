Export All
============

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




