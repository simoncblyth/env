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
    

DB extend takinh ages and filling disk
----------------------------------------

Argh its filling the disk::

    simon:gdml_dae_wrl blyth$ du -hs *
    4.9M    g4_00.dae
    2.4M    g4_00.dae.db
    3.9M    g4_00.gdml
     82M    g4_00.wrl
    253M    g4_00.wrl.db
    simon:gdml_dae_wrl blyth$ rm g4_00.wrl.db


Add Sequence flexibility
--------------------------

Using sequence with WRL first::

   G4DAE_EXPORT_SEQUENCE VVVDDDGGGVVVVDVDVD

Suggests that what matters is who exports first (perhaps creating Polyhedron that are then reused perhaps)::

    [blyth@belle7 gdml]$ cd  /data1/env/local/env/geant4/geometry/gdml/20131116-1645/ 
    [blyth@belle7 20131116-1645]$ l *.wrl
    -rw-rw-r-- 1 blyth blyth 85400082 Nov 16 16:49 g4_08.wrl
    -rw-rw-r-- 1 blyth blyth 85400082 Nov 16 16:48 g4_07.wrl
    -rw-rw-r-- 1 blyth blyth 85400082 Nov 16 16:48 g4_06.wrl
    -rw-rw-r-- 1 blyth blyth 85400082 Nov 16 16:48 g4_05.wrl
    -rw-rw-r-- 1 blyth blyth 85400082 Nov 16 16:48 g4_04.wrl
    -rw-rw-r-- 1 blyth blyth 85400082 Nov 16 16:48 g4_03.wrl
    -rw-rw-r-- 1 blyth blyth 85400082 Nov 16 16:48 g4_02.wrl
    -rw-rw-r-- 1 blyth blyth 85400082 Nov 16 16:47 g4_01.wrl
    -rw-rw-r-- 1 blyth blyth 85400082 Nov 16 16:47 g4_00.wrl
    [blyth@belle7 20131116-1645]$ 
    [blyth@belle7 20131116-1645]$ l *.dae
    -rw-rw-r-- 1 blyth blyth 4852486 Nov 16 16:49 g4_05.dae
    -rw-rw-r-- 1 blyth blyth 4852486 Nov 16 16:48 g4_04.dae
    -rw-rw-r-- 1 blyth blyth 4852486 Nov 16 16:48 g4_03.dae
    -rw-rw-r-- 1 blyth blyth 4852486 Nov 16 16:48 g4_02.dae
    -rw-rw-r-- 1 blyth blyth 4852486 Nov 16 16:48 g4_01.dae
    -rw-rw-r-- 1 blyth blyth 4852486 Nov 16 16:48 g4_00.dae
    [blyth@belle7 20131116-1645]$ 
    [blyth@belle7 20131116-1645]$ l *.gdml
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 16 16:48 g4_02.gdml
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 16 16:48 g4_01.gdml
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 16 16:48 g4_00.gdml


Changing sequence to  DAE first ``DVVVDDDGGGVVVVDVDVD`` confirms this, 
the first export influences all::

    [blyth@belle7 20131116-1659]$ l *.gdml
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 16 17:02 g4_02.gdml
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 16 17:02 g4_01.gdml
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 16 17:02 g4_00.gdml
    [blyth@belle7 20131116-1659]$ l *.dae 
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 16 17:03 g4_06.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 16 17:02 g4_05.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 16 17:02 g4_04.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 16 17:02 g4_03.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 16 17:02 g4_02.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 16 17:02 g4_01.dae
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 16 17:01 g4_00.dae
    [blyth@belle7 20131116-1659]$ l *.wrl
    -rw-rw-r-- 1 blyth blyth 86458076 Nov 16 17:03 g4_08.wrl
    -rw-rw-r-- 1 blyth blyth 86458076 Nov 16 17:02 g4_07.wrl
    -rw-rw-r-- 1 blyth blyth 86458076 Nov 16 17:02 g4_06.wrl
    -rw-rw-r-- 1 blyth blyth 86458076 Nov 16 17:02 g4_05.wrl
    -rw-rw-r-- 1 blyth blyth 86458076 Nov 16 17:02 g4_04.wrl
    -rw-rw-r-- 1 blyth blyth 86458076 Nov 16 17:02 g4_03.wrl
    -rw-rw-r-- 1 blyth blyth 86458076 Nov 16 17:02 g4_02.wrl
    -rw-rw-r-- 1 blyth blyth 86458076 Nov 16 17:01 g4_01.wrl
    -rw-rw-r-- 1 blyth blyth 86458076 Nov 16 17:01 g4_00.wrl




