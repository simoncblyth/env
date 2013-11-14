Geant4 Geometry Exports
===========================

Nov 14 2013, review
--------------------

Prior exports::

    [blyth@belle7 gdml]$ l $(local-base)/env/geant4/geometry/gdml/
    total 7204
    -rw-r--r-- 1 blyth blyth       0 Sep 23 18:03 g4_01.gdml.db
    -rw-rw-r-- 1 blyth blyth 4111332 Sep 23 13:18 g4_01.gdml
    -rw-rw-r-- 1 blyth blyth 3254604 Sep 20 20:16 g4_00.gdml

    [blyth@belle7 gdml]$ l $(local-base)/env/geant4/geometry/vrml2/
    total 1010348
    -rw-rw-r-- 1 blyth blyth  85360182 Oct  3 12:38 g4_00.wrl.noext
    -rw-rw-r-- 1 blyth blyth  83937916 Oct  3 12:06 g4_00.wrl.nosolid
    -rw-rw-r-- 1 blyth blyth   1336930 Sep 18 21:01 exportdbg.log
    -rw-rw-r-- 1 blyth blyth  85400082 Sep 18 21:00 g4_06.wrl
    -rw-rw-r-- 1 blyth blyth  85400082 Sep 18 20:32 g4_05.wrl
    -rw-rw-r-- 1 blyth blyth  85400082 Sep 18 20:06 g4_04.wrl
    -rw-rw-r-- 1 blyth blyth  85400082 Sep 18 19:43 g4_03.wrl
    -rw-r--r-- 1 blyth blyth 265057280 Sep 16 12:45 g4_01.db
    -rw-rw-r-- 1 blyth blyth  85399453 Sep 12 18:58 g4_01.wrl
    -rw-rw-r-- 1 blyth blyth  85400082 Sep  9 13:41 g4_02.wrl
    -rw-r--r-- 1 blyth blyth      3072 Sep  2 20:43 g4_00.db
    -rw-rw-r-- 1 blyth blyth  85399453 Sep  2 17:23 g4_00.wrl

    [blyth@belle7 gdml]$ l $(local-base)/env/geant4/geometry/xdae/
    total 5008
    -rw-r--r-- 1 blyth blyth       0 Nov 14 12:16 g4_01.db
    -rw-rw-r-- 1 blyth blyth 5115547 Oct 17 09:55 g4_01.dae

Newly minted::

    [blyth@belle7 gdml]$ ls -l g4_00.*
    -rw-rw-r-- 1 blyth blyth 5126579 Nov 14 18:52 g4_00.dae
    -rw-rw-r-- 1 blyth blyth 4111332 Nov 14 18:52 g4_00.gdml

Place the newly minted pair at slot 10, in ``gdml`` to reflect creation folder::

    [blyth@belle7 gdml]$ cp g4_00.gdml $(local-base)/env/geant4/geometry/gdml/g4_10.gdml
    [blyth@belle7 gdml]$ cp g4_00.dae  $(local-base)/env/geant4/geometry/gdml/g4_10.dae



