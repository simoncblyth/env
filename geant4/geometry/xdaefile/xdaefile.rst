XDAEFILE
==========

.. contents:: :local:

Compare Original with DAE WRL output 
--------------------------------------

Original
       G4 geometry exported with VRML2FILE into WRL
DAE WRL
       G4 geometry exported as a GDML file 
       GDML file parsed into a G4 geomety
       G4 geometry extracted with the VRML2FILE functionality copied over under DAE name  

::

      [blyth@belle7 xdae]$ wc  $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl g4_00.wrl
      3272951  12046872  85399453 /data1/env/local/env/geant4/geometry/vrml2/g4_00.wrl
      3272951  12046872  85384624 g4_00.wrl
      6545902  24093744 170784077 total


SOLID names
~~~~~~~~~~~~~

Tag extension integers differ.

::

    [blyth@belle7 xdae]$ diff $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl g4_00.wrl | head -10
    10c10
    < #---------- SOLID: /dd/Structure/Sites/db-rock.1000
    ---
    > #---------- SOLID: /dd/Structure/Sites/db-rock.0
    47c47
    < #---------- SOLID: /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000
    ---
    > #---------- SOLID: /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.0
    93c93
    < #---------- SOLID: /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover.1000
    [blyth@belle7 xdae]$ diff $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl g4_00.wrl | tail -10
    ---
    > #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab7.0
    3272755c3272755
    < #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab8.1008
    ---
    > #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab8.0
    3272787c3272787
    < #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab9.1009
    ---
    > #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab9.0
    [blyth@belle7 xdae]$ 

All coming out 0::

    [blyth@belle7 xdae]$ grep SOLID: g4_00.wrl | cut -d" " -f3 | cut -d"." -f2 | sort | uniq 
    0

Whereas original gives 269 different tag extension integers::

    [blyth@belle7 xdae]$ grep SOLID: $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl  | cut -d" " -f3 | cut -d"." -f2 | sort | uniq 
    1
    10
    1000
    1001
    1002
    1003
    ...

    [blyth@belle7 xdae]$ grep SOLID: $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl  | cut -d" " -f3 | cut -d"." -f2 | sort | uniq | wc -l 
    269


208 z-shifts of +0.1 mm 
~~~~~~~~~~~~~~~~~~~~~~~~~

All(?) solid names differ, so get rid of those first.

::

    [blyth@belle7 xdae]$ cat $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl | grep -v SOLID > $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl.nosolid 
    [blyth@belle7 xdae]$ cat g4_00.wrl | grep -v SOLID > g4_00.wrl.nosolid 
    [blyth@belle7 xdae]$ wc -l $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl.nosolid   g4_00.wrl.nosolid
      3260722 /data1/env/local/env/geant4/geometry/vrml2/g4_00.wrl.nosolid
      3260722 g4_00.wrl.nosolid
      6521444 total
    [blyth@belle7 xdae]$ diff $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl.nosolid   g4_00.wrl.nosolid | wc -l
    832

::

    In [1]: 832/4   # 4 lines are output per difference
    Out[1]: 208


::

    [blyth@belle7 xdae]$ diff $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl.nosolid   g4_00.wrl.nosolid | head -20 
    2524482c2524482
    <                                       -22589.1 -801140 -11589.8,
    ---
    >                                       -22589.1 -801140 -11589.9,
    2524492c2524492
    <                                       -22589.1 -801140 -11524.3,
    ---
    >                                       -22589.1 -801140 -11524.4,
    2524494c2524494
    <                                       -22579.6 -801155 -11524.3,
    ---
    >                                       -22579.6 -801155 -11524.4,
    2524504c2524504
    <                                       -22579.6 -801155 -11589.8,
    ---
    >                                       -22579.6 -801155 -11589.9,
    2524509c2524509
    <                                       -22626.2 -801107 -11589.8,
    ---
    >                                       -22626.2 -801107 -11589.9,
    [blyth@belle7 xdae]$ 
    [blyth@belle7 xdae]$ diff $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl.nosolid   g4_00.wrl.nosolid | tail -20
    2942657c2942657
    <                                       -22016.2 -798045 -10571.4,
    ---
    >                                       -22016.2 -798045 -10571.5,
    2942665c2942665
    <                                       -21863.9 -798126 -10571.4,
    ---
    >                                       -21863.9 -798126 -10571.5,
    2942669c2942669
    <                                       -21863.9 -798126 -10543.9,
    ---
    >                                       -21863.9 -798126 -10544,
    2942677c2942677
    <                                       -21853.9 -798080 -10543.9,
    ---
    >                                       -21853.9 -798080 -10544,
    2942681c2942681
    <                                       -21853.9 -798080 -10571.4,
    ---
    >                                       -21853.9 -798080 -10571.5,


26 Solids with +Z 0.1 mm shifts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Are the differences clustered ?  YES by inspection of the line numbers of the diff observer 26 solids afflicted.

::

    [blyth@belle7 xdae]$ cat $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl | perl -p -e 's,(#---------- SOLID: \S*)\.\d*,$1,' - > $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl.noext
    [blyth@belle7 xdae]$ cat g4_00.wrl | perl -p -e 's,(#---------- SOLID: \S*)\.\d*,$1,' - > g4_00.wrl.noext
    [blyth@belle7 xdae]$ diff $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl.noext  g4_00.wrl.noext | wc -l
    832
    [blyth@belle7 xdae]$ diff $LOCAL_BASE/env/geant4/geometry/vrml2/g4_00.wrl.noext  g4_00.wrl.noext > pdif.txt





