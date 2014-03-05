Create Incremental Patch
=========================

Geant4 mods necessary for G4DAE were made against the 
pre-patched NuWa geant4 which already was patched with::

   geant4.9.2.p01.patch
   geant4.9.2.p01.patch2

Naming the incremental patch after this base: `geant4.9.2.p01.patch.patch2`, arrange
a diff between the standard nuwa-geant4 patches in `dyb` and the customized in `dyb.old`::

    [blyth@belle7 ~]$ mkdir g4makepatch
    [blyth@belle7 ~]$ cd g4makepatch/
    [blyth@belle7 g4makepatch]$ ll
    total 24
    lrwxrwxrwx  1 blyth blyth    54 Mar  5 17:04 geant4.9.2.p01.patch.patch2.orig -> /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01
    lrwxrwxrwx  1 blyth blyth    58 Mar  5 17:19 geant4.9.2.p01.patch.patch2 -> /data1/env/local/dyb.old/external/build/LCG/geant4.9.2.p01
    drwxr-xr-x 50 blyth blyth 12288 Mar  5 17:40 ..
    drwxrwxr-x  2 blyth blyth  4096 Mar  5 17:40 patches
    drwxrwxr-x  3 blyth blyth  4096 Mar  5 17:40 .
    [blyth@belle7 g4makepatch]$ rm -rf patches
    [blyth@belle7 g4makepatch]$ diff.py geant4.9.2.p01.patch.patch2.orig  geant4.9.2.p01.patch.patch2 -p
    2014-03-05 18:22:39,913 env.base.diff INFO     /home/blyth/env/bin/diff.py geant4.9.2.p01.patch.patch2.orig geant4.9.2.p01.patch.patch2 -p
    2014-03-05 18:22:39,913 env.base.diff INFO     a [32]geant4.9.2.p01.patch.patch2.orig b [27]geant4.9.2.p01.patch.patch2 
    2014-03-05 18:22:39,913 env.base.diff INFO     diff name geant4.9.2.p01.patch.patch2 
    diff
          ('source/visualization/VRML/GNUmakefile', 'geant4.9.2.p01.patch.patch2.orig/source/visualization/VRML/GNUmakefile', 'geant4.9.2.p01.patch.patch2/source/visualization/VRML/GNUmakefile')
          ('source/visualization/VRML/src/G4VRML2FileSceneHandler.cc', 'geant4.9.2.p01.patch.patch2.orig/source/visualization/VRML/src/G4VRML2FileSceneHandler.cc', 'geant4.9.2.p01.patch.patch2/source/visualization/VRML/src/G4VRML2FileSceneHandler.cc')
          ('source/visualization/VRML/include/G4VRML2FileSceneHandler.hh', 'geant4.9.2.p01.patch.patch2.orig/source/visualization/VRML/include/G4VRML2FileSceneHandler.hh', 'geant4.9.2.p01.patch.patch2/source/visualization/VRML/include/G4VRML2FileSceneHandler.hh')
          ('source/materials/src/G4MaterialPropertyVector.cc', 'geant4.9.2.p01.patch.patch2.orig/source/materials/src/G4MaterialPropertyVector.cc', 'geant4.9.2.p01.patch.patch2/source/materials/src/G4MaterialPropertyVector.cc')
          ('source/materials/include/G4MaterialPropertiesTable.hh', 'geant4.9.2.p01.patch.patch2.orig/source/materials/include/G4MaterialPropertiesTable.hh', 'geant4.9.2.p01.patch.patch2/source/materials/include/G4MaterialPropertiesTable.hh')
          ('source/materials/include/G4MaterialPropertyVector.hh', 'geant4.9.2.p01.patch.patch2.orig/source/materials/include/G4MaterialPropertyVector.hh', 'geant4.9.2.p01.patch.patch2/source/materials/include/G4MaterialPropertyVector.hh')
          ('source/persistency/gdml/src/G4GDMLWrite.cc', 'geant4.9.2.p01.patch.patch2.orig/source/persistency/gdml/src/G4GDMLWrite.cc', 'geant4.9.2.p01.patch.patch2/source/persistency/gdml/src/G4GDMLWrite.cc')
          ('source/persistency/gdml/include/G4GDMLWrite.hh', 'geant4.9.2.p01.patch.patch2.orig/source/persistency/gdml/include/G4GDMLWrite.hh', 'geant4.9.2.p01.patch.patch2/source/persistency/gdml/include/G4GDMLWrite.hh')
    right
          ('geant4.9.2.p01.patch.patch2/.geant4.9.2.p01.patch3',)
          ('geant4.9.2.p01.patch.patch2/environments/g4py/lib',)
          ('geant4.9.2.p01.patch.patch2/environments/g4py/config.status',)
          ('geant4.9.2.p01.patch.patch2/environments/g4py/config/config.gmk',)
          ('geant4.9.2.p01.patch.patch2/source/materials/src/G4MaterialPropertyVector.cc.orig',)
          ('geant4.9.2.p01.patch.patch2/source/materials/include/G4MaterialPropertiesTable.hh.orig',)
          ('geant4.9.2.p01.patch.patch2/source/materials/include/G4MaterialPropertyVector.hh.orig',)
    left
    2014-03-05 18:22:43,477 env.base.diff INFO     creating patchdir patches 
    [blyth@belle7 g4makepatch]$ 


Need another patch to hookup the dae into the Geant4 build::

    [blyth@belle7 g4makepatch]$ diff.py geant4.9.2.p01.patch.patch2.orig  geant4.9.2.p01.patch.patch2 -p
    2014-03-05 19:29:24,316 env.base.diff INFO     /home/blyth/env/bin/diff.py geant4.9.2.p01.patch.patch2.orig geant4.9.2.p01.patch.patch2 -p
    2014-03-05 19:29:24,316 env.base.diff INFO     a [32]geant4.9.2.p01.patch.patch2.orig b [27]geant4.9.2.p01.patch.patch2 
    2014-03-05 19:29:24,316 env.base.diff INFO     diff name geant4.9.2.p01.patch.patch2 
    diff
          ('source/persistency/GNUmakefile', 'geant4.9.2.p01.patch.patch2.orig/source/persistency/GNUmakefile', 'geant4.9.2.p01.patch.patch2/source/persistency/GNUmakefile')
    right
          ('geant4.9.2.p01.patch.patch2/environments/g4py/lib',)
          ('geant4.9.2.p01.patch.patch2/environments/g4py/config.status',)
          ('geant4.9.2.p01.patch.patch2/environments/g4py/config/config.gmk',)
          ('geant4.9.2.p01.patch.patch2/source/materials/src/G4MaterialPropertyVector.cc.orig',)
          ('geant4.9.2.p01.patch.patch2/source/materials/include/G4MaterialPropertiesTable.hh.orig',)
          ('geant4.9.2.p01.patch.patch2/source/materials/include/G4MaterialPropertyVector.hh.orig',)
    left
          ('geant4.9.2.p01.patch.patch2.orig/.geant4.9.2.p01.patch.patch2_source_materials_src_G4MaterialPropertyVector.cc.patch',)
          ('geant4.9.2.p01.patch.patch2.orig/.geant4.9.2.p01.patch.patch2_source_materials_include_G4MaterialPropertyVector.hh.patch',)
          ('geant4.9.2.p01.patch.patch2.orig/.geant4.9.2.p01.patch.patch2_source_visualization_VRML_src_G4VRML2FileSceneHandler.cc.patch',)
          ('geant4.9.2.p01.patch.patch2.orig/.geant4.9.2.p01.patch.patch2_source_persistency_gdml_include_G4GDMLWrite.hh.patch',)
          ('geant4.9.2.p01.patch.patch2.orig/.geant4.9.2.p01.patch.patch2_source_persistency_gdml_src_G4GDMLWrite.cc.patch',)
          ('geant4.9.2.p01.patch.patch2.orig/.geant4.9.2.p01.patch.patch2_source_materials_include_G4MaterialPropertiesTable.hh.patch',)
          ('geant4.9.2.p01.patch.patch2.orig/source/persistency/dae',)
    2014-03-05 19:29:27,873 env.base.diff INFO     creating patchdir patches 
    [blyth@belle7 g4makepatch]$ 




Changes
~~~~~~~~~~

* environments_g4py, unrelated investigation to drop
* source_materials, expose property maps and vector content (fixed in later geant4)
* source_persistency_gdml, increase tempStr buffer size (fixed in later geant4) 
* source_visualization_VRML, necessary precision fix (fixed in later geant4) + commented G4DAE_DEBUG to drop


VRML
-----

* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/4140/trunk/geant4/geometry/VRML/src/G4VRML2SceneHandlerFunc.icc
* http://aralbalkan.com/1381/


::

    [blyth@belle7 src]$ pwd 
    /home/blyth/e/geant4/geometry/VRML/src

     
::

    [blyth@belle7 src]$ svn diff -r4290:4139 G4VRML2SceneHandlerFunc.icc
    svn: The location for 'G4VRML2SceneHandlerFunc.icc' for revision 4139 does not exist in the repository or refers to an unrelated object

    [blyth@belle7 src]$ svn diff -r4290:4140 G4VRML2SceneHandlerFunc.icc
    Index: G4VRML2SceneHandlerFunc.icc
    ===================================================================
    --- G4VRML2SceneHandlerFunc.icc (revision 4290)
    +++ G4VRML2SceneHandlerFunc.icc (revision 4140)
    @@ -189,17 +189,6 @@
            //std::cerr << "SCB " << pv_name << "\n";
            fDest << "#---------- SOLID: " << pv_name << "\n";
     
    -
    -/*
    -#if defined G4DAE_DEBUG
    -    std::stringstream ss ; 
    -    ss << "n " <<  pv_name << " " ; 
    -    ss << "v " <<  polyhedron.GetNoVertices() << " " ; 
    -    ss << "f " <<  polyhedron.GetNoFacets() << " " ; 
    -    fSummary.push_back(ss.str()); 
    -#endif
    -*/
    -
            if ( IsPVPickable() ) {
     
             fDest << "Anchor {" << "\n";
    [blyth@belle7 src]$ 


    [blyth@belle7 src]$ svn merge -r4290:4140 G4VRML2SceneHandlerFunc.icc
    --- Reverse-merging r4290 through r4141 into 'G4VRML2SceneHandlerFunc.icc':
    U    G4VRML2SceneHandlerFunc.icc

    [blyth@belle7 src]$ st 
    M       G4VRML2SceneHandlerFunc.icc

    [blyth@belle7 src]$ svn diff
    Index: G4VRML2SceneHandlerFunc.icc
    ===================================================================
    --- G4VRML2SceneHandlerFunc.icc (revision 4290)
    +++ G4VRML2SceneHandlerFunc.icc (working copy)
    @@ -189,17 +189,6 @@
            //std::cerr << "SCB " << pv_name << "\n";
            fDest << "#---------- SOLID: " << pv_name << "\n";
     
    -
    -/*
    -#if defined G4DAE_DEBUG
    -    std::stringstream ss ; 
    -    ss << "n " <<  pv_name << " " ; 
    -    ss << "v " <<  polyhedron.GetNoVertices() << " " ; 
    -    ss << "f " <<  polyhedron.GetNoFacets() << " " ; 
    -    fSummary.push_back(ss.str()); 
    -#endif
    -*/
    -
            if ( IsPVPickable() ) {
     
             fDest << "Anchor {" << "\n";
    [blyth@belle7 src]$ 




