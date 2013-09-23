
.. _gdml_root:

GDML from ROOT
================

Not enabled by default
--------------------------

::

    root [1] TGeoManager::Import("/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml")
    Info in <TGeoManager::Import>: Reading geometry from file: /data1/env/local/env/geant4/geometry/gdml/g4_01.gdml
    Error: Function StartGDML("/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml") is not defined in current scope  (tmpfile):1:
    *** Interpreter error recovered ***
    Error in <TGeoManager::Import>: Cannot open file
    (class TGeoManager*)0x0
    root [2] 


rebuild root
-------------

The dybinst externals order has ROOT before geant4, so cannot depend on 
G4 includes for the standard ROOT build. 
BUT checking `geom/gdml/inc/TGDMLParse.h` seems to be no G4 dependency, thats good.

Check for anything non-standard with the ROOT build::

    [blyth@belle7 ROOT]$ cd $DYB/NuWa-trunk/lcgcmt/LCG_Builders/ROOT   
    [blyth@belle7 ROOT]$ grep do_configure scripts/ROOT_config.sh
    do_configure ${LCG_ROOT_CONFIG_ARCH} ${LCG_ROOT_CONFIG_OPTIONS}

    [blyth@belle7 ROOT]$ cd $DYB/NuWa-trunk/lcgcmt/LCG_Builders/geant4

A lot, but not GDML::

    cd $DYB/external/build/LCG/root
    ./configure ${LCG_ROOT_CONFIG_ARCH} ${LCG_ROOT_CONFIG_OPTIONS} --enable-gdml

::

    [blyth@belle7 root]$ vi geom/gdml/inc/TGDMLParse.h 
    [blyth@belle7 root]$ make install
    Installing binaries in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/bin
    Installing libraries in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/lib
    Installing headers in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/include
    Installing main/src/rmain.cxx in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/include
    Installing cint/cint/include cint/cint/lib and cint/cint/stl in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/cint
    Installing icons in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/icons
    Installing fonts in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/fonts
    Installing misc docs in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root
    Installing tutorials in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/tutorials
    Installing tests in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/test
    Installing macros in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/macros
    Installing man(1) pages in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/man/man1
    Installing config files in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/etc
    Installing Autoconf macro in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/build/misc
    Installing Emacs Lisp library in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/build/misc
    Installing GDML conversion scripts in /data1/env/local/dyb/NuWa-trunk/../external/ROOT/5.26.00e/i686-slc5-gcc41-dbg/root/lib
    [blyth@belle7 root]$ 


test GDML load into ROOT
------------------------

::

    ROOT 5.26/00e (tags/v5-26-00e@36332, Oct 13 2010, 12:40:14 on linux)

    CINT/ROOT C/C++ Interpreter version 5.17.00, Dec 21, 2008
    Type ? for help. Commands must be C++ statements.
    Enclose multiple statements between { }.
    root [0] 
    root [0]  TGeoManager::Import("/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml")
    Info in <TGeoManager::Import>: Reading geometry from file: /data1/env/local/env/geant4/geometry/gdml/g4_01.gdml
    Info in <TGeoManager::TGeoManager>: Geometry Geometry, default geometry created
    Error: Unsupported GDML Tag Used :isotope. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :atom. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :fraction. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :isotope. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :atom. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :fraction. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :isotope. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :atom. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :fraction. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :isotope. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :atom. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :fraction. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :isotope. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :atom. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :fraction. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :isotope. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :atom. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :fraction. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :isotope. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :atom. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :fraction. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :isotope. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :atom. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :fraction. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :isotope. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :atom. Please Check Geometry/Schema.
    Error: Unsupported GDML Tag Used :fraction. Please Check Geometry/Schema.
    Info in <TGeoManager::SetTopVolume>: Top volume is World. Master volume is World
    Info in <TGeoManager::CheckGeometry>: Fixing runtime shapes...
    Info in <TGeoManager::CheckGeometry>: ...Nothing to fix
    Info in <TGeoManager::CloseGeometry>: Counting nodes...
    Info in <TGeoManager::Voxelize>: Voxelizing...
    Info in <TGeoManager::CloseGeometry>: Building cache...
    Info in <TGeoNavigator::BuildCache>: --- Maximum geometry depth set to 100
    Info in <TGeoManager::CloseGeometry>: 12230 nodes/ 249 volume UID's in default geometry
    Info in <TGeoManager::CloseGeometry>: ----------------modeler ready----------------
    (class TGeoManager*)0x89be148




ROOT GL Viewer
------------------

::

    [blyth@belle7 graf3d]$ vi $DYB/external/build/LCG/root/graf3d/gl/src/TGLSAViewer.cxx


::

    DIRECT SCENE INTERACTIONS

    Press:

    `w`  
          wireframe mode
    `e`   
          switch between dark/light color-set
    `r`   
          filled polygons mode
    `t`   
          outline mode
    `j`   
          ZOOM in
    `k`  
          ZOOM out
    Arrow Keys 
          PAN (TRUCK) across scene
    Home   
          reset current camera

    LEFT mouse button -- ROTATE (ORBIT) the scene by holding the mouse button and moving
    the mouse (perspective camera, needs to be enabled in menu for orthograpic cameras).
    By default, the scene will be rotated about its center. To select arbitrary center
    bring up the viewer-editor (e.g., shift-click into empty background) and use
    'Camera center' controls in the 'Guides' tab.

    MIDDLE mouse button or arrow keys --  PAN (TRUCK) the camera.

    RIGHT mouse button action depends on camera type:
         orthographic -- zoom,
         perspective  -- move camera forwards / backwards

    By pressing Ctrl and Shift keys the mouse precision can be changed:
         Shift      -- 10 times less precise
         Ctrl       -- 10 times more precise
         Ctrl Shift -- 100 times more precise

    Mouse wheel action depends on camera type:
         orthographic -- zoom,
         perspective  -- change field-of-view (focal length)

    Double clik will show GUI editor of the viewer (if assigned).

    RESET the camera via the button in viewer-editor or Home key.

    SELECT a shape with Shift+Left mouse button click.

    SELECT the viewer with Shift+Left mouse button click on a free space.

    MOVE a selected shape using Shift+Mid mouse drag.

    Invoke the CONTEXT menu with Shift+Right mouse click.n"
    "Secondary selection and direct render object interaction is initiated
    by Alt+Left mouse click (Mod1, actually). Only few classes support this option.
    When 'Alt' is taken by window manager, try Alt-Ctrl-Left.

    CAMERA

    The "Camera" menu is used to select the different projections from 
    the 3D world onto the 2D viewport. There are three perspective cameras:

    * Perspective (Floor XOZ)
    * Perspective (Floor YOZ)
    * Perspective (Floor XOY)

    In each case the floor plane (defined by two axes) is kept level.




ROOT GDML DOCS
---------------

* `$DYB/external/build/LCG/root/geom/gdml/README`

The GDML to TGeo converter uses the TXMLEngine to parse 
the GDML files. This XML parser is a DOM parser and returns the DOM 
tree to the class TGDMLParse.  This class then interprets the GDML file 
and adds the bindings in their TGeo equivilent.

* `$DYB/external/build/LCG/root/geom/gdml/inc/TGDMLParse.h`
* `$DYB/external/build/LCG/root/geom/gdml/src/TGDMLParse.cxx`



