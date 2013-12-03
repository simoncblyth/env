Meshlab Plugins 
=================

* https://bitbucket.org/scb-/meshlab


Collada Import
----------------

Meshlab collada importer takes 40 minutes to import g4_00.dae, compared to 40s by pycollada, 
suspect horrendously inefficient XML handling is the culprit.

Despite collada import functionality coming mainly from VCGLIB, there is 
enough meshlab (eg the Meshmodel) in the importer to make development
within meshlab to be more appropriate that attempting to operate 
at vcglib level.


Separate io_collada meshlabplugin build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a separate collada plugin build in ``$ENV_HOME/graphics/meshlab/meshlabplugins/io_collada/``
by customizing the ``io_collada.pro`` to get sources from directories pointed to by newly minted envvars. 

#. ``MESHLAB_DIR``
#. ``MESHLAB_VCGLIB``

Also create separate ``meshlabserver`` main in ``$ENV_HOME/graphics/meshlab/meshlabserver/`` 
via another customized ``.pro``. Allowing testing of the the above plugin.


::

    simon:meshlab blyth$ pwd
    /Users/blyth/e/graphics/meshlab
    simon:meshlab blyth$ cp $(meshlab-dir)/meshlabplugins/io_collada/io_collada.pro meshlabplugins/io_collada/
    simon:meshlab blyth$ cp $(meshlab-dir)/meshlabplugins/io_collada/io_collada.cpp meshlabplugins/io_collada/
    simon:meshlab blyth$ cp $(meshlab-dir)/meshlabplugins/io_collada/io_collada.h meshlabplugins/io_collada/
    simon:io_collada blyth$ make distclean   # after customisaing the .pro to use difference source directories
    simon:io_collada blyth$ qmake
    simon:io_collada blyth$ make



Investigation environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:~ blyth$ meshlab-server-test
    meshlab-server-test is a function
    meshlab-server-test () 
    { 
        type $FUNCNAME;
        local app=$(meshlab-server-app);
        cd_func $app/Contents/MacOS;
        ./meshlabserver -i /usr/local/env/geant4/geometry/gdml/VDGX_20131121-1957/g4_00.dae
    }
    Input mesh  /usr/local/env/geant4/geometry/gdml/VDGX_20131121-1957/g4_00.dae
    Loading Plugins:
    The base dir is /Users/blyth/env/graphics/meshlab/distrib/meshlab.app
    The base dir is /Users/blyth/env/graphics/meshlab/distrib/meshlab.app
    The base dir is /Users/blyth/env/graphics/meshlab/distrib/meshlab.app
    Current Plugins Dir is: /Users/blyth/env/graphics/meshlab/distrib/meshlab.app/plugins 
    checking: /Users/blyth/env/graphics/meshlab/distrib/meshlab.app/plugins/libio_collada.dylib 
    Attempt pluginLoad: /Users/blyth/env/graphics/meshlab/distrib/meshlab.app/plugins/libio_collada.dylib 
    io pluginLoad: /Users/blyth/env/graphics/meshlab/distrib/meshlab.app/plugins/libio_collada.dylib 
    Total 0 filtering actions
    Total 1 io plugins
    Opening a file with extention dae
    Node 0 geom id = 'near_top_cover_box0xbd7b3d8'
    Node 1 geom id = 'RPCStrip0xb9da238'
    Node 2 geom id = 'RPCGasgap140xbc25fb0'
    Node 3 geom id = 'RPCBarCham140xbffbdc8'
    Node 4 geom id = 'RPCGasgap230xbd500e0'
    Node 5 geom id = 'RPCBarCham230xbd4f0e0'
    Node 6 geom id = 'RPCFoam0xbd60490'
    ...
    Node 245 geom id = 'near-radslab-box-90xc8e73c0'
    Node 246 geom id = 'near_hall_bot0xbc250a0'
    Node 247 geom id = 'near_rock0xb9fa8b8'
    Node 248 geom id = 'WorldBox0xc8e27e0'
    Time elapsed: 1869 ms
    ====== searching among library_effects the effect with id '__dd__Materials__Vacuum_fx_0xbd75258' 
    Parsing matrix node; text value is '-0.543174 -0.83962 0 -16520 0.83962 -0.543174 0 -802110 0 0 1 -2110 0.0 0.0 0.0 1.0'
    ====== searching among library_effects the effect with id '__dd__Materials__Rock_fx_0xb9fd460' 
    Parsing matrix node; text value is '1 0 0 2500 0 1 0 -500 0 0 1 7500 0.0 0.0 0.0 1.0'
    ====== searching among library_effects the effect with id '__dd__Materials__Air_fx_0xb9ff8d0' 
    Parsing matrix node; text value is '1 0 0 -2500 0 1 0 500 0 0 1 -7478 0.0 0.0 0.0 1.0'
    ====== searching among library_effects the effect with id '__dd__Materials__PPE_fx_0xbc22f80' 
    ====== searching among library_effects the effect with id '__dd__Materials__PPE_fx_0xbc22f80' 
    Parsing matrix node; text value is '0.99995 0.0100372 0 -2560.55 -0.0100372 0.99995 0 -5305.87 0 0 1 -4706.1 0.0 0.0 0.0 1.0'
    ====== searching among library_effects the effect with id '__dd__Materials__Aluminium_fx_0xbfcb718' 


Looking at collada load implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/usr/local/env/graphics/meshlab/vcglib/wrap/io_trimesh/import_dae.h::

     716         static int Open(OpenMeshType& m,const char* filename, InfoDAE& info, CallBackPos *cb=0)
     717         {
     718             (void)cb;
     719 
     720             QDEBUG("----- Starting the processing of %s ------",filename);
     721             //AdditionalInfoDAE& inf = new AdditionalInfoDAE();
     722             //info = new InfoDAE();
     723 





