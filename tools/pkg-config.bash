pkg-config-source(){   echo ${BASH_SOURCE} ; }
pkg-config-edir(){ echo $(dirname $(pkg-config-source)) ; }
pkg-config-ecd(){  cd $(pkg-config-edir); }
pkg-config-dir(){  echo $LOCAL_BASE/env/tools/pkg-config ; }
pkg-config-cd(){   cd $(pkg-config-dir); }
pkg-config-vi(){   vi $(pkg-config-source) ; }
pkg-config-env(){  elocal- ; }
pkg-config-usage(){ cat << EOU

pkg-config
-----------

::

    epsilon:~ blyth$ pkg-config --variable pc_path pkg-config
    /opt/local/lib/pkgconfig:/opt/local/share/pkgconfig


* https://www.freedesktop.org/wiki/Software/pkg-config/

* https://people.freedesktop.org/~dbn/pkg-config-guide.html

My library z uses libx internally, but does not expose libx data types in its public API. 
What do I put in my z.pc file?

Again, add the module to Requires.private if it supports pkg-config. In this
case, the compiler flags will be emitted unnecessarily, but it ensures that the
linker flags will be present when linking statically. If libx does not support
pkg-config, add the necessary linker flags to Libs.private.


testing relocatability

::

    epsilon:opticks blyth$ PKG_CONFIG_PATH=/tmp/dummy/lib/pkgconfig:/tmp/dummy/externals/lib/pkgconfig pkg-config sysrap --cflags --define-prefix
    -DOPTICKS_SYSRAP -DOPTICKS_OKCONF -I/tmp/dummy/include/SysRap -I/tmp/dummy/include/OKConf -I/tmp/dummy/externals/plog/include



Hmm split pkgconfig dirs at different depths is problematic with --define-prefix::

    epsilon:opticks blyth$ oc-pkg-config --cflags yoctogl 
    -I/usr/local/opticks/externals/include/YoctoGL

    epsilon:opticks blyth$ oc-pkg-config --cflags yoctogl --define-prefix
    -I/usr/local/opticks/externals/externals/include/YoctoGL


Try a tricky solution::

    ln -s externals/lib xlib







::

    epsilon:opticks blyth$ find /usr/local/opticks -name '*.pc' | wc -l
    67

    epsilon:opticks blyth$ PKG_CONFIG_PATH=/usr/local/opticks/lib/pkgconfig pkg-config --libs --cflags sysrap
    -DOPTICKS_SYSRAP -DOPTICKS_OKCONF -I/usr/local/opticks/include/SysRap -I/usr/local/opticks/include/OKConf -L/usr/local/opticks/lib -lSysRap

    epsilon:opticks blyth$ PKG_CONFIG_PATH=/usr/local/opticks/lib/pkgconfig pkg-config --libs --cflags SysRap
    -DOPTICKS_SYSRAP -DOPTICKS_OKCONF -I/usr/local/opticks/include/SysRap -I/usr/local/opticks/include/OKConf -L/usr/local/opticks/lib -lSysRap

    epsilon:opticks blyth$ PKG_CONFIG_PATH=/usr/local/opticks/lib/pkgconfig:/usr/local/opticks/externals/lib/pkgconfig pkg-config --libs --cflags assimp
    -I/usr/local/opticks/externals/include/assimp -L/usr/local/opticks/externals/lib -lassimp

    epsilon:opticks blyth$ export PKG_CONFIG_PATH=/usr/local/opticks/lib/pkgconfig:/usr/local/opticks/externals/lib/pkgconfig
    epsilon:opticks blyth$ pkg-config --libs --cflags assimp
    -I/usr/local/opticks/externals/include/assimp -L/usr/local/opticks/externals/lib -lassimp

    

    epsilon:boostrap blyth$ PKG_CONFIG_LIBDIR=/dev/null PKG_CONFIG_PATH=/usr/local/opticks/lib/pkgconfig:/usr/local/opticks/externals/lib/pkgconfig pkg-config --list-all
    opticksgeo           opticksgeo - No description
    npy                  npy - No description
    ggeo                 ggeo - No description
    cudarap              cudarap - No description
    useinstance          useinstance - No description
    openmeshrap          openmeshrap - No description
    extg4                extg4 - No description
    oglrap               oglrap - No description
    optickscore          optickscore - No description
    x4gen                x4gen - No description
    sysrap               sysrap - No description
    usecuda              usecuda - No description
    assimp               Assimp - Import various well-known 3D model formats in an uniform manner.
    dualcontouringsample dualcontouringsample - No description
    opticksgl            opticksgl - No description
    optixrap             optixrap - No description
    g4daeopticks         g4daeopticks - No description
    boostrap             boostrap - No description
    glew                 glew - The OpenGL Extension Wrangler library
    thrustrap            thrustrap - No description
    yoctogl              yoctogl - No description
    assimprap            assimprap - No description
    okconf               okconf - No description
    okg4                 okg4 - No description
    useglm               useglm - No description
    useboost             useboost - No description
    yoctoglrap           yoctoglrap - No description
    csgbsp               csgbsp - No description
    cfg4                 cfg4 - No description
    g4dae                g4dae - No description
    implicitmesher       implicitmesher - No description
    usesymbol            usesymbol - No description
    ok                   ok - No description
    g4ok                 g4ok - No description
    okop                 okop - No description
    useg4                useg4 - No description
    glfw3                GLFW - A multi-platform library for OpenGL, window and input
    epsilon:boostrap blyth$ 


/usr/local/opticks/externals/share/bcm/cmake/BCMPkgConfig.cmake

::

    epsilon:cmake blyth$ grep pkgconfig *.cmake
    BCMDeploy.cmake:    bcm_auto_pkgconfig(TARGET ${PARSE_TARGETS})
    BCMPkgConfig.cmake:function(bcm_generate_pkgconfig_file)
    BCMPkgConfig.cmake:function(bcm_preprocess_pkgconfig_property VAR TARGET PROP)
    BCMPkgConfig.cmake:function(bcm_auto_pkgconfig_each)
    BCMPkgConfig.cmake:    bcm_preprocess_pkgconfig_property(LINK_LIBS ${TARGET} INTERFACE_LINK_LIBRARIES)
    BCMPkgConfig.cmake:    bcm_preprocess_pkgconfig_property(INCLUDE_DIRS ${TARGET} INTERFACE_INCLUDE_DIRECTORIES)
    BCMPkgConfig.cmake:    bcm_preprocess_pkgconfig_property(COMPILE_DEFS ${TARGET} INTERFACE_COMPILE_DEFINITIONS)
    BCMPkgConfig.cmake:    bcm_preprocess_pkgconfig_property(COMPILE_OPTS ${TARGET} INTERFACE_COMPILE_OPTIONS)
    BCMPkgConfig.cmake:    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME_LOWER}.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
    BCMPkgConfig.cmake:function(bcm_auto_pkgconfig)
    BCMPkgConfig.cmake:        bcm_auto_pkgconfig_each(TARGET ${PARSE_TARGET} NAME ${PARSE_NAME})
    BCMPkgConfig.cmake:            bcm_auto_pkgconfig_each(TARGET ${TARGET} NAME ${TARGET})
    epsilon:cmake blyth$ 
    epsilon:cmake blyth$ pwd
    /usr/local/opticks/externals/share/bcm/cmake


::

    epsilon:sysrap blyth$ PKG_CONFIG_LIBDIR=/dev/null PKG_CONFIG_PATH=/usr/local/opticks/lib/pkgconfig:/usr/local/opticks/externals/lib/pkgconfig pkg-config SysRap --print-requires
    Package plog was not found in the pkg-config search path.
    Perhaps you should add the directory containing `plog.pc'
    to the PKG_CONFIG_PATH environment variable
    Package 'plog', required by 'SysRap', not found
    epsilon:sysrap blyth$ 








EOU
}
pkg-config-get(){
   local dir=$(dirname $(pkg-config-dir)) &&  mkdir -p $dir && cd $dir

}
