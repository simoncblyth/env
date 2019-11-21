# === func-gen- : graphics/optix7/optix7sandbox fgp graphics/optix7/optix7sandbox.bash fgn optix7sandbox fgh graphics/optix7 src base/func.bash
optix7sandbox-source(){   echo ${BASH_SOURCE} ; }
optix7sandbox-edir(){ echo $(dirname $(optix7sandbox-source)) ; }
optix7sandbox-ecd(){  cd $(optix7sandbox-edir); }
optix7sandbox-dir(){  echo $LOCAL_BASE/env/graphics/optix7/Optix7Sandbox ; }
optix7sandbox-cd(){   cd $(optix7sandbox-dir); }
optix7sandbox-vi(){   vi $(optix7sandbox-source) ; }
optix7sandbox-env(){  elocal- ; }
optix7sandbox-usage(){ cat << EOU

OptiX7Sandbox
===============

Uses a Premake a CMake alternative, but its all windows

* http://blog.johannesmp.com/2016/10/29/getting-started-with-premake/


Looking for Instances
----------------------

::

    [blyth@localhost Optix7Sandbox]$ find . -type f -exec grep -H OPTIX_BUILD {} \;
    ./framework/optix7_core/excludeFromBuild/OptixAccel.cpp:        accelOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
    ./framework/optix7_core/excludeFromBuild/OptixAccel.cpp:        instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ./framework/optix7_core/excludeFromBuild/OptixAccel.cpp:    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE; 
    ./framework/optix7_core/excludeFromBuild/OptixAccel.cpp:    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    ./framework/optix7_core/excludeFromBuild/OptixAccel.cpp:    instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ./framework/optix7_core/excludeFromBuild/OptixMesh.cpp: triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    ./framework/optix7_core/excludeFromBuild/OptixMesh.cpp: accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
    ./framework/optix7_core/excludeFromBuild/OptixMesh.cpp:     | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
    ./framework/optix7_core/excludeFromBuild/OptixMesh.cpp: accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    ./sandbox/Dreamer/source/SceneConfig.cpp:   dreamerConfig.options.accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    ./sandbox/Dreamer/source/SceneConfig.cpp:   dreamerConfig.options.accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;






EOU
}
optix7sandbox-get(){
   local dir=$(dirname $(optix7sandbox-dir)) &&  mkdir -p $dir && cd $dir

    [ ! -d Optix7Sandbox ] && git clone git@github.com:simoncblyth/Optix7Sandbox.git

}
