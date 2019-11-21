lighthouse2-source(){   echo ${BASH_SOURCE} ; }
lighthouse2-edir(){ echo $(dirname $(lighthouse2-source)) ; }
lighthouse2-ecd(){  cd $(lighthouse2-edir); }
lighthouse2-dir(){  echo $LOCAL_BASE/env/graphics/lighthouse2/lighthouse2 ; }
lighthouse2-cd(){   cd $(lighthouse2-dir); }
lighthouse2-vi(){   vi $(lighthouse2-source) ; }
lighthouse2-env(){  elocal- ; }
lighthouse2-usage(){ cat << EOU


LightHouse 2
===============

OptiX based renderer from author of Brigade renderer Jacco Bikker.
Brigade was subsequently bought by OTOY.

* https://jacco.ompf2.com/
* https://github.com/jbikker/lighthouse2

Linux port in progress : https://github.com/MarijnS95/lighthouse2


Looking for Instancing examples
---------------------------------

::

    [blyth@localhost lighthouse2]$ find . -type f -exec grep -H OPTIX_BUILD_INPUT {} \;
    ...
    ./lib/RenderCore_Optix7Filter/core_mesh.cpp:    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    ./lib/RenderCore_Optix7Filter/rendercore.cpp:   buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ./lib/rendercore_optix7/core_mesh.cpp:  buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    ./lib/rendercore_optix7/rendercore.cpp: buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    [blyth@localhost lighthouse2]$ 



Instructive to compare implementations for the various flavors of OptiX
--------------------------------------------------------------------------

::

    [blyth@localhost lighthouse2]$ find . -type f -exec grep -H class\ CoreInstance {} \;
    ./lib/RenderCore_Optix7Filter/core_mesh.h:class CoreInstance
    ./lib/RenderCore_OptixPrime_B/core_mesh.h:class CoreInstance
    ./lib/RenderCore_OptixPrime_BDPT/core_mesh.h:class CoreInstance
    ./lib/RenderCore_PrimeRef/core_mesh.h:class CoreInstance
    ./lib/rendercore_optix7/core_mesh.h:class CoreInstance
    [blyth@localhost lighthouse2]$ 


lib/RenderCore_PrimeRef/core_mesh.h::

     49 class CoreInstance
     50 {
     51 public:
     52     // constructor / destructor
     53     CoreInstance() = default;
     54     // data
     55     int mesh;                               // ID of the mesh used for this instance
     56     mat4 transform = mat4();
     57     optix::GeometryGroup geometryGroup;     // minimum OptiX scene: GeometryGroup, referencing
     58     optix::GeometryInstance geometryInstance; // GeometryInstance, which in turn references Geometry.
     59 };

lib/rendercore_optix7/core_mesh.h::

     27 class CoreMesh
     28 {
     29 public:
     30     // constructor / destructor
     31     CoreMesh() = default;
     32     ~CoreMesh();
     33     // methods
     34     void SetGeometry( const float4* vertexData, const int vertexCount, const int triCount, const CoreTri* tris, const uint* alphaFlags = 0 );
     35     // data
     36     int triangleCount = 0;                  // number of triangles in the mesh
     37     CoreBuffer<float4>* positions4 = 0;     // vertex data for intersection
     38     CoreBuffer<CoreTri4>* triangles = 0;    // original triangle data, as received from RenderSystem, for shading
     39     CoreBuffer<uchar>* buildTemp = 0;       // reusable temporary buffer for Optix BVH construction
     40     CoreBuffer<uchar>* buildBuffer = 0;     // reusable target buffer for Optix BVH construction
     41     // aceleration structure
     42     uint32_t inputFlags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT /* handled in CUDA shading code instead */ };
     43     OptixBuildInput buildInput;             // acceleration structure build parameters
     44     OptixAccelBuildOptions buildOptions;    // acceleration structure build options
     45     OptixAccelBufferSizes buildSizes;       // buffer sizes for acceleration structure construction
     46     OptixTraversableHandle gasHandle;       // handle to the mesh BVH
     47     CUdeviceptr gasData;                    // acceleration structure data
     48     // global access
     49     static RenderCore* renderCore;          // for access to material list, in case of alpha mapped triangles
     50 };


     56 class CoreInstance
     57 {
     58 public:
     59     // constructor / destructor
     60     CoreInstance() = default;
     61     // data
     62     int mesh = 0;                           // ID of the mesh used for this instance
     63     OptixInstance instance;
     64     float transform[12];                    // rigid transform of the instance
     65 };


CoreMesh 
    geometry primitive holding traversable gasHandle and config for building the acceleration structure

CoreInstance
    mesh ID, OptiXInstance and transform 


::


     20 //  +-----------------------------------------------------------------------------+
     21 //  |  CoreAPI                                                                    |
     22 //  |  Interface between the RenderCore and the RenderSystem.               LH2'19|
     23 //  +-----------------------------------------------------------------------------+
     24 class CoreAPI : public CoreAPI_Base
     25 {   
     26 public:
     .. 
     52     // SetGeometry: update the geometry for a single mesh.
     53     void SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles, const uint* alphaFlags = 0 );
     54     // SetInstance: update the data on a single instance.
     55     void SetInstance( const int instanceIdx, const int modelIdx, const mat4& transform );
     56     // UpdateTopLevel: trigger a top-level BVH update.
     57     void UpdateToplevel();
     58 };
     59 


lib/RenderSystem/rendersystem.cpp::

    138 //  +-----------------------------------------------------------------------------+
    139 //  |  RenderSystem::UpdateSceneGraph                                             |
    140 //  |  Walk the scene graph:                                                      |
    141 //  |  - update all node matrices                                                 |
    142 //  |  - update the instance array (where an 'instance' is a node with            |
    143 //  |    a mesh)                                                            LH2'19|
    144 //  +-----------------------------------------------------------------------------+
    145 void RenderSystem::UpdateSceneGraph()
    146 {
    147     // walk the scene graph to update matrices
    148     Timer timer;
    149     int instanceCount = 0;
    150     bool instancesChanged = false;
    151     for (int s = (int)HostScene::scene.size(), i = 0; i < s; i++)
    152     {
    153         int nodeIdx = HostScene::scene[i];
    154         HostNode* node = HostScene::nodes[nodeIdx];
    155         mat4 T;
    156         instancesChanged |= node->Update( T /* start with an identity matrix */, instanceCount );
    157     }
    158     stats.sceneUpdateTime = timer.elapsed();
    159     // synchronize instances to device if anything changed
    160     if (instancesChanged || meshesChanged)
    161     {
    162         // resize vector (this is free if the size didn't change)
    163         HostScene::instances.resize( instanceCount );
    164         // send instances to core
    165         for (int instanceIdx = 0; instanceIdx < instanceCount; instanceIdx++)
    166         {
    167             HostNode* node = HostScene::nodes[HostScene::instances[instanceIdx]];
    ///
    ///   instances contains sequence of node indices
    ///
    168             node->instanceID = instanceIdx;
    169             int dummy = node->Changed(); // prevent superfluous update in the next frame
    170             core->SetInstance( instanceIdx, node->meshID, node->combinedTransform );
    171         }
    172         // finalize
    173         core->UpdateToplevel();
    174         meshesChanged = false;
    175     }
    176 }


lib/RenderSystem/host_scene.h focus on geo/structure::

     23 //  +-----------------------------------------------------------------------------+
     24 //  |  HostScene                                                                  |
     25 //  |  Module for scene I/O and host-side management.                             |
     26 //  |  This is a pure static class; we will not have more than one scene.   LH2'19|
     27 //  +-----------------------------------------------------------------------------+
     28 class HostNode;
     29 class HostScene
     30 {
     31 public:
     ...
     45     static int FindNode( const char* name );
     46     static void SetNodeTransform( const int nodeId, const mat4& transform );
     51     static int AddMesh( const char* objFile, const char* dir, const float scale = 1.0f );
     52     static int AddMesh( const int triCount );
     53     static void AddTriToMesh( const int meshId, const float3& v0, const float3& v1, const float3& v2, const int matId );
     54     static int AddScene( const char* sceneFile, const char* dir, const mat4& transform );
     55     static int AddInstance( const int meshId, const mat4& transform );
     56     static void RemoveInstance( const int instId );
     ...
     64     static vector<int> scene; // node indices for scene 0; each of these may have children. TODO: scene 1..X.
     65     static vector<HostNode*> nodes; 
     66     static vector<HostMesh*> meshes;
     ...
     69     static vector<int> instances; // list of indices of nodes that point to a mesh
     ...
     76     static Camera* camera;
     ...
     79 };  




lib/rendercore_optix7/rendercore.cpp::

    031 //  +-----------------------------------------------------------------------------+
     32 //  |  RenderCore                                                                 |
     33 //  |  Encapsulates device code.                                            LH2'19|
     34 //  +-----------------------------------------------------------------------------+
     35 class RenderCore
     36 {
     ...      
     55     // geometry and instances:
     56     // a scene is setup by first passing a number of meshes (geometry), then a number of instances.
     57     // note that stored meshes can be used zero, one or multiple times in the scene.
     58     // also note that, when using alpha flags, materials must be in sync.
     59     void SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles, const uint* alphaFlags = 0 );
     60     void SetInstance( const int instanceIdx, const int modelIdx, const mat4& transform );
     61     void UpdateToplevel();
     65 private:
     ... 
     67     void CreateOptixContext( int cc );
     ...
     75     vector<CoreMesh*> meshes;                       // list of meshes, to be referenced by the instances
     76     vector<CoreInstance*> instances;                    // list of instances: model id plus transform
    ...

    118 public:
    120     static OptixDeviceContext optixContext;         // static, for access from CoreMesh
    ...
    122     OptixShaderBindingTable sbt;
    123     OptixModule ptxModule;
    124     OptixPipeline pipeline;
    125     OptixProgramGroup progGroup[5];
    126     OptixTraversableHandle bvhRoot;
    127     Params params;
    128     CUdeviceptr d_params;
    129 };






::

    331 //  +-----------------------------------------------------------------------------+
    332 //  |  RenderCore::SetGeometry                                                    |
    333 //  |  Set the geometry data for a model.                                   LH2'19|
    334 //  +-----------------------------------------------------------------------------+
    335 void RenderCore::SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles, const uint* alphaFlags )
    336 {
    337     // Note: for first-time setup, meshes are expected to be passed in sequential order.
    338     // This will result in new CoreMesh pointers being pushed into the meshes vector.
    339     // Subsequent mesh changes will be applied to existing CoreMeshes. This is deliberately
    340     // minimalistic; RenderSystem is responsible for a proper (fault-tolerant) interface.
    341     if (meshIdx >= meshes.size()) meshes.push_back( new CoreMesh() );
    342     meshes[meshIdx]->SetGeometry( vertexData, vertexCount, triangleCount, triangles, alphaFlags );
    343 }
    344 
    345 //  +-----------------------------------------------------------------------------+
    346 //  |  RenderCore::SetInstance                                                    |
    347 //  |  Set instance details.                                                LH2'19|
    348 //  +-----------------------------------------------------------------------------+
    349 void RenderCore::SetInstance( const int instanceIdx, const int meshIdx, const mat4& matrix )
    350 {
    351     // Note: for first-time setup, meshes are expected to be passed in sequential order.
    352     // This will result in new CoreInstance pointers being pushed into the instances vector.
    353     // Subsequent instance changes (typically: transforms) will be applied to existing CoreInstances.
    354     if (instanceIdx >= instances.size())
    355     {
    356         // create a geometry instance
    357         CoreInstance* newInstance = new CoreInstance();
    358         memset( &newInstance->instance, 0, sizeof( OptixInstance ) );
    359         newInstance->instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    360         newInstance->instance.instanceId = instanceIdx;
    361         newInstance->instance.sbtOffset = 0;
    362         newInstance->instance.visibilityMask = 255;
    363         newInstance->instance.traversableHandle = meshes[meshIdx]->gasHandle;
    ///
    ///  instance holds the gasHandle to identify which mesh it refers to 
    ///
    364         memcpy( newInstance->transform, &matrix, 12 * sizeof( float ) );
    365         memcpy( newInstance->instance.transform, &matrix, 12 * sizeof( float ) );
    366         instances.push_back( newInstance );
    367     }
    368     // update the matrices for the transform
    369     memcpy( instances[instanceIdx]->transform, &matrix, 12 * sizeof( float ) );
    370     memcpy( instances[instanceIdx]->instance.transform, &matrix, 12 * sizeof( float ) );
    371     // set/update the mesh for this instance
    372     instances[instanceIdx]->mesh = meshIdx;
    373 }


* note that the instances array holds multiple types of instance 



lib/rendercore_optix7/rendercore.cpp::

    375 //  +-----------------------------------------------------------------------------+
    376 //  |  RenderCore::UpdateToplevel                                                 |
    377 //  |  After changing meshes, instances or instance transforms, we need to        |
    378 //  |  rebuild the top-level structure.                                     LH2'19|
    379 //  +-----------------------------------------------------------------------------+
    380 void RenderCore::UpdateToplevel()
    381 {
    382     // resize instance array if more space is needed
    383     if (instances.size() > (size_t)instanceArray->GetSize())
    /// 
    ///     instances: vector of CoreInstance (meshID, OptixInstance, transform) 
    ///
    384     {
    385         delete instanceArray;
    386         instanceArray = new CoreBuffer<OptixInstance>( instances.size() + 4, ON_HOST | ON_DEVICE );
    387     }
    388     // copy instance descriptors to the array, sync with device
    389     for (int s = (int)instances.size(), i = 0; i < s; i++)
    390     {
    391         instances[i]->instance.traversableHandle = meshes[instances[i]->mesh]->gasHandle;
    ///      
    ///         fill in the traversableHandles of the OptixInstances with the gasHandles from the meshes   
    ///
    392         instanceArray->HostPtr()[i] = instances[i]->instance;
    ///  
    ///         tee up host side instances with the gasHandles

    393     }
    394     instanceArray->CopyToDevice();
    /// 
    ///     copy over all instances for different meshes   

    395     // build the top-level tree
    396     OptixBuildInput buildInput = {};
    397     buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    398     buildInput.instanceArray.instances = (CUdeviceptr)instanceArray->DevPtr();
    399     buildInput.instanceArray.numInstances = (uint)instances.size();

    ///   describes the instances array  
    ///   huh what about bbox ?  that must be accessed via the gasHandle  
    ///   nope : lighthouse deals in tris, bbox only needed for custom (analytic) geometry ?
 
    400     OptixAccelBuildOptions options = {};
    401     options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    402     options.operation = OPTIX_BUILD_OPERATION_BUILD;
    403     static size_t reservedTemp = 0, reservedTop = 0;
    404     static CoreBuffer<uchar> *temp, *topBuffer = 0;
    405     OptixAccelBufferSizes sizes;
    406     CHK_OPTIX( optixAccelComputeMemoryUsage( optixContext, &options, &buildInput, 1, &sizes ) );
    407     if (sizes.tempSizeInBytes > reservedTemp)
    408     {
    409         reservedTemp = sizes.tempSizeInBytes + 1024;
    410         delete temp;
    411         temp = new CoreBuffer<uchar>( reservedTemp, ON_DEVICE );
    412     }
    413     if (sizes.outputSizeInBytes > reservedTop)
    414     {
    415         reservedTop = sizes.outputSizeInBytes + 1024;
    416         delete topBuffer;
    417         topBuffer = new CoreBuffer<uchar>( reservedTop, ON_DEVICE );
    418     }
    419     CHK_OPTIX( optixAccelBuild( optixContext, 0, &options, &buildInput, 1, (CUdeviceptr)temp->DevPtr(),
    420         reservedTemp, (CUdeviceptr)topBuffer->DevPtr(), reservedTop, &bvhRoot, 0, 0 ) );

    ///
    ///     IAS always take one OptixBuildInput (GAS can take multiple)
    ///     the bvhRoot output handle is set, corresponding to top traversable that
    //      is needed in launch parameters 
    ///

    421 }


::

     76     vector<CoreInstance*> instances;                    // list of instances: model id plus transform
     ...
     99     CoreBuffer<OptixInstance>* instanceArray = 0;   // instance descriptors for Optix













::

    [blyth@localhost lighthouse2]$ find . -type f -exec grep -H class\ CoreBuffer {} \;
    ./lib/CUDA/shared_host_code/cudatools.h:template <class T> class CoreBuffer





EOU
}
lighthouse2-get(){
   local dir=$(dirname $(lighthouse2-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d lighthouse2 ] && git clone git@github.com:simoncblyth/lighthouse2.git
}

lighthouse2-f () 
{ 
    local str="${1:-OptixBuildInput}";
    local opt=${2:--H};
    local iwd=$PWD;
    lighthouse2-cd;
    find . \( -name '*.sh' -or -name '*.bash' -or -name '*.cu' -or -name '*.cc' -or -name '*.hh' -or -name '*.cpp' -or -name '*.hpp' -or -name '*.h' -or -name '*.txt' -or -name '*.cmake' -or -name '*.py' \) -exec grep $opt "$str" {} \;
}


