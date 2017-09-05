Instcull Classes
====================


Prime Actors
--------------


ICDemo.hh
     top level app class doing both instance culling and LOD forking 

Demo.hh
     early dev stage (monolithic) top level app class doing just instance culling, without LOD forking
     rather monolithic : prior to splitting off into separate shaders done in ICDemo

CullShader.hh
     used from ICDemo without WITH_LOD

LODCullShader.hh 
     used from ICDemo with WITH_LOD
     LODCullShader::setupFork
     LODCullShader::applyFork
     LODCullShader::applyForkStreamQueryWorkaround


InstShader.hh
     used from ICDemo

Shader.hh
     trivial non-instancd render shader, not using SContext

SContext.hh
     common cross-shader uniform block handling 


Geometry
-----------

Box.hh  
    pre-Prim shape, still used ?
Pos.hh
     early testing source geometry vertex positions, still used ?

UV.hh
     parametric coordinate utility for mesh generation
Prim.hh
     geometry primitive, housing vertex and element Buf with concatenation capabilities used for LOD-ing 
Cube.hh
     geometry Prim subclass
Sphere.hh
     geometry Prim subclass
Tri.hh
     geometry Prim subclass
Primitives.hh
     Tri, Cube, Sphere headers

Tra.hh
     used to generate buffers of instance transforms

Geom.hh
     collection of buffers and transforms for specific geometries

BB.hh
    Bounding box with creation from vectors and buffers capabilty


Projection Math
-----------------

Cam.hh
    lightweight camera providing projection matrices
Vue.hh
    lightweight viewpoint: eye, look, up relative to center extent of region of interest
Comp.hh
     composition coodinating Cam and Vue

Buffers
---------

Buf.hh
    lightweight buffer with OpenGL upload capabilities
Buf4.hh
    collection of up to 4 Buf

Utilities
-----------

Att.hh
    OpenGL attribute querying utility 
Prog.hh
    bare bones GLSL shader program creation,compilation,linking
G.hh
    glm dumping 
GU.hh
    OpenGL utility, error dumping
GLEQ.hh
    interface to gleq keyboard/mouse event system
Frame.hh
    GLFW/GLEW OpenGL context setup



