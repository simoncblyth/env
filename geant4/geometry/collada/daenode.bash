# === func-gen- : geant4/geometry/collada/daenode fgp geant4/geometry/collada/daenode.bash fgn daenode fgh geant4/geometry/collada
daenode-src(){      echo geant4/geometry/collada/daenode.bash ; }
daenode-source(){   echo ${BASH_SOURCE:-$(env-home)/$(daenode-src)} ; }
daenode-vi(){       vi $(daenode-source) ; }
daenode-env(){      elocal- ; }
daenode-usage(){ cat << EOU

DAENODE
=========

Xcode Scene Kit Viewer
-----------------------

* https://developer.apple.com/library/mac/recipes/xcode_help-scene_kit_editor/Articles/AboutSceneKitEditor.html

Trackpad Gestures usable on the 3D view:

* zoom: splayed finger pinch 
* rotate: splayed finder rotate 
* pan: two finger slide


White Screen in Preview.app
----------------------------

No error messages in console, thumbnail shows geometry

::

    delta:~ blyth$ strings /System/Library/Frameworks/SceneKit.framework/Versions/A/SceneKit | sort | uniq | grep COLLADA
    10domCOLLADA
    An error occurred while locating the resources needed to open this COLLADA file. Please check that it has not been corrupted.
    An error occurred while parsing the COLLADA file. Please check that it has not been corrupted.
    C3DIO_COLLADA_AccessorType
    C3DIO_COLLADA_C3DShaderStageFromStage
    C3DIO_COLLADA_C3DShaderStageFromStage - unknown stage
    C3DIO_COLLADA_CreateInstanceKernel
    C3DIO_COLLADA_LoadScene
    ...
    The document does not appear to be a valid COLLADA file. Please check that is has not been corrupted.
    The document does not appear to be valid. Please re-create it from your original COLLADA assets.
    Trying to load an invalid COLLADA version for this DOM build!
    __C3DIO_COLLADA_CopySampler failed to resolve image reference
    __C3DIO_COLLADA_CopySampler: error
    ____C3DIO_COLLADA_LoadImageSurface
    ____C3DIO_COLLADA_LoadImageSurface: can't find image
    ____C3DIO_COLLADA_LoadImageSurface: can't find sub image
    exportAsCOLLADAOperationWithDestinationURL:attributes:delegate:didEndSelector:userInfo:
    http://www.collada.org/2005/11/COLLADASchema
    delta:~ blyth$ 



Someones stacktrace 

https://discussions.apple.com/thread/4167225?start=0&tstart=0

::

    Thread 0 Crashed:: Dispatch queue: com.apple.main-thread
    0   com.apple.SceneKit                 0x15ee47aa GenericVertexSet::parseInputsAndP(MeshSourceInfo const&, domP*, daeTArray<daeSmartRef<domInputLocalOffset> > const&, unsigned long long, daeTArray<unsigned long long>&) + 290
    1   com.apple.SceneKit                 0x15ee0aec bool prepareDeindexing<GenericVertexSet>(MeshSourceInfo const&, domMesh const*, GenericVertexSet&) + 247
    2   com.apple.SceneKit                 0x15edf92f Deindexer::deindexGeometry(domGeometry const*) + 501
    3   com.apple.SceneKit                 0x15ee2d07 Deindexer::execute() + 2625
    4   com.apple.SceneKit                 0x15db6cbf C3DIO_COLLADA_LoadScene + 2649
    5   com.apple.SceneKit                 0x15dd2f4f C3DSceneSourceCreateSceneAtIndex + 578
    6   com.apple.SceneKit                 0x15e84ec1 -[SCNSceneSource _createSceneRefWithOptions:statusHandler:] + 480
    7   com.apple.SceneKit                 0x15e85467 -[SCNSceneSource _sceneWithOptions:statusHandler:] + 129
    8   com.apple.SceneKit                 0x15e854d0 -[SCNSceneSource sceneWithOptions:statusHandler:] + 47
    9   com.apple.SceneKit                 0x15e85542 -[SCNSceneSource sceneWithOptions:error:] + 109
    10  com.apple.SceneKit                 0x15e8561c -[SCNSceneSource propertyForKey:] + 60




EOU
}
daenode-dir(){ echo $(env-home)/geant4/geometry/collada ; }
daenode-cd(){  cd $(daenode-dir); }
daenode-mate(){ mate $(daenode-dir) ; }
