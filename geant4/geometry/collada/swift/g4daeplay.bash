# === func-gen- : geant4/geometry/collada/swift/g4daeplay fgp geant4/geometry/collada/swift/g4daeplay.bash fgn g4daeplay fgh geant4/geometry/collada/swift
g4daeplay-src(){      echo geant4/geometry/collada/swift/g4daeplay.bash ; }
g4daeplay-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4daeplay-src)} ; }
g4daeplay-vi(){       vi $(g4daeplay-source) ; }
g4daeplay-env(){      elocal- ; }
g4daeplay-usage(){ cat << EOU

G4DAEPLAY
==========

Experiment with reading COLLADA DAE exported from Geant4
into OSX SceneKit Swift Playground.

Keeping notes here, as using Xcode Beta Version 6.0 (6A254o)
means that playground format is liable to change.

Usage tips
------------

* to force a rerun, do a dummy edit
* avoid excessive reruns by arranging code to have a small "main" 


Refs
----

#. http://stackoverflow.com/questions/24126669/using-scenekit-in-swift-playground
#. https://developer.apple.com/library/ios/recipes/xcode_help-source_editor/chapters/ExploringandEvaluatingSwiftCodeinaPlayground.html
#. https://developer.apple.com/library/mac/documentation/SceneKit/Reference/SCNCamera_Class/Reference/SCNCamera.html
#. https://developer.apple.com/library/mac/documentation/3DDrawing/Conceptual/SceneKit_PG/Introduction/Introduction.html
#. https://developer.apple.com/library/prerelease/mac/documentation/SceneKit/Reference/SCNSceneRenderer_Protocol/index.html


To use the camera to display a scene, attach it to the camera property of node,
then select that node using the pointOfView property of the view (or layer or
renderer) rendering the scene.

to ensure that a camera always points at a particular element of your scene
even when that element moves, attach a SCNLookAtConstraint object to the node
containing the camera.

FUNCTIONS
-----------

*g4daeplay-open*
     Open into Xcode, for graphical view:
  
     * shortcut: opt-cmd-ret
     * *View > Assistant Editor > Show Assistant Editor*


EOU
}

g4daeplay-name(){ echo g4daeplay.playground ; }
g4daeplay-dir(){ echo $(env-home)/geant4/geometry/collada/swift ; }
g4daeplay-path(){ echo $(g4daeplay-dir)/$(g4daeplay-name) ; }
g4daeplay-cd(){  cd $(g4daeplay-dir); }
g4daeplay-mate(){ mate $(g4daeplay-dir) ; }
g4daeplay-get(){
   local dir=$(dirname $(g4daeplay-dir)) &&  mkdir -p $dir && cd $dir

}

g4daeplay-open(){

   #export-
   #export-export


   export DAE_NAME_AD=/Users/blyth/Desktop/dae/g4dae_export_dayabay_ad_stripped.dae

   open $(g4daeplay-path)
}

