# === func-gen- : graphics/scenekit/scenekit fgp graphics/scenekit/scenekit.bash fgn scenekit fgh graphics/scenekit
scenekit-src(){      echo graphics/scenekit/scenekit.bash ; }
scenekit-source(){   echo ${BASH_SOURCE:-$(env-home)/$(scenekit-src)} ; }
scenekit-vi(){       vi $(scenekit-source) ; }
scenekit-env(){      elocal- ; }
scenekit-usage(){ cat << EOU


SceneKit
=========



Xcode 9.3 : SCNScene 
----------------------

How to overlay HUD ?
~~~~~~~~~~~~~~~~~~~~~~~~~

* https://stackoverflow.com/questions/24419193/how-do-i-create-a-hud-on-top-of-my-scenekit-scene

+1 :) You can find more info in the docs and in the Bananas and Vehicle sample
code projects. Another benefit of SK overlays is you can make your HUD cross
platform, instead of using UIKit on iOS and AppKit on OS X. – rickster Jun 26 2014 at 16:31

* https://developer.apple.com/documentation/scenekit/scnscenerenderer/1524051-overlayskscene

::

    SKContainerOverlay *skContainerOverlay = [[SKContainerOverlay alloc] initWithSize:self.sceneView.bounds.size];
    self.sceneView.overlaySKScene = skContainerOverlay;
    self.sceneView.overlaySKScene.hidden = NO;
    self.sceneView.overlaySKScene.scaleMode = SKSceneScaleModeResizeFill; // Make sure SKScene bounds are the same as our SCNScene
    self.sceneView.overlaySKScene.userInteractionEnabled = YES;



Dynamic Loading
----------------

* http://www.the-nerd.be/2014/11/07/dynamically-load-collada-files-in-scenekit-at-runtime/#more-457


ColladaDOM Warnings 
--------------------

Elements within **extra** tags all yield warnings::

    (chroma_env)delta:swift blyth$ ./g4daeparse.swift > out
    (chroma_env)delta:swift blyth$ grep ColladaDOM out | wc -l
        6136

    (chroma_env)delta:swift blyth$ tail -10 out

    ColladaDOM Warning: The DOM was unable to create an element named bordersurface at line 153314. Probably a schema violation.

    ColladaDOM Warning: The DOM was unable to create an element named bordersurface at line 153318. Probably a schema violation.

    ColladaDOM Warning: The DOM was unable to create an element named bordersurface at line 153324. Probably a schema violation.

    ColladaDOM Warning: The DOM was unable to create an element named meta at line 153324. Probably a schema violation.

    5892


According to https://www.apple.com/opensource/ SceneKit uses colladaDOM 2.2

* http://sourceforge.net/projects/collada-dom/
* http://sourceforge.net/projects/collada-dom/files/Collada%20DOM/Collada%20DOM%202.2/Collada%20DOM%202.2.zip/download


Extracting Vertices using SCNGeometrySource
---------------------------------------------

* http://stackoverflow.com/questions/17250501/extracting-vertices-from-scenekit


SceneKit resources
-------------------


::

    delta:dae blyth$ ll /System/Library/Frameworks/SceneKit.framework/Versions/A/Resources/
    total 208
    -rw-r--r--   1 root  wheel    6477 Sep 13  2013 SCNKitTypeInspector.nib
    -rw-r--r--   1 root  wheel    7544 Sep 13  2013 SCNKitSceneGraphView.nib
    -rw-r--r--   1 root  wheel    8065 Sep 13  2013 SCNKitRenderPassView.nib
    -rw-r--r--   1 root  wheel    5915 Sep 13  2013 SCNKitLibraryView.nib
    drwxr-xr-x   4 root  wheel     136 Oct  5  2013 BridgeSupport
    -rw-r--r--   1 root  wheel    4726 Oct  6  2013 xml.xsd
    -rw-r--r--   1 root  wheel   32054 Oct  6  2013 SCNRendererOptions.nib
    -rw-r--r--   1 root  wheel    5343 Oct  6  2013 SCNMonitor.nib
    -rw-r--r--   1 root  wheel   11100 Oct  6  2013 SCNKitImagePackerView.nib
    -rw-r--r--   1 root  wheel  413957 Oct  6  2013 COLLADASchema.xsd
    -rw-r--r--   1 root  wheel     457 Jul 14 16:04 version.plist
    -rw-r--r--   1 root  wheel    1079 Jul 14 16:04 Info.plist
    drwxr-xr-x   7 root  wheel     238 Jul 14 16:05 ..
    drwxr-xr-x  14 root  wheel     476 Jul 14 16:05 .



EOU
}
scenekit-dir(){ echo $(local-base)/env/graphics/scenekit/graphics/scenekit-scenekit ; }
scenekit-cd(){  cd $(scenekit-dir); }
scenekit-mate(){ mate $(scenekit-dir) ; }
scenekit-get(){
   local dir=$(dirname $(scenekit-dir)) &&  mkdir -p $dir && cd $dir

}

scenekit-rdir(){ echo /System/Library/Frameworks/SceneKit.framework/Versions/A/Resources ; }
scenekit-validate(){
    xmllint --noout --schema $(scenekit-rdir)/COLLADASchema.xsd $*
}

