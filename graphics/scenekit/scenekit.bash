# === func-gen- : graphics/scenekit/scenekit fgp graphics/scenekit/scenekit.bash fgn scenekit fgh graphics/scenekit
scenekit-src(){      echo graphics/scenekit/scenekit.bash ; }
scenekit-source(){   echo ${BASH_SOURCE:-$(env-home)/$(scenekit-src)} ; }
scenekit-vi(){       vi $(scenekit-source) ; }
scenekit-env(){      elocal- ; }
scenekit-usage(){ cat << EOU



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

