# === func-gen- : osx/graphicstools/graphicstools fgp osx/graphicstools/graphicstools.bash fgn graphicstools fgh osx/graphicstools
graphicstools-src(){      echo osx/graphicstools/graphicstools.bash ; }
graphicstools-source(){   echo ${BASH_SOURCE:-$(env-home)/$(graphicstools-src)} ; }
graphicstools-vi(){       vi $(graphicstools-source) ; }
graphicstools-env(){      elocal- ; }
graphicstools-usage(){ cat << EOU

Graphics Tools for Xcode, March 2014
======================================

* https://developer.apple.com/downloads/index.action
* graphics_tools_for_xcode_5.1__march_2014.dmg

::

    delta:~ blyth$ ls -1 /Volumes/Graphics\ Tools/
    Acknowledgments.pdf
    Icon Composer.app
    License.rtf
    OpenGL Driver Monitor.app
    OpenGL Profiler.app
    OpenGL Shader Builder.app
    Pixie.app
    Quartz Composer Visualizer.app
    Quartz Composer.app
    Quartz Debug.app


OpenGL Profiler
---------------

* https://developer.apple.com/library/mac/documentation/GraphicsImaging/Conceptual/OpenGLProfilerUserGuide/Introduction/Introduction.html



EOU
}
graphicstools-dir(){ echo $(local-base)/env/osx/graphicstools/osx/graphicstools-graphicstools ; }
graphicstools-cd(){  cd $(graphicstools-dir); }
graphicstools-mate(){ mate $(graphicstools-dir) ; }
graphicstools-get(){
   local dir=$(dirname $(graphicstools-dir)) &&  mkdir -p $dir && cd $dir

}

graphicstools-export(){
   export GL_ENABLE_DEBUG_ATTACH=YES
}


