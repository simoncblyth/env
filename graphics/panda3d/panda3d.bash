# === func-gen- : graphics/panda3d/panda3d fgp graphics/panda3d/panda3d.bash fgn panda3d fgh graphics/panda3d
panda3d-src(){      echo graphics/panda3d/panda3d.bash ; }
panda3d-source(){   echo ${BASH_SOURCE:-$(env-home)/$(panda3d-src)} ; }
panda3d-vi(){       vi $(panda3d-source) ; }
panda3d-env(){      elocal- ; }
panda3d-usage(){ cat << EOU

PANDA3D
=========

* http://www.panda3d.org/download.php?platform=macosx&version=1.7.2&sdk
* http://www.panda3d.org/download.php?sdk&version=1.8.1
* https://developer.nvidia.com/cg-toolkit

OSX
----

* http://www.panda3d.org/manual/index.php/Getting_Started_on_OSX

GUI dmg installer places examples in /Developer/Examples/Panda3D/


Need to use the system python 2.5.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:Solar-System blyth$ python Tut-Step-1-Blank-Window.py 
    Traceback (most recent call last):
      File "Tut-Step-1-Blank-Window.py", line 11, in <module>
        import direct.directbase.DirectStart
    ImportError: No module named direct.directbase.DirectStart


Need to install Cg Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See `cg-`. 

::

    simon:Solar-System blyth$ /usr/bin/python Tut-Step-1-Blank-Window.py 
    ...
    from libpandaModules import *
      File "/Developer/Panda3D/lib/pandac/libpandaModules.py", line 2, in <module>
        Dtool_PreloadDLL("libpanda")
      File "/Developer/Panda3D/lib/pandac/extension_native_helpers.py", line 79, in Dtool_PreloadDLL
        imp.load_dynamic(module, pathname)
    ImportError: dlopen(/Developer/Panda3D/lib/libpanda.dylib, 2): Library not loaded: @executable_path/../Frameworks/Cg.framework/Cg
      Referenced from: /Developer/Panda3D/lib/libpanda.dylib
      Reason: image not found


Version incompatibility ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:Solar-System blyth$ /usr/bin/python Tut-Step-1-Blank-Window.py
    DirectStart: Starting the game.
    Traceback (most recent call last):
      File "Tut-Step-1-Blank-Window.py", line 11, in <module>
        import direct.directbase.DirectStart
      File "dstroot/pythoncode/Developer/Panda3D/lib/direct/directbase/DirectStart.py", line 3, in <module>
      File "dstroot/pythoncode/Developer/Panda3D/lib/direct/showbase/ShowBase.py", line 10, in <module>
      File "/Developer/Panda3D/lib/pandac/PandaModules.py", line 32, in <module>
        from libp3visionModules import *
      File "/Developer/Panda3D/lib/pandac/libp3visionModules.py", line 2, in <module>
        Dtool_PreloadDLL("libp3vision")
      File "/Developer/Panda3D/lib/pandac/extension_native_helpers.py", line 79, in Dtool_PreloadDLL
        imp.load_dynamic(module, pathname)
    ImportError: dlopen(/Developer/Panda3D/lib/libp3vision.dylib, 2): Symbol not found: _cvCreateCameraCapture
      Referenced from: /Developer/Panda3D/lib/libp3vision.dylib
      Expected in: dynamic lookup


* https://www.panda3d.org/forums/viewtopic.php?f=5&t=16374
* http://www.panda3d.org/forums/viewtopic.php?t=9494

   * developer advised to disable vision module

/Developer/Panda3D/lib/pandac/PandaModules.py::

     30 
     31 #try:
     32 #  from libp3visionModules import *
     33 #except ImportError, err:
     34 #  if "DLL loader cannot find" not in str(err):
     35 #    raise
     36 




EOU
}
panda3d-dir(){ echo $(local-base)/env/graphics/panda3d/graphics/panda3d-panda3d ; }
panda3d-cd(){  cd $(panda3d-dir); }
panda3d-mate(){ mate $(panda3d-dir) ; }
panda3d-get(){
   local dir=$(dirname $(panda3d-dir)) &&  mkdir -p $dir && cd $dir


   http://www.panda3d.org/download.php?platform=macosx&version=1.7.2&sdk

}

panda3d-examples(){
   cd /Developer/Examples/Panda3D/Solar-System
   
   #ppython Tut-Step-1-Blank-Window.py   # opens a blank window
   ppython  Tut-Step-3-Load-Model.py     # stylized planet, no 3D interface
   ppython  Tut-Step-5-Complete-Solar-System.py   # multiple planets moving around



}


