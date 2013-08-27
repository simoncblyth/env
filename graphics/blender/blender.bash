# === func-gen- : graphics/blender/blender fgp graphics/blender/blender.bash fgn blender fgh graphics/blender
blender-src(){      echo graphics/blender/blender.bash ; }
blender-srcd(){     echo $(env-home)/graphics/blender ; }
blender-source(){   echo ${BASH_SOURCE:-$(env-home)/$(blender-src)} ; }
blender-vi(){       vi $(blender-source) ; }
blender-env(){      elocal- ; }
blender-usage(){ cat << EOU
BLENDER
========

* http://wiki.blender.org/index.php/Doc:2.4/Manual
* http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python
* http://www.blender.org/documentation/blender_python_api_2_61_0/info_tips_and_tricks.html
* http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Console

* http://wiki.blender.org/index.php/Extensions:2.4/Py/Scripts/Import/X3D_VRML97

* http://accad.osu.edu/~pgerstma/class/vnv/resources/info/AnnotatedVrmlRef/ch3-323.htm


PYTHON CONSOLE ACCESS OSX Blender 2.59
----------------------------------------

* Each Blender View (Pane) has an selector dropdown at bottom left 
* Choose "Python Console" 
* Note that sometimes panes start with zero size, so to see the content need to drag
  the adjacent pane to give the original some room for content to be visible



BLENDER BASICS
-----------------

#. Use the Scene tree pane, to select an object then can change mode from "Object" in order to see the vertices



IMPORT X3D/VRML
----------------

* `File > Import`


::

    simon:blender-2.59-OSX_10.5_ppc blyth$ ll $(blender-scripts)/addons/io_scene_x3d/
    total 320
    -rw-r--r--@  1 blyth  wheel  89169 17 Aug  2011 import_x3d.py
    -rw-r--r--@  1 blyth  wheel  61750 17 Aug  2011 export_x3d.py
    -rw-r--r--@  1 blyth  wheel   6991 17 Aug  2011 __init__.py
    drwxr-xr-x  70 blyth  wheel   2380 17 Aug  2011 ..
    drwxr-xr-x   6 blyth  wheel    204 27 Aug 16:31 .
    drwxr-xr-x   4 blyth  wheel    136 27 Aug 16:44 __pycache__

::

    >>> import io_scene_x3d
    >>> 
    >>> dir(io_scene_x3d)
    ['BoolProperty', 'EnumProperty', 'ExportHelper', 'ExportX3D', 'ImportHelper', 'ImportX3D', 'StringProperty', '__addon_enabled__', '__builtins__', '__cached__', '__doc__', '__file__', '__name__', '__package__', '__path__', '__time__', 'axis_conversion', 'bl_info', 'bpy', 'gpu', 'menu_func_export', 'menu_func_import', 'path_reference_mode', 'register', 'unregister']
    >>> 

    >>> bpy.ops
    <module like class 'bpy.ops'>

    >>> dir(bpy.ops)
    ['action', 'anim', 'armature', 'boid', 'brush', 'buttons', 'cloth', 'console', 'constraint', 'curve', 'ed', 'export_anim', 'export_mesh', 'export_scene', 'file', 'fluid', 'font', 'gpencil', 'graph', 'group', 'help', 'image', 'import_anim', 'import_curve', 'import_mesh', 'import_scene', 'info', 'lamp', 'lattice', 'logic', 'marker', 'material', 'mball', 'mesh', 'nla', 'node', 'object', 'outliner', 'paint', 'particle', 'pose', 'poselib', 'ptcache', 'render', 'scene', 'screen', 'script', 'sculpt', 'sequencer', 'sketch', 'sound', 'surface', 'text', 'texture', 'time', 'transform', 'ui', 'uv', 'view2d', 'view3d', 'wm', 'world']

    >>> dir(bpy.ops.import_scene)
    ['autodesk_3ds', 'obj', 'x3d']

    >>> import_x3d = bpy.ops.import_scene.x3d

    >>> import_x3d("EXEC_DEFAULT",filepath="/Users/blyth/e/graphics/vrml/samples/basics-1.wrl")
    {'FINISHED'}




Sample WRL
------------

* http://tecfa.unige.ch/guides/vrml/vrmlman/node6.html
* http://tecfa.unige.ch/guides/vrml/examples/basics/basics-1.wrl


Interactive GUI Operation
--------------------------

::

    simon:blender-2.59-OSX_10.5_ppc blyth$ blender-x
    ndof: 3Dx driver not found
    Info: Config directory with "startup.blend" file not found.
    found bundled python: /usr/local/env/graphics/blender/blender-2.59-OSX_10.5_ppc/blender.app/Contents/MacOS/2.59/python


Commandline Operation
----------------------

::

    simon:~ blyth$ blender-x --help
    Blender 2.59 (sub 0)
    Usage: blender [args ...] [file] [args ...]


GUI Python Console Pane
------------------------


::

   bpy.data.objects['ShapeIndexedFaceSet'].data.vertices[0].co

   for _ in bpy.data.objects['ShapeIndexedFaceSet'].data.vertices:
        print(_.co)

   for _ in bpy.data.objects['ShapeIndexedFaceSet'].data.faces:
        print(_.index)   # MeshFace

   for _ in bpy.data.objects['ShapeIndexedFaceSet'].data.faces:
        print(",".join(map(str,_.vertices)))     # 1,2,3,4    5,6,1   1,4,5     (NB 1-based, unlike source wrl)

   print("\n".join(map(str,bpy.data.objects)))




Python Console
---------------

* http://www.blender.org/documentation/blender_python_api_2_59_0/info_overview.html

::

    simon:blender-2.59-OSX_10.5_ppc blyth$ blender-x --python-console
    ndof: 3Dx driver not found
    found bundled python: /usr/local/env/graphics/blender/blender-2.59-OSX_10.5_ppc/blender.app/Contents/MacOS/2.59/python
    Python 3.2 (r32:88445, Mar  1 2011, 21:58:29) 
    [GCC 4.2.1 (Apple Inc. build 5664)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    (InteractiveConsole)
    >>> import bpy
    >>> bpy.data.objects["Cube"].data.vertices[0]
    bpy.data.meshes["Cube"].vertices[0]
    >>> bpy.data.objects["Cube"].data.vertices[0].co
    Vector((1.0, 0.9999999403953552, -1.0))
    >>> bpy.data.objects["Cube"].data.vertices[0].co.x
    1.0

    >>> bpy.types
    <RNA_Types object at 0xbdb32c0>
    >>> dir(bpy.types)
    ['ACTION_OT_clean', 'ACTION_OT_clickselect', 'ACTION_OT_copy',
    'ACTION_OT_delete', 'ACTION_OT_duplicate', 'ACTION_OT_duplicate_move',
    'ACTION_OT_extrapolation_type', 'ACTION_OT_frame_jump',
    'ACTION_OT_handle_type', 'ACTION_OT_interpolation_type', 
    ...
    'WorldTextureSlot', 'WorldTextureSlots', 'XnorController', 'XorController']

    >>> len(dir(bpy.types))
    2352


EOU
}
blender-dir(){ echo $(local-base)/env/graphics/blender/$(blender-name) ; }
blender-cd(){  cd $(blender-dir); }
blender-mate(){ mate $(blender-dir) ; }

blender-url(){  echo http://download.blender.org/release/Blender2.59/blender-2.59-OSX_10.5_ppc.zip ; } #  last PPC release ?
blender-name(){ echo blender-2.59-OSX_10.5_ppc ; }
blender-version(){ echo 2.59 ; }

blender-get(){
   local dir=$(dirname $(blender-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(blender-url)
   local zip=$(basename $url)
   local nam=$(blender-name)

   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -d "$nam" ] && unzip $zip
}


#blender-app(){       echo $HOME/Applications/$(blender-name)/blender.app ; }
blender-app(){       echo $(blender-dir)/blender.app ; }
blender-macos(){     echo $(blender-app)/Contents/MacOS ; }
blender-exe(){       echo $(blender-macos)/blender ; }
blender-scripts(){   echo $(blender-macos)/$(blender-version)/scripts ; }


blender-bpy(){       echo $(blender-srcd)/bpy ; }
blender-x(){         `blender-exe` $* ; } 

blender-bpy-ln(){
   local msg="=== $FUNCNAME :"
   local cmd="ln -s $(blender-bpy) $(blender-scripts)/myscripts "
   echo $msg $cmd
   eval $cmd
}


blender-otool(){ otool -L $(blender-exe) ; } 
blender-tmp(){   echo /tmp/workflow/blender ; }

blender-run(){   
  mkdir -p $(blender-tmp)
  cd $(blender-tmp)
 $(blender-exe)  $*  
}

blender-mha(){
  blender-run -P $(blender-bpy)/mhalpha.py
}




