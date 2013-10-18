Collada Viewers
================

* :google:`collada viewers OSX`
* http://stackoverflow.com/questions/5793642/collada-files-viewer


Preview.app and Xcode.app
----------------------------

From OSX 10.6 Preview.app (and Xcode.app) supports it 


Blender
---------

So slow for full model (on OSX 10.5) that did not succeed to load (just like VRML2). 

Need to work out how to select and extract valid sub geometries using pycollada 
for this to be usable.

Need to find the code, but looks to be rather limited importer

* http://wiki.blender.org/index.php/Doc:2.6/Manual/Data_System/Files/Import/COLLADA


test subcopy.dae import
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    found bundled python: /usr/local/env/graphics/blender/blender-2.59-OSX_10.5_ppc/blender.app/Contents/MacOS/2.59/python
    Cannot find node to instanciate.   ### this repeated 49 times
    cannot find Object for Node with id="top"
    cannot find Object for Node with id="top"
    cannot find Object for Node with id="top"
    cannot find Object for Node with id="top"
    got 50 library nodes to free

Swapping the order of instance_node in the .dae succeeds to get them all into blender.
But looks like heirarcy is lost.


GLC Player
------------

* http://www.glc-player.net/

For compiling GLC_Player 2.3, first install QT4.6 or Qt4.7 and GLC_lib 2.2. 
Then go to the intallation directory and run these commands.

::

    qmake
    make release

Sketch Up
-------------

No go with 10.5. 
Using an old version of the app from an untrusted source is not wise

* http://www.oldapps.com/mac/sketchup.php?system=mac_os_x_10.5_leopard_powerpc

ColladaViewer2 
----------------

* http://cocoadesignpatterns.squarespace.com/home/2012/10/6/loading-and-displaying-collada-models.html
* http://cocoadesignpatterns.squarespace.com/learning-opengl-es-sample-code/
* https://github.com/erikbuck/COLLADAViewer2

meshtool
---------

* https://github.com/pycollada/meshtool
* http://www.panda3d.org/download.php?sdk&version=1.7.2

Project by the pycollada author based on panda3d.


daeview
---------

A demo viewer that comes with pycollada `$(collada-dir)/examples/daeview` depending on pyglet.
It access the triangles/vertices with pycollada and converts to the needed OpenGL vbo etc.. structures
(GLSL renderer only ~250 lines).

::

    sudo port install py26-pyglet 

::

    $(pycollada-dir)/examples/daeview/daeview.py $(env-home)/graphics/collada/pycollada/test.dae


After switching from `polygons` to `polylist` daeview succeeds to load the geometry, but nothing 
visible is apparent.







