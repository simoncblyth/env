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






