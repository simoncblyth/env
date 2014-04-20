#!/usr/bin/env python
"""
DAEVIEWGL
==========

.. seealso:: User instructions :doc:`/geant4/geometry/collada/daeview/daeviewgl_usage`

Features
---------

usage text
~~~~~~~~~~~

On pressing "U" usage text describing the actions of each key 
are written to stdout.  (TODO: display to screen)

bookmarks
~~~~~~~~~~~

Number keys are used to create and visit bookmarks. 
While pressing a number key 1-9 click on a solid to create a bookmark, 
the view adopts the coordinate frame corresponding to the solid clicked. 
Subsequently pressing number keys 0-9 visit the bookmarks created, 
and pressing SPACE updates the current bookmark (last one created/visited) 
to bake any offsets made from the view into the view. 
Bookmark 0 is created at startup for the initial viewpoint.

A bookmark comprises: 

* a solid (or the entiremesh), which defines the view coordinate system. 
  Unit of length is the extent of the solid 
* "eye" position, eg -2,-2,0  
* "look" position, eg 0,0,0 : about which trackball rotations are applied
* "up" direction, eg 0,0,1 

Note that trackball translations/rotations do not update the "view", 
although they do of course update what you see. To solidify trackballing
offsets into the current view press SPACE. 

* drag around to rotate about the "look" point using a virtual trackball,
  XY positions are projected onto virtual sphere trackball, which allow
  offset rotations to be obtained via some Quaternion math   
* press "X" while dragging around to translate in screen XY direction 
* press "Z" while dragging up/down to tranlate in screen Z direction (in/out)

parallel projection
~~~~~~~~~~~~~~~~~~~~

Press "P" to toggle between orthographic/parallel projection and the default
perspective projection.

TODO: Get chroma raycast to work in parallel projection mode


placemarks
~~~~~~~~~~~

The commandline to return to a viewpoint and camera configuration
is written to stdout on exiting or on pressing "W".

near/far controls (and some wierdness)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you approach too closely to some geometry it will disappear due to 
near clipping. Similarly distant geometry can disappear from far clipping.
To control the near/far clipping planes:

* press "N" while dragging up/down to change "near" distance
* press "F" while dragging up/down to change "far" distance
 
Somehow changing "near" somehow acts to change "far" clipping. 
Possibly this is due to limited depth buffering, the issue 
seems less prevalent the less extreme "near" and "far" values.

To illustrate the viewing frustum (square pyramid chopped at near/far planes
with "eye" at the apex) and near/far planes press "K" to switch on 
markers and trackball away from the view into order to look back 
at its frustum. Also change "near" and "far" to see how that 
changes the depth planes.

solid picking
~~~~~~~~~~~~~~

#. Clicking pixels with mouse/trackpad, yields an (x,y,z) screen position 
   the z value comes from the depth buffer representing the  position of nearest surface.
#. An unprojection transforms the screen space coordinate into world space.
#. This coordinate is then used to determine the list of solids which 
   contain the point within their bounding box. The solid indices, names and 
   extents in mm are written to the screen.
#. The smallest solid is regarded as "picked". Key "O" toggles high-lighting 
   of picked solids with wireframe spheres.

TODO
^^^^^

Make more use of this, eg to display material/surface properties, 
position in heirarchy 



remote control
~~~~~~~~~~~~~~~

A subset of the commandline launch options can be sent over the network to the running 
application. This allows numerical control of viewpoint and camera parameters.::

   udp.py -t 7000 --eye=10,0,0 --look=0,0,0 --up=0,0,1
   udp.py -t 7000_10,0,0_0,0,0_0,0,1                    # equivalent short form

The viewpoint is defined by the `eye` and `look` positions and the `up` direction, which 
are provided in the coordinate frame of the target solid. NB rotations are performed about the 
look position, that is often set to 0,0,0 corresponding to the center of the solid. 
The "K" key toggle markers indicating the eye and look position. 

The options that are accepted interactively are marked with "I" in the options list::

    daeview.sh --help


raycast launch control
~~~~~~~~~~~~~~~~~~~~~~~~

To avoid GPU panics/system crashes

* subsequent raycast launches are aborted when launch times exceed *max-time* cutoff 
* launch configuration is controlled by eg *launch=3,2,1* and *block=8,8,1* 
  options which configure 2D launch 
 
* raycast launches tyically use 2D pixel thread blocks, 
  some speedups achieved by moving from line of pixels to 2D regions
  in order for the work within a warp of 32 threads to be more uniform 

markers
~~~~~~~~~

Switch on markers with "K", the look point is illustrated with a 
wireframe cube with wireframe sphere inside. Also the frustum of the current view 
excluding any offset trackball rotation + translation and raycast direction/origin
are illustrated.
 
interactive switch to metric pixels presentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The chroma raycast metrics available for display must be defined at 
launch with eg::

   daeviewgl.py --metric time/tri/node  

The restricted flexibility is due to needing to compile
the kernel to change the metric. This is to avoid little 
used branching in the kernel.

Kernel flags can be controlled by remote control, eg::

   udp.py --flags 15,0    # does 15 controls bit shift, here "metric >> 15"  


screen capture
~~~~~~~~~~~~~~~

Pressing "E" will create a screen capture and write a timestamp dated .png 
to the current directory.

movie capture
~~~~~~~~~~~~~~

Not implemented, as find that on OSX can simply use `QuickTime Player.app` 

* `File > New Screen Recording` to create a very large .mov (~1GB for ~2min) 
* `File > Export ...` to compress .mov to .m4v 

orbiting mode
~~~~~~~~~~~~~~

Press "V" to setup a flyaround or orbit mode for the current bookmark view.
Following this setup, pressing "M" will toggle interpolation animation.
During the animation trackball translation/rotation can still be used to 
adjust the effective viewpoint.  The initial "look" direction is tangential, 
so you might need to turn inwards to see the target. Switching to Chroma Raycast
mode can also be used.

TODO: auto switch off animation, when jumping bookmarks to non-interpolatable view.



In Progress
------------

interactive target switching  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While pressing "Q" clicking a solid switches target to that solid.

* currently the default launch eye,look,up parameters are adopted, which is jarring 
* adopting the parameters of the prior view is also jarring 
* need to transform transient/offset eye/look/up to world 

* unclear how best to do this : maybe interpolate views 

Intend:

* switch coordinate frame to adopt that of the target, ie switching view
* changes "look" rotation point to the center of the clicked solid
* allows to raycast for any viewpoint without relying on raycasting 
  being fast enough to be interactive 


Next
-----
 
#. calculate what the trackball translate factor should actually be based on the 
   camera nearsize/farsize, and size of the virtual trackball 
   rather than using adhoc factor

   * will probably need to scale it up anyhow, but would be better not
     to require user tweaking all the time when move between scales

#. take control of lighting, add headlamp (for inside AD)

#. chroma hybrid mode, record propagation progress in VBO, provide 
   OpenGL representation of that 


Ideas
------

#. improve screen text flexibility, columns, matrices, ...

#. clipping plane controls, that are fixed in world coordinates

#. coloring by material

#. animation speed control

   * speed dependant on distance
   * make changing speed not cause jumps in the interpolation




Division of concerns
----------------------

`DAEConfig`
        argument parsing

`DAEGeometry`
        parsing pycollada file and preparing vertex, triangle arrays ready to make VBO

`DAEScene`
        hold state and coordinate

`DAEFrameHandler`
        control of underlying glumpy Frame, presenting graphical view

`DAEInteractivityHandler`
        handle mouse/keyboard inputs and propagate desired actions 

`DAEViewpoint`
         point of view

`DAETrackball`
         rotation and projection, transient offsets from `DAEViewpoint`



"""
import os, sys, logging
log = logging.getLogger(__name__)

import glumpy as gp  

from daeconfig import DAEConfig
from daegeometry import DAEGeometry
from daescene import DAEScene
from daeinteractivityhandler import DAEInteractivityHandler
from daeframehandler import DAEFrameHandler


from env.cuda.cuda_launch import CUDACheck


def main():
    config = DAEConfig(__doc__)
    config.init_parse()
    config.report()
   
    if config.args.cuda_profile: 
        cudacheck = CUDACheck(config)  # MUST be done before pycuda autoinit, for setting of CUDA_PROFILE envvar 
    else:
        cudacheck = None
    config.cudacheck = cudacheck


    geometry = DAEGeometry(config.args.nodes, path=config.args.path)
    geometry.flatten()

    figure = gp.Figure(size=config.size)
    frame = figure.add_frame(size=config.frame)

    scene = DAEScene(geometry, config )

    vbo = geometry.make_vbo(scale=scene.scaled_mode, rgba=config.rgba)
    mesh = gp.graphics.VertexBuffer( vbo.data, vbo.faces )

    frame_handler = DAEFrameHandler( frame, mesh, scene )
    fig_handler = DAEInteractivityHandler(figure, frame_handler, scene, config  )
    frame_handler.fig_handler = fig_handler


    gp.show()



if __name__ == '__main__':
    main()

