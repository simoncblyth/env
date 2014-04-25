#!/usr/bin/env python
"""
DAEVIEWGL
==========

.. seealso:: User instructions :doc:`/geant4/geometry/collada/daeview/daeviewgl_usage`


Next
-----

interactive target switching  **NEEDS OVERHAUL**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


smaller things 
~~~~~~~~~~~~~~~~

#. there is no point proceeding when Chroma is forced into "splitting" 
   geometry due to congested GPU memory. Chroma renders will not succeed. 
   Detect when this happens and assert at launch. 
   Better to do this independantly of chroma with env.cuda.cuda_info 
   and have a configurable minimum required GPU memory free to proceed. 

#. key to reverse animation direction

#. key to temporarily allow about-eye-point rather than about-look-point trackball rotation ? 
 
#. calculate what the trackball translate factor should actually be based on the 
   camera nearsize/farsize, and size of the virtual trackball 
   rather than using adhoc factor

   * will probably need to scale it up anyhow, but would be better not
     to require user tweaking all the time when move between scales

#. take control of lighting, add headlamp (for inside AD)

#. chroma hybrid mode, record propagation progress in VBO, provide 
   OpenGL representation of that 

#. improve screen text flexibility, columns, matrices, ...

#. clipping plane controls, that are fixed in world coordinates

#. coloring by material




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


    geometry = DAEGeometry(config.args.geometry, path=config.path)
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

