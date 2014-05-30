#!/usr/bin/env python
"""
G4DAEVIEW
==========

.. seealso:: User instructions :doc:`/geant4/geometry/collada/g4daeview/g4daeview_usage`


NB default node selection only applies to to dyb
-------------------------------------------------

For viewing Geant4 LXe example::

    g4daeview.sh -p lxe -g 1: 


Observe
--------

#. sometimes "reload" not working, although stepping does


Photon History Flags
---------------------

::

    NO_HIT                                     1   0x1      
    BULK_ABSORB                                2   0x2      
    SURFACE_DETECT                             4   0x4      
    SURFACE_ABSORB                             8   0x8      
    RAYLEIGH_SCATTER                          16   0x10      
    REFLECT_DIFFUSE                           32   0x20      
    REFLECT_SPECULAR                          64   0x40      
    SURFACE_REEMIT                           128   0x80      
    SURFACE_TRANSMIT                         256   0x100      
    BULK_REEMIT                              512   0x200      
    NAN_ABORT                         2147483648   0x80000000      


Next
-----


Menu Live Updating Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

Live updating the displayed event arriving via ZMQ (using ssh tunnel to thwart the network gnomes) 
leads to messed up partial glut menus and results in segv::

    2014-05-30 20:15:18,890 env.geant4.geometry.collada.g4daeview.daephotons:120 nflag 56 unique flag combinations len(history) 1 
    2014-05-30 20:15:18,890 env.geant4.geometry.collada.g4daeview.daemenu:163 not calling glut.glutSetMenu as menu is None 
    2014-05-30 20:15:18.890 python[99235:d07] GLUT Warning: The following is a new check for GLUT 3.0; update your code.
    2014-05-30 20:15:18.890 python[99235:d07] GLUT Fatal Error: menu manipulation not allowed while menus in use.
    /Users/blyth/env/bin/g4daeview.sh: line 60: 99235 Segmentation fault: 11  g4daeview.py $*
     
Need some protections against menu in use ? And debugging menu creation.



ChromaPhotonList Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. efficent GPU resident (and shared with Chroma) CPL handling, to handle 1000x the number of photons

   * current recreate VBO all the time approach might not not be workable
   * look into OpenGL PointSet or volume field representations of photon clouds

#. other particle communication and representation, primaries 
#. add axes, for checking whilst testing with Geant4 particle guns   

issues
^^^^^^^^

* fmcpmuon.py refers to volumes with DE names like `/dd/Structure/Pool/db-ows`  geometry 
  nodes available via daenode are all `/dd/Geometry/..`


Avoid Geometry Duplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. currently geometry info is on GPU twice, once for OpenGL once for Chroma
   investigate getting them to share the same arrays (similar to whats done with pixels)


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


GPU Out-of-memory during BVH construction with full Juno geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    g4daeview.sh -p juno --with-chroma


    074 2014-05-26 10:51:22,642 env.geant4.geometry.collada.collada_to_chroma:297 ColladaToChroma adding BVH
    075 2014-05-26 10:51:23,879 chroma.loader       :155 Building new BVH using recursive grid algorithm.
    076 Expanding 422925 parent nodes
    077 Merging 50232688 nodes to 15826587 parents
    078 Expanding 48194 parent nodes
    079 Merging 16250194 nodes to 4923755 parents
    080 Merging 4971964 nodes to 1289438 parents
    081 Merging 1289438 nodes to 266462 parents
    082 Merging 266462 nodes to 51806 parents
    083 Merging 51806 nodes to 10332 parents
    084 Merging 10332 nodes to 2216 parents
    085 Merging 2216 nodes to 480 parents
    086 Merging 480 nodes to 104 parents
    087 Merging 104 nodes to 32 parents
    088 Merging 32 nodes to 8 parents
    089 Merging 8 nodes to 2 parents
    090 Merging 2 nodes to 1 parent
    091 Traceback (most recent call last):
    092   File "/Users/blyth/env/bin/g4daeview.py", line 4, in <module>
    093     main()
    094   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/g4daeview.py", line 186, in main
    095     scene = DAEScene(geometry, config )
    096   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daescene.py", line 45, in __init__
    097     chroma_geometry = geometry.make_chroma_geometry()
    098   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daegeometry.py", line 315, in make_chroma_ge    ometry
    099     cc.convert_geometry(nodes=self.nodes())
    100   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/collada_to_chroma.py", line 291, in convert_geometry
    101     self.add_bvh()
    102   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/collada_to_chroma.py", line 304, in add_bvh
    103     cuda_device=cuda_device)
    104   File "/usr/local/env/chroma_env/src/chroma/chroma/loader.py", line 160, in load_bvh
    105     bvh = make_recursive_grid_bvh(geometry.mesh, target_degree=3)
    106   File "/usr/local/env/chroma_env/src/chroma/chroma/bvh/grid.py", line 91, in make_recursive_grid_bvh
    107     nodes, layer_bounds = concatenate_layers(layers)
    108   File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/bvh.py", line 266, in concatenate_layers
    109     grid=(nblocks_this_iter,1))
    110   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 355, in function_call
    111     handlers, arg_buf = _build_arg_buf(args)
    112   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 125, in _build_arg_buf
    113     arg_data.append(int(arg.get_device_alloc()))
    114   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 59, in get_device_alloc
    115     self.dev_alloc = mem_alloc_like(self.array)
    116   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 608, in mem_alloc_like
    117     return mem_alloc(ary.nbytes)
    118 pycuda._driver.MemoryError: cuMemAlloc failed: out of memory


Restricting to about half the geometry succeeds::

     g4daeview.sh -p juno -g 1:25000 --with-chroma --launch 3,3,1




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
from daemenu import DAEMenu

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

    rmenu = DAEMenu("rtop")
    config.rmenu = rmenu


    geocachepath = config.geocachepath
    if os.path.exists(geocachepath) and config.args.geocache:
        geometry = DAEGeometry.load_from_cache( config )
    else:
        geometry = DAEGeometry(config.args.geometry, config)
        geometry.flatten()
        if config.args.geocache:
            geometry.save_to_cache(geocachepath)
        pass
    pass

    figure = gp.Figure(size=config.size)
    frame = figure.add_frame(size=config.frame)

    scene = DAEScene(geometry, config )

    vbo = geometry.make_vbo(scale=scene.scaled_mode, rgba=config.rgba)
    mesh = gp.graphics.VertexBuffer( vbo.data, vbo.faces )

    frame_handler = DAEFrameHandler( frame, mesh, scene )
    fig_handler = DAEInteractivityHandler(figure, frame_handler, scene, config  )
    frame_handler.fig_handler = fig_handler

    rmenu.push_handlers(fig_handler)   # so events from rmenu such as on_needs_redraw are routed to the fig_handler


    rmenu.create("RIGHT")

    gp.show()



if __name__ == '__main__':
    main()

