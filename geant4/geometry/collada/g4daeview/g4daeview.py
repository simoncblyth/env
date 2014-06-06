#!/usr/bin/env python
"""
G4DAEVIEW
==========

.. seealso:: User instructions :doc:`/geant4/geometry/collada/g4daeview/g4daeview_usage`


NB default node selection only applies to to dyb
-------------------------------------------------

For viewing Geant4 LXe example::

    g4daeview.sh -p lxe -g 1: 


Live Updating Test
--------------------

#. start the "worker" with ZMQ tunnel node specified as the SSH config "alias" of the node 
   on which the broker is running::

   g4daeview.sh --zmqtunnelnode=N       # starts up within a few seconds

#. start the "client" on N::
 
   csa.sh    # takes several minutes to get going, currently only 100 events

   
The "worker" can be stopped and started whilst the "client" runs and 
new live ChromaPhotonList are presented as they are simulated and ZMQ
transported. Use auto-created bookmark 9 to find them.

The Segmentation Violation issue due to changing GLUT menus whilst 
they are in use appears solved by the improved menu structuring 
and using a pending update slot. However suspect that menu entries 
from multiple events may appear together in some difficult to 
reproduce circumstances.


Observe
--------

#. sometimes "reload" not working, although stepping does


Next
-----

load menu assertion
~~~~~~~~~~~~~~~~~~~~

#. loading an event from file at launch `--load 1` trips an assertion, as DAEPhoton 
   menus are attempting to be changed before being created in first place

   * symptom of DAEPhotons doing too much, split menu creation elsewhere 


improved structuring
~~~~~~~~~~~~~~~~~~~~~~

Should look into adopting some design patterns to make the code
more tractable, things are getting unweildy eg Menu-ing.



Histogramming 
~~~~~~~~~~~~~~~

* GPU histogramming, eg photon wavelength spectrum
* how to present ?

  * dump numpy arrays for separate presentation
  * separate OpenGL window, would allow live updating during propagation 
  

Reemission Wavelength Debugging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* material property readout interface ?


Efficient Photon Drawing 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. efficent GPU resident (and shared with Chroma) CPL handling, to handle 1000x the number of photons

   * current recreate VBO all the time approach might not not be workable
   * look into OpenGL PointSet or volume field representations of photon clouds

#. basic sticking point is to get PyCUDA/Chroma propagation to work with OpenGL created VBO, 
   in the same way as pixel arrays were used from raycasting 

issues
^^^^^^^^

* fmcpmuon.py refers to volumes with DE names like `/dd/Structure/Pool/db-ows`  geometry 
  nodes available via daenode are all `/dd/Geometry/..`



OpenGL PointSet, Volume Field, Particle System representations 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Speculative, probably need more control that these approaches allow.


Photon provenance
~~~~~~~~~~~~~~~~~~~

Use the currently unused CPL pmtid as a bitfield for provenance info

#. process: Cerenkov, Scintillation
#. some compressed generation tree info  (full parentID, trackID not needed ) perhaps
 
   * if parent is primary (trackID 1)
   * PDG code of parent, look at possibilities counts and compress into 2-3 bits 

#. need some explorations to arrive at an appropriately terse sketch of the tree 


ChromaOtherList for transport over ZMQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It would be interesting for debugging to visualize other tracks, not just optical photons.
Do some counting to see how complete want to go, and arrive at a terse representation
analogous to CPL.

Geant4 Chroma Process
~~~~~~~~~~~~~~~~~~~~~~

Geant4 frowns on changing track in anything other than a process. So look into 
when and how a process gets access to OPs compared to a stack action.

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

#. improve clipping plane (W) feature:

   * switch planes on/off
   * make persistent, stored with bookmarks

#. coloring by material


GPU Out-of-memory during BVH construction with full Juno geometry (50M nodes?)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maybe can reorganize the work to avoid using too much memory ?::

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
from daemenu import DAEMenu, DAEMenuGLUT

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

    rmenu_glut = DAEMenuGLUT()
    rmenu = DAEMenu("rtop", backend=rmenu_glut)
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

    rmenu_glut.setup_glutMenuStatusFunc()

    scene = DAEScene(geometry, config )

    vbo = geometry.make_vbo(scale=scene.scaled_mode, rgba=config.rgba)
    mesh = gp.graphics.VertexBuffer( vbo.data, vbo.faces )

    frame_handler = DAEFrameHandler( frame, mesh, scene )
    fig_handler = DAEInteractivityHandler(figure, frame_handler, scene, config  )
    frame_handler.fig_handler = fig_handler

    rmenu.push_handlers(fig_handler)   # so events from rmenu such as on_needs_redraw are routed to the fig_handler
    rmenu_glut.create(rmenu, "RIGHT")

    gp.show()



if __name__ == '__main__':
    main()

