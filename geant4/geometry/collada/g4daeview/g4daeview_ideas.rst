G4DAEView Ideas
===============

JSON State vending
--------------------

It would be useful for g4daeview.py to vend some JSON regarding current high level state
over ZMQ or UDP ? EG for interoperation with an ipython session.

* are communicating JSON metadata over ZMQ already, but this only flows with the 
  photon propagation request/response


Tight zooming not easy
-------------------------------------

Tight zooming (eg viewing steps propagating thru the acrylic surfaces) 
is difficult as bookmarking cause view jumps. 

Attempts to do rotation interpolation with no geometry visible cause NaN fails
---------------------------------------------------------------------------------


During VBO structural changes
------------------------------

Avoid VBO comparison assertions using `--wipepropagate` to delete any 
preexisting persisted propagation files, rather that compare against them 
and assert on differences::

    g4daeview.sh --with-chroma --load 1 --debugkernel --wipepropagate

Some rare discrepancies in material codes have been observered
between propagations see :doc:`propagated_flags_mismatch`


Live Style Change issues FIXED
---------------------------------

Formerly could not make style transitions in some directions. In particular
styles that use geometry shaders (noodles, movie) cannot be transitioned to from
styles that do not. Actually the transition happens without error but a blank render results.

This was fixed by moving to keeping GLSL shaders around and swapping between them 
rather than attempting to delete shaders.

Step Interpolation Visualization IMPLEMENTED
-----------------------------------------------

Keeping multiple propagation steps in the VBO at once provides an interesting
visualization possibility.  

#. Run the propagation at event load to a fixed maximum number of steps 

   * in propagate_vbo.cu store the step results into the vbo in slots by step number
   * within the pre-draw DAEPhotonsRenderer/DAEPhotonsKernel CUDA call calculate 
     photon parameters based on an input "time" constant (that can be varied 
     forwards and backwards interactively) 

     * find the pair of steps whose times straddle the input time and 
       use linear interpolation to calculate the photon position at that time, 
       lay down the result into a slot in the VBO

     * invoke geometry shader rendereing as normal with appropriate stride to 
       pickup the dynamically calculated photon parameters    

This way you have a "real" time view of simulation progression rather than
the very artificicial view of discrete steps.

What about non-Chroma devices ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chroma needs CUDA. 
To visualize on devices without Chroma/CUDA but 
with OpenCL (or Metal):

* provide way to pullback/persist/transport VBO, without ROOT dependency.

Dumb (non-compute) devices will not be able to do the interpolation, 
but they could do the fake glDrawElements time cutting via changing the count. 


Geometry Shaders
~~~~~~~~~~~~~~~~~~

Current rendering requires geometry shading capability, that is 
not available on OpenGL ES would need to construct line strip VBO 
rather than generating lines.

Even OpenGL ES Next (expected released 2014) will not have geometry
shaders.

* http://www.anandtech.com/show/7657/khronos-offers-a-quick-peek-at-the-next-version-of-opengl-es


Node selection only applies to to particular geometry dyb/juno/lxe
--------------------------------------------------------------------

Current workaround is via envvars like DAE_GEOMETRY_DYB.  
That does not scale so well, probably need to adopt defaults files maybe:

* `~/.g4daeview/defaults.cfg`  
* `~/.g4daeview/dyb/defaults.cfg`  
* `~/.g4daeview/juno/defaults.cfg`  

Pulled out the default geometry nodeselection, but not geo specific starting eye, near etc..::

    g4daeview.sh -p lxe -C


Live Updating Test
--------------------


worker
~~~~~~~~

Start the "worker" with ZMQ tunnel node specified as the SSH config "alias" of the node
on which the broker is running

::

   g4daeview.sh --zmqtunnelnode=N       # starts up within a few seconds


client
~~~~~~~

Start the "client" on N::

   csa.sh    # takes several minutes to get going, currently only 100 events

The "worker" can be stopped and started whilst the "client" runs and 
new live ChromaPhotonList are presented as they are simulated and ZMQ
transported. Use auto-created bookmark 9 to find them.

The Segmentation Violation issue due to changing GLUT menus whilst 
they are in use appears solved by the improved menu structuring 
and using a pending update slot. However suspect that menu entries 
from multiple events may appear together in some difficult to 
reproduce circumstances.


Next
-----

improved structuring
~~~~~~~~~~~~~~~~~~~~~~

Should look into adopting some design patterns to make the code
more tractable, things are getting unweildy eg Menu-ing.

* http://www.songho.ca/opengl/gl_mvc.html

Histogramming 
~~~~~~~~~~~~~~~

* GPU histogramming, eg photon wavelength spectrum
* how to present ?

  * dump numpy arrays for separate presentation
  * separate OpenGL window, would allow live updating during propagation 
  * communicate to a separate ipython session over ZMQ ? and use matplotlib from there 

    * i like the compartmentalization 
  
issues
^^^^^^^^

* fmcpmuon.py refers to volumes with DE names like `/dd/Structure/Pool/db-ows`  geometry 
  nodes available via daenode are all `/dd/Geometry/..`


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
   investigate getting them to share the same arrays (similar to whats done with pixels and now photon steps)

interactive target switching  **NEEDS OVERHAUL**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While pressing "Q" clicking a solid switches target to that solid.

* currently the default launch eye,look,up parameters are adopted, which is jarring 
* adopting the parameters of the prior view is also jarring 
* need to transform transient/offset eye/look/up to world 

* unclear how best to do this : maybe interpolate views 


Using bookmarks is a better way, but that often makes the 
target disappear due to near cull. Perhaps need to auto-adjust 
near.


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

#. coloring by material




