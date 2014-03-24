OpenGL CUDA Interop
=====================

Objectives
-----------

#. Present OpenGL projected geometry (probably wireframes) on top of PyCUDA/Chroma ray-traced images
#. visualize Chroma/CUDA calculated photon propagations
#. sharing to avoid duplication in GPU memory 

What can be shared ?
---------------------

* vertex and face arrays
* pixel array results from Chroma/CUDA ray tracing
* photon propagation position histories 


GLUT Overlay
-------------

* http://pyopengl.sourceforge.net/documentation/manual-3.0/glutEstablishOverlay.html

GLSL shader texture overlay
----------------------------

* http://stackoverflow.com/questions/12218419/overlay-an-image-over-video-using-opengl-es-shaders
* http://www.clockworkcoders.com/oglsl/tutorial8.htm
* http://antongerdelan.net/opengl/overlays.html

VBO level interop
---------------------

* http://www.icp.uni-stuttgart.de/~icp/CUDA_examples

Ideal gas with direct OpenGL visualization, uses the CUDA-OpenGL-binding to render an ideal gas in a rotating box. 
An OpenGL vertex buffer is written directly from CUDA, which runs the ideal gas as a very simple kernel.

::

    simon:t blyth$ curl -L -O http://www.icp.uni-stuttgart.de/~icp/mediawiki/images/b/bf/PyGL.tar.gz
    simon:t blyth$ tar ztvf PyGL.tar.gz 
    drwxr-xr-x arnolda/icp       0 2011-02-08 01:09:50 pygl/
    -rw-r--r-- arnolda/icp    5359 2011-02-08 01:03:54 pygl/pygl.py
    -rw-r--r-- arnolda/icp   35147 2011-02-08 01:08:11 pygl/COPYING
    -rw-r--r-- arnolda/icp     324 2011-02-08 01:09:44 pygl/README

CUDA calculates point positions, from pygl.py::

    057     def GLInit(self):
    ...
    106         self.gas_kernel = gpu_code.get_function("gas_kernel")
    107         self.gas_kernel.prepare("PPPi", (512,1,1))
    108    
    109         # create vbo buffer
    110         self.vbo = glGenBuffers(1)
    111         glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
    112         glBufferData(GL_ARRAY_BUFFER,
    113                      np.random.random(4*n_particles).astype(np.float32), GL_DYNAMIC_DRAW)
    114         glBindBuffer(GL_ARRAY_BUFFER, 0)
    115         # CudaGL interop for the VBO
    116         self.resource = cudagl.BufferObject(int(self.vbo))
    117 
    118     def calculate_vertices(self):
    119         # map 1 VBO given by vbo_resource, synchronize with stream 0
    120         map = self.resource.map()
    121         g_vbo = map.device_ptr()
    122         self.gas_kernel.prepared_call(
    123             ((n_particles+511)/512,1),
    124             g_vbo, self.g_positions, self.g_velocities, np.intc(n_particles))
    125         # unmap again
    126         map.unmap()
    127
    128     def GLDraw(self):
    129         self.sim_time += 0.1
    130 
    131         self.calculate_vertices()
    ...
    ...         OpenGL setup matrices etc.. 
    ...
    142         # render vbo as buffer of points
    143         glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
    144         glVertexPointer(4, GL_FLOAT, 0, None)
    145 
    146         glEnableClientState(GL_VERTEX_ARRAY)
    147         glColor3f(0.0, 0.0, 1.0)
    148         glDrawArrays(GL_POINTS, 0, n_particles)
    149         glDisableClientState(GL_VERTEX_ARRAY)
    150 



In summary to get from an OpenGL VBO name to a CUDA usable pointer into the mapped buffer::

     map = cudagl.BufferObject(int(self.vbo)).map()
     g_vbo = map.device_ptr()     
     # now pass that as parameter to the CUDA prepared kernel call
     map.unmap()    # presumably says that are done with it, allowing OpenGL to resume control 


Adapt to visualize Chroma photons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

This looks to be a good way for visualizing simulated photons, but not ray traced pixels.




