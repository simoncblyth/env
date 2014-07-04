DAEVertexBuffer Dev Notes
===========================

Interop
--------

Motivation:

    To avoid buffer recreation for visualization need to get Chroma/CUDA to 
    use the OpenGL buffers.

Interop steps gleaned from raycaster PBO usage:

#. import pycuda.gl as cuda_gl                  # special GL enabled CUDA context  (handle in DAEChromaContext)
#. cuda_pbo = cuda_gl.BufferObject(long(pbo))   # needs CUDA context, pbo is OpenGL buffer id
#. cuda_pbo.unregister()                        # ends CUDA responsibility, allows OpenGL access for drawing 

#. pbo_mapping = cuda_pbo.map()                 # CUDA takes responsibility 
#. pbo_device_ptr = pbo_mapping.device_ptr()    # pointer to device memory to pass as kernel call argument 
#. pbo_mapping.unmap()                          # after the kernel call, declares end of CUDA usage

Currently are using the simple and inefficient technique 
of recreating the VBOs whenever need to change anything.

Ruminations

#. having separate VBOs for drawing lines and points is inconvenient, especially 
   when have CUDA in the mix

   * should be able to draw points from the lines VBO via indices control
     (ie have one vertex array and which is used with two different indices arrays)
    
   * how about flag control photon selections ? glumpy VertexBuffer glues together
     vertices and indices as pair of ctor arguments : they do not need to go together.

     ie could split that off and do the indices selection on CPU in similar manner
     to current ? But what about on GPU selection, it just needs bit field comparison to 
     uniform argument mask or history bitfield ... hmm but still would need to 
     pull out the selection bits CPU side to setup the OpenGL indices ? So little gain.
     Does OpenGL have some concept of a mask array ?

     Can control color in the kernel, so make unselected invisible (set alpha to 0)

     * :google:`OpenGL CUDA vertex selection`
     * http://www.opengl.org/wiki/Vertex_Rendering
 
   * 2nd point of line pair can be calculated in the kernel following CUDA propagation, 
     with line length being a uniform
     NB need to keep the slot, initially fill it from numpy for the non-CUDA installs 
 
#. need to retain drawing of initial photon positions/directions for non-chroma installs 


Too Many Moving parts
~~~~~~~~~~~~~~~~~~~~~~

#. OpenGL vertex attributes description of initial numpy ndarray
#. shader attribute binding to access them from shaders
#. what is PyCUDA GPUArray actually doing, as using OpenGL buffers
   direct from PyCUDA replaces this 
#. visualization gymnastics should not substantially impact 
   without-vis propagation 

AoS or SoA
~~~~~~~~~~~~

* http://stackoverflow.com/questions/17924705/structure-of-arrays-vs-array-of-structures-in-cuda

Data flow
~~~~~~~~~~~

#. ROOT deserializes the ChromaPhotonList bytes arriving from file or ZMQ into a ChromaPhotonList 
   instance (a collection std::vector<float> and std::vector<int>) 

#. copy `pos`, `dir`, `wavelength`, `t` etc into numpy arrays inside a chroma.event.Photons instance

#. copy subset of those arrays into "pdata" numpy ndarray  

The underlying data is coming from a numpy ndarray. Perhaps pycuda has 
solved the problem already ? http://documen.tician.de/pycuda/array.html
Maybe, but for interop need to make pycuda use the OpenGL buffers.


What needs to be shared between CUDA and OpenGL ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. position
#. momdir (for lines)
#. wavelength
#. propagation flags, for selection/history 



Single VBO approach (array-of-structs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This approach bangs into CUDA alignment/padding complexities device side and arriving 
at the struct that matches the OpenGL buffer layout.

* http://www.igorsevo.com/Article.aspx?article=Million+particles+in+CUDA+and+OpenGL
* http://on-demand.gputechconf.com/gtc/2013/presentations/S3477-GPGPU-In-Film-Production.pdf

::

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, 16, 0);
    glColorPointer(4,GL_UNSIGNED_BYTE,16,(GLvoid*)12);

In CUDA stuffs the colors into a float using a union::

    union Color
    {
        float c;
        uchar4 components;
    };

    Color temp;
    temp.components = make_uchar4(128/length,(int)(255/(velocity*51)),255,10);
    pos[y*width+x].w = temp.c;


Ahha, users of OpenGL compute shaders face the same issues

* http://stackoverflow.com/questions/21342814/rendering-data-in-opengl-vertices-and-compute-shaders
* http://www.opengl.org/registry/doc/glspec43.core.20130214.pdf

  *  7.6.2.2 - Standard Uniform Block Layout 



NVIDIA Example
~~~~~~~~~~~~~~~~

* /usr/local/env/cuda/NVIDIA_CUDA-5.5_Samples/2_Graphics/simpleGL


Targetted googling
~~~~~~~~~~~~~~~~~~~~~

* :google:`cuda kernel float4* VBO`

andyswarm
^^^^^^^^^^^

#. color and position both as float4 with colors offset after position
#. Advantage is can use `float4 *dptr` just like simpleGL example.

* http://www.evl.uic.edu/aej/525/code/andySwarm.cu
* http://www.evl.uic.edu/aej/525/code/andySwarm_kernel.cu

::

     // render from the vbo
     glBindBuffer(GL_ARRAY_BUFFER, vbo);
     glVertexPointer(4, GL_FLOAT, 0, 0);
     glColorPointer(4, GL_FLOAT, 0, (GLvoid *) (mesh_width * mesh_height * sizeof(float)*4));


Separate VBO approach (struct-of-arrays)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This approach avoids the struct problems at expense of high level
bookkeeping for the multiple VBOs. Potentially an OpenGL draw performance hit 
too.


* http://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/225200412?pgno=6

Example uses separate VBOs for position and color and does 
manual linear addressing to change them from CUDA. 
Then OpenGL draws by binding to the multiple different VBO.

This is nice and simple at expense of lots of VBOs 

::

    __global__ void kernel(float4* pos, uchar4 *colorPos,
               unsigned int width, unsigned int height, float time)
    {
        unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
        ...

        // write output vertex
        pos[y*width+x] = make_float4(u, w, v, 1.0f);
        colorPos[y*width+x].w = 0;
        colorPos[y*width+x].x = 255.f *0.5*(1.f+sinf(w+x));
        colorPos[y*width+x].y = 255.f *0.5*(1.f+sinf(x)*cosf(y));
        colorPos[y*width+x].z = 255.f *0.5*(1.f+sinf(w+time/10.f));
    }

The splitting between arrays is done at glBindBuffer::

    void renderCuda(int drawMode)
    {
      glBindBuffer(GL_ARRAY_BUFFER, vertexVBO.vbo);
      glVertexPointer(4, GL_FLOAT, 0, 0);
      glEnableClientState(GL_VERTEX_ARRAY);
       
      glBindBuffer(GL_ARRAY_BUFFER, colorVBO.vbo);
      glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
      glEnableClientState(GL_COLOR_ARRAY);
     



glBindBuffer
~~~~~~~~~~~~~~

* http://www.khronos.org/opengles/sdk/docs/man/xhtml/glBindBuffer.xml

glBindBuffer lets you create or use a named buffer object. Calling glBindBuffer
with target set to GL_ARRAY_BUFFER or GL_ELEMENT_ARRAY_BUFFER and buffer set to
the name of the new buffer object binds the buffer object name to the target.
When a buffer object is bound to a target, the previous binding for that target
is automatically broken.

When vertex array pointer state is changed by a call to glVertexAttribPointer,
the current buffer object binding (GL_ARRAY_BUFFER_BINDING) is copied into the
corresponding client state for the vertex attrib array being changed, one of
the indexed GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDINGs. While a non-zero buffer
object is bound to the GL_ARRAY_BUFFER target, the vertex array pointer
parameter that is traditionally interpreted as a pointer to client-side memory
is instead interpreted as an offset within the buffer object measured in basic
machine units.




OpenGL Drawing Techniques
---------------------------


glDrawElements
~~~~~~~~~~~~~~~   


Buffer offset default of 0 corresponds to glumpy original None, (ie (void*)0 )
the integer value is converted with `ctypes.c_void_p(offset)`   
allowing partial buffer drawing.

* http://pyopengl.sourceforge.net/documentation/manual-3.0/glDrawElements.html
* http://stackoverflow.com/questions/11132716/how-to-specify-buffer-offset-with-pyopengl
* http://pyopengl.sourceforge.net/documentation/pydoc/OpenGL.arrays.vbo.html
* http://www.opengl.org/discussion_boards/showthread.php/151386-VBO-BUFFER_OFFSET-and-glDrawElements-broken

A C example of glDrawElements from /Developer/NVIDIA/CUDA-5.5/samples/5_Simulations/smokeParticles/SmokeRenderer.cpp::

     glDrawElements(GL_POINTS, count, GL_UNSIGNED_INT, (void *)(start*sizeof(unsigned int)));    # start is an int 


====================  ==============
type
====================  ==============
GL_UNSIGNED_BYTE        0:255
GL_UNSIGNED_SHORT,      0:65535
GL_UNSIGNED_INT         0:4.295B
====================  ==============

===================   ====================================
   mode 
===================   ====================================
  GL_POINTS
  GL_LINE_STRIP
  GL_LINE_LOOP
  GL_LINES
  GL_TRIANGLE_STRIP
  GL_TRIANGLE_FAN
  GL_TRIANGLES
  GL_QUAD_STRIP
  GL_QUADS
  GL_POLYGON
===================   ====================================


The what letters, 'pnctesf' define the meaning of the arrays via 
enabling appropriate attributes.

==================  ==================   ==================   =====
gl***Pointer          GL_***_ARRAY          Att names         *
==================  ==================   ==================   =====
 Color                COLOR                color              c
 EdgeFlag             EDGE_FLAG            edge_flag          e
 FogCoord             FOG_COORD            fog_coord          f
 Normal               NORMAL               normal             n
 SecondaryColor       SECONDARY_COLOR      secondary_color    s
 TexCoord             TEXTURE_COORD        tex_coord          t 
 Vertex               VERTEX               position           p
 VertexAttrib         N/A             
==================  ==================   ==================   =====


glDrawElements offset
~~~~~~~~~~~~~~~~~~~~~~~~

#. **glDrawElements offset applies to the entire indices array**, 

   * ie it controls where to start getting indices from.
   * for offsets within each element have to use VertexAttrib offsets.

glPushClientAttrib
~~~~~~~~~~~~~~~~~~~

http://www.opengl.org/sdk/docs/man2/xhtml/glPushClientAttrib.xml

::

     void glPushClientAttrib(GLbitfield mask); 

glPushClientAttrib takes one argument, a mask that indicates which groups of
client-state variables to save on the client attribute stack. Symbolic
constants are used to set bits in the mask. mask is typically constructed by
specifying the bitwise-or of several of these constants together. The special
mask GL_CLIENT_ALL_ATTRIB_BITS can be used to save all stackable client state.

The symbolic mask constants and their associated GL client state are as follows
(the second column lists which attributes are saved):

GL_CLIENT_PIXEL_STORE_BIT   Pixel storage modes 
GL_CLIENT_VERTEX_ARRAY_BIT  Vertex arrays (and enables)


glMultiDrawElements
~~~~~~~~~~~~~~~~~~~~

* http://stackoverflow.com/questions/11286964/glmultidrawelements-values-to-pass-to-indices-parameter-glvoid

* http://stackoverflow.com/questions/5354710/drawing-polygons-with-varying-vertex-count-in-opengl-es

  * not in OpenGL ES 3.0 (highest in iOS so far), is there an alternate that avoids looping over draw calls ?

* https://www.opengl.org/discussion_boards/showthread.php/176891-glMultiDrawElements-VBO
  
  * MultiDraw with VBO
  * indices and counts arrays are client side, but the indices array holds byte offsets into device side buffer




Read OpenGL buffer back into numpy arrays
-------------------------------------------


::

    def read_array_buffer_0(self):
        """
        * http://www.opengl.org/sdk/docs/man2/xhtml/glMapBuffer.xml
        * http://mail.scipy.org/pipermail/numpy-discussion/2008-September/037131.html

        ctypes.sizeof(ctypes.c_float) == 4

        numpy.ctypeslib.as_array

        Writing into a mapped OpenGL VBO without pyopengl sugar ?

        * http://comments.gmane.org/gmane.comp.python.opengl.user/2069

        ::

             # Map the buffer object to a pointer
             vbo_pointer = ctypes.cast(gl.glMapBuffer(gl.GL_ARRAY_BUFFER, gl.GL_WRITE_ONLY), ctypes.POINTER(ctypes.c_ubyte))
             vbo_array = numpy.ctypeslib.as_array(vbo_pointer, (buffer_size,))  # numpy array from pointer 
             vbo_array[0:data_size_to_copy] = data.view(dtype='uint8').ravel()
             gl.glUnmapBuffer(gl.GL_ARRAY_BUFFER)

        ::

             data = ctypes.string_at(gl.glMapBuffer(gl.GL_ARRAY_BUFFER, gl.GL_READ_ONLY), self.vertices.nbytes )
             x = np.frombuffer(int_asbuffer(ctypes.addressof(data.contents), n*8))


        * http://stackoverflow.com/questions/7543675/how-to-convert-pointer-to-c-array-to-python-array


        """
        pass
        log.info("read_array_buffer")

        nbytes = self.vertices_copy.nbytes 
        dtype = np.dtype(np.float32)
        itemsize = dtype.itemsize
        count = nbytes//itemsize   # integer 
        assert nbytes == count * itemsize 

        ptr = gl.glMapBuffer(gl.GL_ARRAY_BUFFER, gl.GL_READ_ONLY) 

        ArrayType = ctypes.c_float * count
        ap = ctypes.cast(y, ctypes.POINTER(ArrayType))

        #vbo_data = ctypes.string_at(ptr, nbytes )
        #x = np.frombuffer(int_asbuffer(ctypes.addressof(vbo_data),nbytes),dtype=dtype)
        #print x  

        gl.glUnmapBuffer(gl.GL_ARRAY_BUFFER)

        #buffer_size = self.vertices_copy.nbytes
        #vbo_pointer = ctypes.cast(gl.glMapBuffer(gl.GL_ARRAY_BUFFER, gl.GL_READ_ONLY), ctypes.POINTER(ctypes.c_ubyte))
        #vbo_array = np.ctypeslib.as_array(vbo_pointer, (buffer_size,))  # numpy array from pointer 
        #vbo_array[0:data_size_to_copy] = data.view(dtype='uint8').ravel()
        #gl.glUnmapBuffer(gl.GL_ARRAY_BUFFER)

    def read_array_buffer( self ):
        """
        * :google:`glMapBuffer numpy`

          * http://blog.vrplumber.com/b/2009/09/01/hack-to-map-vertex/

        Map the given buffer into a numpy array...
        """
        log.info("read_array_buffer")
        target = gl.GL_ARRAY_BUFFER
        access = gl.GL_READ_ONLY
        nbytes = self.vertices.nbytes 

        func = ctypes.pythonapi.PyBuffer_FromMemory
        func.restype = ctypes.py_object

        ptr = gl.glMapBuffer( target, access )

        buf = func( ctypes.c_void_p(ptr), nbytes )
        arr = np.frombuffer( buf, 'B' )

        self.propagated = arr.view(dtype=self.vertices.dtype)

        gl.glUnmapBuffer(target)
        








