#!/usr/bin/env python
"""
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


Interop Refs

Using PrimitiveRestart

#. http://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/225200412


* http://www.opengl.org/wiki/Vertex_Specification_Best_Practices


OpenGL vertex buffer from CUDA 
-------------------------------

* :google:`OpenGL vertex buffer from CUDA`


Motivation
~~~~~~~~~~~~

To avoid buffer recreation for visualization need to get Chroma/CUDA to 
use the OpenGL buffers.

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



   

"""

import logging
log = logging.getLogger(__name__)
import ctypes

# http://pyopengl.sourceforge.net/documentation/deprecations.html
import OpenGL
OpenGL.FORWARD_COMPATIBLE_ONLY = True

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glumpy as gp

from glumpy.graphics.vertex_buffer import VertexBufferException, \
                                          VertexAttribute, \
                                          VertexAttribute_color


class VertexAttribute_position(VertexAttribute):
    def __init__(self, count, gltype, stride, offset):
        assert count > 1, \
            'Vertex attribute must have count of 2, 3 or 4'
        assert gltype in (gl.GL_SHORT, gl.GL_INT, gl.GL_FLOAT, gl.GL_DOUBLE), \
            'Vertex attribute must have signed type larger than byte'
        VertexAttribute.__init__(self, count, gltype, stride, offset)
    def enable(self):
        """
        http://www.opengl.org/sdk/docs/man2/xhtml/glVertexPointer.xml

        ::

             void glVertexPointer(   
                GLint               size,
                GLenum              type,
                GLsizei             stride,
                const GLvoid *      pointer);

        size
             Specifies the number of coordinates per vertex. 
             Must be 2, 3, or 4. The initial value is 4.

        type
             Specifies the data type of each coordinate in the array. 
             Symbolic constants GL_SHORT, GL_INT, GL_FLOAT, or GL_DOUBLE are accepted. 
             The initial value is GL_FLOAT.

        stride
             Specifies the byte offset between consecutive vertices. 
             If stride is 0, the vertices are understood to be tightly packed in the array. 
             The initial value is 0.

        pointer
             Specifies a pointer to the first coordinate of the first vertex in the array. 
             The initial value is 0.


        http://www.opengl.org/sdk/docs/man2/xhtml/glEnableClientState.xml

        ::

           void glEnableClientState(GLenum capability);

        Specifies the capability to enable. Symbolic constants accepted:

        GL_COLOR_ARRAY,
        GL_EDGE_FLAG_ARRAY, 
        GL_FOG_COORD_ARRAY, 
        GL_INDEX_ARRAY, 
        GL_NORMAL_ARRAY,
        GL_SECONDARY_COLOR_ARRAY, 
        GL_TEXTURE_COORD_ARRAY, 
        GL_VERTEX_ARRAY 

        GL_VERTEX_ARRAY
        If enabled, the vertex array is enabled for writing and used during
        rendering when glArrayElement, glDrawArrays, glDrawElements,
        glDrawRangeElements glMultiDrawArrays, or glMultiDrawElements is called. 
        See glVertexPointer.

        """
        gl.glVertexPointer(self.count, self.gltype, self.stride, self.offset)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)



class VertexAttribute_generic(VertexAttribute):
    def __init__(self, count, gltype, stride, offset, index, name, normalized=False):
        assert count in (1, 2, 3, 4), \
            'Generic attributes must have count of 1, 2, 3 or 4'
        VertexAttribute.__init__(self, count, gltype, stride, offset)
        self.index = index
        self.normalized = normalized
        self.name = name

    def __repr__(self):
        return "%s %s %s " % (self.__class__.__name__, self.index, self.name )

    def enable(self):
        """
        http://www.opengl.org/wiki/GLAPI/glVertexAttribPointer
        https://www.khronos.org/opengles/sdk/docs/man/xhtml/glVertexAttribPointer.xml

        ::

            void glVertexAttribPointer( GLuint index,
                                        GLint size,
                                        GLenum type,
                                        GLboolean normalized,
                                        GLsizei stride,
                                        const GLvoid * pointer);
             
        index 
              Specifies the index of the generic vertex attribute to be modified.

        size 
              Specifies the number of components per generic vertex attribute. 
              Must be 1, 2, 3, or 4. The initial value is 4.

        type 
              Specifies the data type of each component in the array. 
              Symbolic constants GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, 
              GL_UNSIGNED_SHORT, GL_FIXED, or GL_FLOAT are accepted. The initial value is GL_FLOAT.

        normalized 
              Specifies whether fixed-point data values should be normalized
              (GL_TRUE) or converted directly as fixed-point values (GL_FALSE) when they are
              accessed.

        stride 
              Specifies the byte offset between consecutive generic vertex attributes.
              If stride is 0, the generic vertex attributes are understood to be tightly
              packed in the array. The initial value is 0.

        pointer 
              Specifies a pointer to the first component of the first generic vertex
              attribute in the array. The initial value is 0.


        """
        #log.info("enable %s " % repr(self) )
        gl.glVertexAttribPointer( self.index, self.count, self.gltype, self.normalized, self.stride, self.offset )
        gl.glEnableVertexAttribArray( self.index )

    def disable(self):
        #log.info("disable %s " % repr(self) )
        gl.glDisableVertexAttribArray( self.index )


class DAEVertexAttributes(object):
    """
    Convert numpy dtype and stride into VertexAttribute instances
    representing the layout of the data in the buffer.
    """ 
    glnames =  ['position', 'color', 'normal', 'tex_coord',
                'fog_coord', 'secondary_color', 'edge_flag']

    gltypes = { 'float32': gl.GL_FLOAT,
                'float'  : gl.GL_DOUBLE, 'float64': gl.GL_DOUBLE,
                'int8'   : gl.GL_BYTE,   'uint8'  : gl.GL_UNSIGNED_BYTE,
                'int16'  : gl.GL_SHORT,  'uint16' : gl.GL_UNSIGNED_SHORT,
                'int32'  : gl.GL_INT,    'uint32' : gl.GL_UNSIGNED_INT }

    def __init__(self, dtype, stride, force_attribute_zero=None ):
        """
        :param dtype:
        :param stride:
        :param force_attribute_zero: name of generic field to slide into slot 0, "vposition" ? 

        Unsure why but nothing appears unless force the attribute holding "position" 
        into attribute 0 
        """
        names = dtype.names or []
        offset = 0
        index = 1 # Generic attribute indices starts at 1

        log.info("%s stride %s dtype %s " % (self.__class__.__name__, stride, repr(dtype) ))

        self.all_  = {}
        self.first = {}
        self.second = {}
        self.generic = []
        self.attmap = { 1:self.all_, 2:self.first, 3:self.second } 

        for name in names:
            if dtype[name].subdtype is not None:
                gtype = str(dtype[name].subdtype[0])
                count = reduce(lambda x,y:x*y, dtype[name].shape)
            else:
                gtype = str(dtype[name])
                count = 1
            pass
            itemsize = dtype[name].itemsize
            if gtype not in self.gltypes.keys():
                raise VertexBufferException('Data type not understood')
            gltype = self.gltypes[gtype]

            log.info("name %s offset %s itemsize %s gtype %s count %s  " % ( name, offset, itemsize, gtype, count )) 

            if name in self.glnames:
                vclass = 'VertexAttribute_%s' % name
                self.all_[name[0]] = eval(vclass)(count,gltype,stride,offset)    # all the vertices
                self.first[name[0]] = eval(vclass)(count,gltype,2*stride,offset) # just first vertex of the pair 
                self.second[name[0]] = eval(vclass)(count,gltype,2*stride,offset+stride)   # just second vertex of the pair
            else:
                if name == force_attribute_zero:
                    attribute = VertexAttribute_generic(count,gltype,stride,offset,0, name)
                else:
                    attribute = VertexAttribute_generic(count,gltype,stride,offset,index, name)
                    index += 1
                pass
                self.generic.append(attribute)
            pass
            offset += itemsize
        pass


    def bind_shader_attrib(self, program=None):
        """
        * https://www.khronos.org/opengles/sdk/docs/man/xhtml/glBindAttribLocation.xml

        Attribute variable name-to-generic attribute index bindings for a
        program object can be explicitly assigned at any time by calling
        glBindAttribLocation. Attribute bindings do not go into effect until
        glLinkProgram is called. After a program object has been linked successfully,
        the index values for generic attributes remain fixed (and their values can be
        queried) until the next link command occurs.

        Applications are not allowed to bind any of the standard OpenGL vertex
        attributes using this command, as they are bound automatically when needed. Any
        attribute binding that occurs after the program object has been linked will not
        take effect until the next time the program object is linked.

        """ 
        for attribute in self.generic:
            attribute.enable()
            if not program is None:
                gl.glBindAttribLocation( program, attribute.index, attribute.name )  # make attrib accessible from shader
                # this is equivalent to layout qualifier in the shader, http://www.opengl.org/wiki/Layout_Qualifier_(GLSL)


    def enable(self, what, att=1, program=None):

        """ 
        """
        self.bind_shader_attrib(program)

        attributes = self.attmap[att]
        for c in attributes.keys():
            if c in what:
                #log.info("enabling attribute %s %s " % (c, attributes[c]) )
                attributes[c].enable()

    def disable(self, what, att=1, program=None):

        for attribute in self.generic:
            attribute.disable()



class DAEVertexBuffer(object):
    """
    Converts numpy vertices and indices arrays into OpenGL buffers, 
    mapping the numpy dtype into OpenGL vertex attributes.

    Three alternate mappings are done to allow draw time control 
    of strides, useful for pairwise vertices eg representing line
    start and end points.

    For example using 2x striding allow to draw points from a lines VBO 
    http://pyopengl.sourceforge.net/documentation/manual-3.0/glBufferData.html
    """
    def __init__(self, vertices, indices=None, force_attribute_zero=None):
        """
        :param vertices: numpy ndarray with named constituents
        :param indices: numpy ndarray of element indices
        """
        self.force_attribute_zero = force_attribute_zero
        self.init_array_buffer(vertices)

        if indices is None:
            indices = np.arange(vertices.size,dtype=np.uint32)
        self.init_element_array_buffer(indices)

    def make_cuda_buffer_object(self, chroma_context):
        return chroma_context.make_cuda_buffer_object(self.vertices_id)

    def init_array_buffer(self, vertices ):
        """
        Upload numpy array into OpenGL array buffer
        """
        self.attrib = DAEVertexAttributes(vertices.dtype, vertices.itemsize, force_attribute_zero=self.force_attribute_zero)
        self.vertices = vertices
        self.vertices_id = gl.glGenBuffers(1)

        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.vertices_id )
        gl.glBufferData( gl.GL_ARRAY_BUFFER, self.vertices, gl.GL_STATIC_DRAW )
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0 )

    def init_element_array_buffer(self, indices ):
        """
        Upload numpy array into OpenGL element array buffer
        """ 
        self.indices_type = gl.GL_UNSIGNED_INT
        self.indices_size = ctypes.sizeof(gl.GLuint) # 4
        self.indices = indices
        self.indices_id = gl.glGenBuffers(1)

        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id )
        gl.glBufferData( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices, gl.GL_STATIC_DRAW )
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, 0 )


    def draw( self, mode=gl.GL_QUADS, what='pnctesf', offset=0, count=None, att=1, shader=None, shader_mode=0 ):
        """ 
        :param mode: primitive to draw
        :param what: attribute multiple choice by first letter
        :param offset: integer element array buffer offset, default 0 (now in units of the indices type)
                       NB this just controls where in the indices array to start getting elements
                       it does not cause offsets with the vertex items
        :param count: number of elements, default None corresponds to all in self.indices
        :param att:  normally 1, when 2 or 3 use alternate stride/offset attributes allowing 
                     things like 2-stepping through VBO via doubled attribute stride and picking which 
                     of pairwise vertices to use.

        :param shader: DAEShader instance

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

        ==================  ==================   ================   =====
        gl***Pointer          GL_***_ARRAY          Att names         *
        ==================  ==================   ================   =====
         Color                COLOR                color              c
         EdgeFlag             EDGE_FLAG            edge_flag          e
         FogCoord             FOG_COORD            fog_coord          f
         Normal               NORMAL               normal             n
         SecondaryColor       SECONDARY_COLOR      secondary_color    s
         TexCoord             TEXTURE_COORD        tex_coord          t 
         Vertex               VERTEX               position           p
         VertexAttrib         N/A             
        ==================  ==================   ================   =====


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

        """
        assert count

        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.vertices_id )
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id )

        program = None if shader is None else shader.shader.program

        gl.glPushClientAttrib( gl.GL_CLIENT_VERTEX_ARRAY_BIT )
        self.attrib.enable( what=what, att=att, program=program )

        if not shader is None:
            shader.link()        # after setting up attribs 
            shader.bind()
            shader.set_param()
            shader.set_mode(shader_mode)

        gl.glDrawElements( mode, count, self.indices_type, ctypes.c_void_p(self.indices_size*offset) )

        self.attrib.disable( what=what, att=att, program=program )
        gl.glPopClientAttrib( )

        if not shader is None:
            shader.unbind()


        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, 0 ) 
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0 ) 



if __name__ == '__main__':
    pass




