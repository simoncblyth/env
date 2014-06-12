#!/usr/bin/env python
"""
For development nodes see daevertexbuffer.rst
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



        http://www.opengl.org/sdk/docs/man3/xhtml/glVertexAttribPointer.xml
        http://stackoverflow.com/questions/18919927/passing-uint-attribute-to-glsl 

        """
        #
        #if self.gltype in (gl.GL_BYTE, gl.GL_SHORT, gl.GL_INT, gl.GL_UNSIGNED_BYTE, gl.GL_UNSIGNED_SHORT, gl.GL_UNSIGNED_INT):
        #    log.info("enable[I] %s " % repr(self) )
        #    gl.glVertexAttribIPointer( self.index, self.count, self.gltype, self.stride, self.offset )
        #else:
        #    log.info("enable[F] %s " % repr(self) )
        #
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

    def __init__(self, dtype, stride, force_attribute_zero=None, shader=None ):
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

        self.shader = shader
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

    def bind_generic_attrib(self):
        for attribute in self.generic:
            attribute.enable()
            if not self.shader is None:
                self.shader.shader.bind_attribute( attribute.index, attribute.name )  

    def predraw(self, what, att=1):
        """
        #. shader.link is internally skipped when linked already, this
           ensures that the link happens after the attributes are bound on 
           the first draw
        """
        if len(what) > 0:
            gl.glPushClientAttrib( gl.GL_CLIENT_VERTEX_ARRAY_BIT )

        self.bind_generic_attrib()

        if not self.shader is None:
            self.shader.link()  
            self.shader.use()
            self.shader.init_uniforms()   ## check this placement 

        attributes = self.attmap[att]
        for c in attributes.keys():
            if c in what:
                attributes[c].enable()


    def postdraw(self, what, att=1):
        """
        Without the shader unuse, get invalid operation from drawing the geometry 
        """ 
        for attribute in self.generic:
            attribute.disable()
        
        if not self.shader is None:
            self.shader.unuse()

        if len(what) > 0:
            gl.glPopClientAttrib( )


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
    def __init__(self, vertices, indices=None, force_attribute_zero=None, shader=None ):
        """
        :param vertices: numpy ndarray with named constituents
        :param indices: numpy ndarray of element indices
        :param force_attribute_zero: name of 'position' field that will be forced into attribute slot 0 
        :param shader: DAEShader instance
        """
        self.attrib = DAEVertexAttributes(vertices.dtype, vertices.itemsize, force_attribute_zero=force_attribute_zero, shader=shader)
        self.init( vertices, indices )

    def init(self, vertices, indices):
        """
        Upload buffer data to device
        """
        self.init_array_buffer(vertices)
        if indices is None:
            indices = np.arange(vertices.size,dtype=np.uint32)
        self.init_element_array_buffer(indices)

    def init_array_buffer(self, vertices ):
        """
        Upload numpy array into OpenGL array buffer
        """
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


    def draw( self, mode=gl.GL_QUADS, what='pnctesf', offset=0, count=None, att=1 ):
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

        """
        assert count
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.vertices_id )
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id )

        self.attrib.predraw( what=what, att=att)

        gl.glDrawElements( mode, count, self.indices_type, ctypes.c_void_p(self.indices_size*offset) )

        self.attrib.postdraw( what=what, att=att)

        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, 0 ) 
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0 ) 



if __name__ == '__main__':
    pass




