Glumpy Usage Of PyOpenGL
=========================


VertexBuffer
-------------

* `/usr/local/env/graphics/glumpy/glumpy/glumpy/graphics/vertex_buffer.py`

Usage
~~~~~~~

`daeviewgl.py`::

    129 def main():
    130     import sys
    131     logging.basicConfig(level=logging.INFO)
    132 
    133     arg = "4998:6000"
    134     vbo = VBO.from_dae(arg, scale=True)
    135 
    136     fig = gp.figure((1024,768))
    137     frame = fig.add_frame(size=(1,1))
    138 
    139     mesh = gp.graphics.VertexBuffer( vbo.data, vbo.faces )
    140     trackball = gp.Trackball( 65, 135, 1.0, 2.5 )
    141 
    142     FigHandler(fig, trackball)
    143     FrameHandler(frame, mesh, trackball)
    144 
    145     gp.show()



Adapted from `demos/obj-viewer.py` in `daeviewgl.py`::

     38     def __init__(self, vertices, normals, faces):
     39 
     40         data = np.zeros(len(vertices), [('position', np.float32, 3),
     41                                      ('color', np.float32, 3),
     42                                      ('normal',   np.float32, 3)])
     43         data['position'] = vertices
     44         data['color'] = np.random.random(vertices.shape)   # (vertices+1)/2.
     45         data['normal'] = normals
     46 
     47         self.data = data
     48         self.faces = faces


__init__
~~~~~~~~~~~

Dicts of attributes keyed on first char of subdtype name created, buffers created and bound::

    128 class VertexBuffer(object):
    130     def __init__(self, vertices, indices=None):
    ...
    155             if name in['position', 'color', 'normal', 'tex_coord',
    156                        'fog_coord', 'secondary_color', 'edge_flag']:
    157                 vclass = 'VertexAttribute_%s' % name
    158                 attribute = eval(vclass)(count,gltype,stride,offset)
    159                 self.attributes[name[0]] = attribute
    160             else:
    161                 attribute = VertexAttribute_generic(count,gltype,stride,offset,index)
    162                 self.generic_attributes.append(attribute)
    163                 index += 1
    ...
    167         self.vertices_id = gl.glGenBuffers(1)
    168 
    169         gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.vertices_id )
    170         gl.glBufferData( gl.GL_ARRAY_BUFFER, self.vertices, gl.GL_STATIC_DRAW )
    171         gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0 )
    172 
    173         self.indices = indices
    174         self.indices_id = gl.glGenBuffers(1)
    175         gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id )
    176         gl.glBufferData( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices, gl.GL_STATIC_DRAW )
    177         gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, 0 )


draw
~~~~~~

::

    186     def draw( self, mode=gl.GL_QUADS, what='pnctesf' ):
    187         gl.glPushClientAttrib( gl.GL_CLIENT_VERTEX_ARRAY_BIT )          # mask specifies which attributes to save
    188         gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.vertices_id )         # select this objects ARRAY and ELEMENT_ARRAY buffers via id
    189         gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id )
    190         for attribute in self.generic_attributes:
    191             attribute.enable()
    192         for c in self.attributes.keys():
    193             if c in what:
    194                 self.attributes[c].enable()
    195         gl.glDrawElements( mode, self.indices.size, gl.GL_UNSIGNED_INT, None)
    196         gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, 0 )
    197         gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0 )
    198         gl.glPopClientAttrib( )
    199 


VertexAttribute_position/color/normal/...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* http://pyopengl.sourceforge.net/context/tutorials/shader_4.xhtml

::

    ... the use of glVertexPointer and glColorPointer is part of the "legacy" OpenGL
    API. When using shader-based geometry there is no need to restrict oneself to a
    single position, colour or texture for a given vertex.



glVertexPointer
^^^^^^^^^^^^^^^^^

* http://www.opengl.org/sdk/docs/man2/xhtml/glVertexPointer.xml



::

    096 class VertexAttribute_position(VertexAttribute):
    097     def __init__(self, count, gltype, stride, offset):
    098         assert count > 1, \
    099             'Vertex attribute must have count of 2, 3 or 4'
    100         assert gltype in (gl.GL_SHORT, gl.GL_INT, gl.GL_FLOAT, gl.GL_DOUBLE), \
    101             'Vertex attribute must have signed type larger than byte'
    102         VertexAttribute.__init__(self, count, gltype, stride, offset)
    103     def enable(self):
    104         gl.glVertexPointer(self.count, self.gltype, self.stride, self.offset)
    105         gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    106 


glColorPointer
^^^^^^^^^^^^^^^^^

* https://www.opengl.org/sdk/docs/man2/xhtml/glColorPointer.xml


::

     27 class VertexAttribute_color(VertexAttribute):
     28     def __init__(self, count, gltype, stride, offset):
     29         assert count in (3, 4), \
     30             'Color attributes must have count of 3 or 4'
     31         VertexAttribute.__init__(self, count, gltype, stride, offset)
     32     def enable(self):
     33         gl.glColorPointer(self.count, self.gltype, self.stride, self.offset)
     34         gl.glEnableClientState(gl.GL_COLOR_ARRAY)
     35 
     36 


glNormalPointer
^^^^^^^^^^^^^^^^^

::

     60 class VertexAttribute_normal(VertexAttribute):
     61     def __init__(self, count, gltype, stride, offset):
     62         assert count == 3, \
     63             'Normal attribute must have a size of 3'
     64         assert gltype in (gl.GL_BYTE, gl.GL_SHORT,
     65                           gl.GL_INT, gl.GL_FLOAT, gl.GL_DOUBLE), \
     66                                 'Normal attribute must have signed type'
     67         VertexAttribute.__init__(self, 3, gltype, stride, offset)
     68     def enable(self):
     69         gl.glNormalPointer(self.gltype, self.stride, self.offset)
     70         gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
     71 








