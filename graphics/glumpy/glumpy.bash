# === func-gen- : graphics/glumpy/glumpy fgp graphics/glumpy/glumpy.bash fgn glumpy fgh graphics/glumpy
glumpy-src(){      echo graphics/glumpy/glumpy.bash ; }
glumpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(glumpy-src)} ; }
glumpy-vi(){       vi $(glumpy-source) ; }
glumpy-env(){      elocal- ; }
glumpy-usage(){ cat << EOU

GLUMPY
=======

* https://code.google.com/p/glumpy/
* http://groups.google.com/group/glumpy-users

glumpy is a small python library for the rapid vizualization of numpy arrays,
(mainly two dimensional) that has been designed with efficiency in mind. If you
want to draw nice figures for inclusion in a scientific article, better
use matplotlib. If you want to have a sense of whats going on in your
simulation while it is running, then maybe glumpy can help you.

glumpy is made on top of PyOpenGL (http://pyopengl.sourceforge.net/) 


forks
-----

#. https://github.com/davidcox/glumpy  key binding extension


alternatives
--------------

#. https://github.com/rossant/galry
#. https://github.com/vispy/vispy
#. http://vispy.org/
#. https://code.google.com/p/visvis/
#. http://www.pyqtgraph.org/


installations
--------------

Delta
~~~~~

Into chroma virtualenv python (based on macports python 2.7.6)::

    glumpy-get
    cd glumpy
    python setup.py install
    ...
    Writing /usr/local/env/chroma_env/lib/python2.7/site-packages/glumpy-0.2.1-py2.7.egg-info


G4PB
~~~~~

Into macports py26::

    g4pb:~ blyth$ glumpy-
    g4pb:~ blyth$ glumpy-get
    Cloning into 'glumpy'...
    remote: Counting objects: 1026, done.
    Receiving objects: 100% (1026/1026), 4.26 MiB | 899 KiB/s, done.
    Resolving deltas: 100% (699/699), done.
    g4pb:glumpy blyth$ pwd
    /usr/local/env/graphics/glumpy
    g4pb:glumpy blyth$ cd glumpy/
    g4pb:glumpy blyth$ which python
    /opt/local/bin/python
    g4pb:glumpy blyth$ sudo python setup.py install
    ...
    Writing /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/glumpy-0.2.1-py2.6.egg-info
    g4pb:glumpy blyth$ 


issue1 : undefined function glutMainLoopEvent (OSX 10.5.8 macports py26)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Kludge fix via edit to backend_glut.py

::

    Traceback (most recent call last):
      File "/Users/blyth/env/bin/daeviewgl.py", line 4, in <module>
        main()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/env/geant4/geometry/collada/daeview/daeviewgl.py", line 223, in main
        gp.show()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/glumpy/window/backend_glut.py", line 58, in show
        _window.start()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/glumpy/window/backend_glut.py", line 591, in start
        glut.glutMainLoopEvent()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/OpenGL/platform/baseplatform.py", line 340, in __call__
        self.__name__, self.__name__,
    OpenGL.error.NullFunctionError: Attempt to call an undefined function glutMainLoopEvent, check for bool(glutMainLoopEvent) before calling
    g4pb:~ blyth$ 

glumpy/window/backend_glut.py failing in start::

    585         if not self._interactive:
    586             #glut.glutMainLoop()
    587             while not self._stop_mainloop:
    588                 try:
    589                     glut.glutCheckLoop()
    590                 except:
    591                     glut.glutMainLoopEvent()
    592 

Uncommenting the *glut.glutMainLook()* on line 586 succeeds to get graphical view to appear::

    g4pb:~ blyth$ sudo vi /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/glumpy/window/backend_glut.py +586

Similar to http://code.google.com/p/glumpy/issues/detail?id=22


issue2 : key/trackpad binding (OSX 10.5.8 macports py26)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unable to Z/XY pan (key binding/scroll wheel?)


issue3 : fig.save via FBO fails
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

My workaround is to avoid FBO and simply glReadPixels from the framebuffer::

     12   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/daeview/daeinteractivityhandler.py", line 171, in save_to_file
     13     self.fig.save("%s.png" % name )
     14   File "/usr/local/env/chroma_env/lib/python2.7/site-packages/glumpy/figure.py", line 293, in save
     15     fbo.glRenderbufferStorageEXT( fbo.GL_RENDERBUFFER_EXT, fbo.GL_DEPTH_COMPONENT, w, h)
     16 AttributeError: 'module' object has no attribute 'GL_DEPTH_COMPONENT'

Attempted to fix on Delta by commenting depthbuffer, this succeeds to write a file, 
but checking in Preview its distinctly odd looking. Code changes remain::

    (chroma_env)delta:glumpy blyth$ pwd 
    /usr/local/env/chroma_env/lib/python2.7/site-packages/glumpy
    (chroma_env)delta:glumpy blyth$ vi figure.py






demos
------

obj-viewer
~~~~~~~~~~~

Nice trackball operation 

/usr/local/env/graphics/glumpy/glumpy/demos/obj-viewer.py


interesting code
-------------------

graphics/vertex_buffer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attributes describe the meaning of VBO arrays.

Base class: `VertexAttribute(count,gltype,stride,offset)`
with sub classes that check ctor arguments and provide enable methods eg:

VertexAttribute_color
    gl.glColorPointer(self.count, self.gltype, self.stride, self.offset)
    gl.glEnableClientState(gl.GL_COLOR_ARRAY)

VertexAttribute_edge_flag
    gl.glEdgeFlagPointer(self.stride, self.offset)
    gl.glEnableClientState(gl.GL_EDGE_FLAG_ARRAY)

VertexAttribute_fog_coord
    gl.glFogCoordPointer(self.count, self.gltype, self.stride, self.offset)
    gl.glEnableClientState(gl.GL_FOG_COORD_ARRAY)

VertexAttribute_normal
    gl.glNormalPointer(self.gltype, self.stride, self.offset)
    gl.glEnableClientState(gl.GL_NORMAL_ARRAY)

VertexAttribute_secondary_color
    gl.glSecondaryColorPointer(3, self.gltype, self.stride, self.offset)
    gl.glEnableClientState(gl.GL_SECONDARY_COLOR_ARRAY)

VertexAttribute_tex_coord
    gl.glTexCoordPointer(self.count, self.gltype, self.stride, self.offset)
    gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)

VertexAttribute_position
    gl.glVertexPointer(self.count, self.gltype, self.stride, self.offset)
    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)

VertexAttribute_generic
    gl.glVertexAttribPointer( self.index, self.count, self.gltype, self.normalized, self.stride, self.offset );
    gl.glEnableVertexAttribArray( self.index )


==================  ==================   ===================   =====
gl***Pointer          GL_***_ARRAY          Att names            *
==================  ==================   ===================   =====
 Color                COLOR                color                 c
 EdgeFlag             EDGE_FLAG            edge_flag             e
 FogCoord             FOG_COORD            fog_coord             f
 Normal               NORMAL               normal                n
 SecondaryColor       SECONDARY_COLOR      secondary_color       s
 TexCoord             TEXTURE_COORD        tex_coord             t 
 Vertex               VERTEX               position              p
 VertexAttrib         N/A             
==================  ==================   ===================   =====


#. clear pattern followed by all apart from `_generic`.
#. some pointer calls take not all of count/gltype/stride/offset


Buffer identifiers vertices_id and indices_id are generated
and bound to GL_ARRAY_BUFFER and GL_ELEMENT_ARRAY_BUFFER
and the buffer data is passed, causing once only upload to the GPU 
in the ctor.

::

        self.vertices_id = gl.glGenBuffers(1)
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.vertices_id )
        gl.glBufferData( gl.GL_ARRAY_BUFFER, self.vertices, gl.GL_STATIC_DRAW )
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0 )

        self.indices_id = gl.glGenBuffers(1)
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id )
        gl.glBufferData( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices, gl.GL_STATIC_DRAW )
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, 0 )


Thence at VBO draw:

#. GL_ARRAY_BUFFER bound using vertices_id 
#. GL_ELEMENT_ARRAY_BUFFER bound using indices_id
#. any generic attributes are enabled
#. attributes chosen by `what` argument first letters are enabled
   which switch the meanings of the array data

#. gl.glDrawElements( mode, self.indices.size, gl.gl_UNSIGNED_INT, None )
#. GL_ARRAY_BUFFER and GL_ELEMENT_ARRAY_BUFFER are unbound


::

    def draw( self, mode=gl.GL_QUADS, what='pnctesf' ):
        gl.glPushClientAttrib( gl.GL_CLIENT_VERTEX_ARRAY_BIT )
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.vertices_id )
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, self.indices_id )

        for attribute in self.generic_attributes:
            attribute.enable()

        for c in self.attributes.keys():
            if c in what:
                self.attributes[c].enable()
        gl.glDrawElements( mode, self.indices.size, gl.GL_UNSIGNED_INT, None)
        gl.glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, 0 )
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, 0 )
        gl.glPopClientAttrib( )



* http://www.opengl.org/sdk/docs/man2/xhtml/glDrawElements.xml

::

  glDrawElements( GLenum      mode,
                  GLsizei     count,      # Specifies the number of elements to be rendered
                  GLenum      type,       # Specifies the type of the values in indices. 
                                            Must be one of GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, or GL_UNSIGNED_INT.
                  const GLvoid *indices); # Specifies a pointer to the location where the indices are stored 


When glDrawElements is called, it uses *count* sequential elements from an
enabled array, starting at *indices* to construct a sequence of geometric
primitives. *mode* specifies what kind of primitives are constructed and how the
array elements construct these primitives. If more than one array is enabled,
each is used. If GL_VERTEX_ARRAY is not enabled, no geometric primitives are
constructed.

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



* http://www.opengl.org/sdk/docs/man2/xhtml/glDrawRangeElements.xml

::

    void glDrawRangeElements(   GLenum      mode,
                                GLuint      start,
                                GLuint      end,
                                GLsizei     count,
                                GLenum      type,
                                const GLvoid *      indices);


glDrawRangeElements is available only if the GL version is 1.2 or greater.



glumpy timers
---------------

TODO: work out way to switch off timers

/usr/local/env/graphics/glumpy/glumpy/glumpy/window/backend_glut.py::

    560     def start(self):
    561         ''' Starts main loop. '''
    562 
    563         # Start timers
    564         for i in range(len(self._timer_stack)):
    565             def func(index):
    566                 handler, fps = self._timer_stack[index]
    567                 t = glut.glutGet(glut.GLUT_ELAPSED_TIME)
    568                 dt = (t - self._timer_date[index])/1000.0
    569                 self._timer_date[index] = t
    570                 handler(dt)
    571                 glut.glutTimerFunc(int(1000./fps), func, index)
    572                 self._timer_date[index] = glut.glutGet(glut.GLUT_ELAPSED_TIME)
    573             fps = self._timer_stack[i][1]
    574             glut.glutTimerFunc(int(1000./fps), func, i)
     
::

    delta:glumpy blyth$ find . -name '*.py' -exec grep -H _timer {} \;
    ./graphics/vertex_buffer.py:def on_timer(value):
    ./graphics/vertex_buffer.py:    glut.glutTimerFunc(10, on_timer, 0)
    ./graphics/vertex_buffer.py:    glut.glutTimerFunc(10, on_timer, 0)
    ./window/backend_glut.py:        self._timer_stack = []
    ./window/backend_glut.py:        self._timer_date = []
    ./window/backend_glut.py:        for i in range(len(self._timer_stack)):
    ./window/backend_glut.py:                handler, fps = self._timer_stack[index]
    ./window/backend_glut.py:                dt = (t - self._timer_date[index])/1000.0
    ./window/backend_glut.py:                self._timer_date[index] = t
    ./window/backend_glut.py:                self._timer_date[index] = glut.glutGet(glut.GLUT_ELAPSED_TIME)
    ./window/backend_glut.py:            fps = self._timer_stack[i][1]
    ./window/window.py:            self._timer_stack.append((func, fps))
    ./window/window.py:            self._timer_date.append(0)
    delta:glumpy blyth$ 





EOU
}
glumpy-dir(){ echo $(local-base)/env/graphics/glumpy/glumpy ; }
glumpy-cd(){  cd $(glumpy-dir); }
glumpy-mate(){ mate $(glumpy-dir) ; }
glumpy-get(){
   local dir=$(dirname $(glumpy-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://code.google.com/p/glumpy/


}
