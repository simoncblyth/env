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






EOU
}
glumpy-dir(){ echo $(local-base)/env/graphics/glumpy/glumpy ; }
glumpy-cd(){  cd $(glumpy-dir); }
glumpy-mate(){ mate $(glumpy-dir) ; }
glumpy-get(){
   local dir=$(dirname $(glumpy-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://code.google.com/p/glumpy/


}
