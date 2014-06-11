OpenGL Releases
================

* http://www.opengl.org/wiki/History_of_OpenGL


* OpenGL 4.4 (2013)
* OpenGL 4.3 (2012) : Added Compute Shaders. http://www.opengl.org/wiki/Compute_Shader
* OpenGL 4.2 (2011)
* OpenGL 4.1 (2010)
  
  * OSX 10.9 (Mavericks, Oct 2013)

* OpenGL 4.0 (2010)
* OpenGL 3.3 (2010)
* OpenGL 3.2 (2009)

  * OSX 10.8.5
  * OSX 10.7.5

* OpenGL 3.1 (2009)
* OpenGL 3.0 (2008)
* OpenGL 2.1 (2006)
* OpenGL 2.0 (2004)


OpenGL ES
-----------

* http://stackoverflow.com/questions/4784137/choose-opengl-es-1-1-or-opengl-es-2-0

* OpenGL ES 1.1, fixed pipeline
* OpenGL ES 2.0, transition to shaders : removed the cruft

  * http://glslstudio.com/primer/  (compares ES 1.1 and ES 2.0)

* OpenGL ES 3.0 



PyOpenGL Deprecations
-----------------------

* http://pyopengl.sourceforge.net/documentation/deprecations.html


::

    import OpenGL
    OpenGL.FORWARD_COMPATIBLE_ONLY = True


::

    (chroma_env)delta:g4daeview blyth$ grep -l import\ OpenGL *.py
    daeclipper.py
    daeframehandler.py
    daeillustrate.py
    daelights.py
    daemenu.py
    daephotonsrenderer.py
    daetext.py
    daevertexbuffer.py
    daeviewport.py




OSX OpenGL capabilites matrix
-------------------------------

The ``GL_ARB`` extension system means that correspondence between an  OpenGL feature and the OS  implmentation
is not cut and dryed.  Hence the matrix: 

* https://developer.apple.com/graphicsimaging/opengl/capabilities/index.html

However:

* For OSX 10.9 (Mavericks, Oct 2013) compatibility only extends up to features from OpenGL 4.1 
* For OSX 10.8.5 () compatibility only extends up to features from OpenGL 3.2


OpenGL Superbible OSX examples on Mavericks
---------------------------------------------

* http://www.openglsuperbible.com/2013/11/11/porting-samples-to-mac/








