OpenGL Versions
==================

pyopengl 
---------

::

    2014-06-11 14:24:14,625 OpenGL.extensions   :92  OpenGL Version: 2.1 NVIDIA-8.24.9 310.40.25f01



Mavericks Supports OpenGL 4.1 by glut/glumpy/pyopengl sees 2.1
----------------------------------------------------------------

Should
~~~~~~~

* https://developer.apple.com/graphicsimaging/opengl/capabilities/

GeForce 750M on Mavericks shoud support 

OpenGL 4.1
GLSL   4.10

* http://support.apple.com/kb/HT5942


Superbible
~~~~~~~~~~~~

* https://github.com/openglsuperbible/sb6code
* http://www.openglsuperbible.com/2013/11/11/porting-samples-to-mac/

On October 22, 2013, Apple released OS X Mavericks, also known as OS X version 10.9. 
This version of the operating system included a long awaited update to the 
supported version of OpenGL. The 6th edition of the OpenGL SuperBible is
about OpenGL version 4.3, and unfortunately, Apple’s latest and greatest only
supports version 4.1 of the API. As OpenGL 4.1 was released on July 26, 2010,
this puts OS X more than three years behind. 


Andon M. Coleman
~~~~~~~~~~~~~~~~~~~

* http://stackoverflow.com/questions/20264814/glsl-version-130-on-mac-os-x-causes-error

Unless you select a pixel format that explicitly asks for a core profile, then
you are going to get an OpenGL 2.1 implementation. See my answer to another
question for more details on how to do this; 
this is a new change to the CGL /
NSOpenGL APIs that was introduced with OS X 10.7, so some older books may not
document it.

There is a big difference between what your GPU supports and what you actually
get. On a lot of other platforms, without using changes to the window system
APIs that were introduced alongside OpenGL 3.2, you can get all the features of
an OpenGL 3.2+ implementation and the legacy things from OpenGL 2.1 and earlier
by default (this is known as a compatibility profile).

OS X is different, it does not support compatibility profiles. You either get
the legacy OpenGL 2.1 implementation or the core 3.2 (3.3/4.1 in OS X 10.9)
implementation but you can never mix-and-match features from both. Furthermore,
unless you modify your code to ask for a core profile, you will be limited to
OpenGL 2.1 by default.



* http://stackoverflow.com/questions/19865463/opengl-4-1-under-mavericks/19868861#19868861



When you request your pixel format using one of the lower-level APIs on OS X,
you need to add the following to your attribute list in order to use a core
profile:

CGL:

  kCGLPFAOpenGLProfile,     kCGLOGLPVersion_3_2_Core NSOpenGL:

  NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core 

Now, while the
particular constant is named ...3_2Core, what it actually means is request a
context that removes all deprecated features and supports at least OpenGL 3.2
(a core profile, in other words). You can get a 4.1 or 3.3 context using this
same constant; in all honesty, including an actual version number in the
constant was probably a poor choice.

If you do not specify this when you request your pixel format, OS X will give
you kCGLOGLPVersion_Legacy or NSOpenGLProfileVersionLegacy respectively. And
this will limit you to OpenGL 2.1 functionality.

If you are using a higher-level framework, then you will need to consult your
API reference. However, be aware that on OS X you must have a core profile
context to access anything newer than OpenGL 2.1.


There is FreeGLUT, which is maintained... the problem with using it on OS X is
that it goes through X11. X11 is provided on OS X through a compatibility X
server called XQuartz, you cannot get a 3.2+ OpenGL context using the XQuartz
server because it does not implement the necessary GLX extension to request a
core profile. So that means any framework that uses 3.2 on OS X has to go
through native APIs like CGL (Fullscreen and C) or NSOpenGL
(Fullscreen/Windowed and Objective C). GLFW3 and many other frameworks do this,
FreeGLUT still does not. –  Andon M. Coleman Jun 2 at 18:09


Looks like stuck with OpenGL 2.1 / GLSL 120 until go native
-------------------------------------------------------------

g4daeview.py is underpinned by glumpy/pyopengl/glut
making the effort to move g4daeview.py 
beyond OpenGL 2.1 and GLSL 120 probably not worthwhile,
as will probably go native in a few months anyhow

* https://github.com/adamlwgriffiths/PyGLy
* https://github.com/nightcracker/pyglfw


GLSL Version 120
-------------------

* https://github.com/mattdesl/lwjgl-basics/wiki/GLSL-Versions

Are stuck with GLSL 120 without extreme contortions::

    Exception: Shader compilation error GL_VERTEX_SHADER (0x8B31) : ERROR: 0:1: '' :  version '130' is not supported





