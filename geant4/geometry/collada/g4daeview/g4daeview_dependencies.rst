G4DAEView Dependencies
========================

Dependencies Tree
-------------------

* PyOpenGL
* glumpy 

  * OpenGL.GLUT (PyOpenGL wrapped)

    * GLUT.framework

* ROOT, used for ChromaPhotonList deserialization


Alternatives to GLUT ?
-------------------------

The glumpy dependency `GLUT.framework` prevents going beyond OpenGL 2.1, GLSL 120.
This is despite OSX 10.9 system and MBP hardware supporting OpenGL 4.1.

Implications of ancient OpenGL 2.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. developing against obsolete API, eg the geometry shader extension API
#. GLSL 120 (even with extensions) have very poor int/uint support
   so forced to do even simple things like photon selection based on flags in CUDA 
   rather than the shader

GLFW
~~~~~

* Modern OpenGL support on OSX, Linux, Windows
* Not a GLUT drop in (no menu-ing)

* http://www.geeks3d.com/20121109/overview-of-opengl-support-on-os-x/
* http://stackoverflow.com/questions/19993078/looking-for-a-simple-opengl-3-2-python-example-that-uses-glfw

glfwcy with PyOpenGL and OpenGL 3.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/electronut/pp/blob/master/simplegl/simpleglfw.py

glfw python wrappers
~~~~~~~~~~~~~~~~~~~~~

* https://pypi.python.org/pypi/glfw/1.0.1
* https://github.com/enthought/glfwpy
* https://github.com/glfw/glfw
    
migration from glumpy to glfw
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://groups.google.com/forum/m/#!msg/pupil-discuss/qrxH8KGh7vU/ZvRpbg7Mnu0J

  * example of a python project making the move to GLFW



