Porting Desktop OpenGL (eg g4daeview.py) to iOS
================================================

Motivation
-----------

Easy distribution and installation are huge advantages to 
iOS Apps compared to Linux.  The potential
number of users is amplified by factors of millions, 
mainly because app discovery and install are so easy.

Current iOS Limitations
------------------------

* no CUDA 
* no OpenCL (apparently it is there, but only thru private API, meaning cannot distribute using normal channels  ) 

  * https://github.com/linusyang/opencl-test-ios

OpenGL Compute abuse using TransformFeedback is possible

* http://ciechanowski.me/blog/2014/01/05/exploring_gpgpu_on_ios/  


Metal
-------

Latest devices with A7 chip (with integrated GPU) support Metal 

* http://en.wikipedia.org/wiki/Apple_A7 



Porting Practicalities
-----------------------

* SceneKit (on iOS8), potentially a very high level way to do 3D

  * SceneKit is able to load geometry from COLLADA dae 

* Touch interface, Easy interactivity (cf GLUT)
* iOS OpenGL ES 3.0 (in latest devices) 

  * lacks Geometry shaders
  * OpenGL fixed functions  
  * no glu, matrix support : have to do your own math
  * http://auc.edu.au/2011/09/porting-desktop-opengl-to-ios/ 
  * https://github.com/BradLarson/MoleculesMac 


