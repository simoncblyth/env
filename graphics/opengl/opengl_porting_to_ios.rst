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


Transform Feedback
~~~~~~~~~~~~~~~~~~~~

* http://prideout.net/blog/?tag=opengl-transform-feedback



Metal
-------

Latest devices with A7 chip (with integrated GPU) support Metal 
which provides CUDA like compute capabilities.

* http://en.wikipedia.org/wiki/Apple_A7 

* http://www.anandtech.com/show/8116/some-thoughts-on-apples-metal-api

    On a final note, while we’ve discussed graphics almost exclusively thus far,
    it’s interesting to note that Apple is pitching Metal as an API for GPU compute
    as well as graphics. Despite being one of the initial promoters of the OpenCL
    API, Apple has never implemented OpenCL or any other GPU compute API on iOS
    thus far, even after they adopted the compute-friendly PowerVR Rogue GPU for
    their A7 SoC. As a result GPU compute on iOS has been limited to what OpenGL ES
    can be coaxed into, which although not wholly incapable, it is an API designed
    for dealing with images as opposed to free form data.



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



OpenGL ES Capabilities of iOS Devices
--------------------------------------

* https://developer.apple.com/library/ios/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/OpenGLESPlatforms/OpenGLESPlatforms.html



