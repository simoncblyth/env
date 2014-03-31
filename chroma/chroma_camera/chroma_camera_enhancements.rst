Chroma Camera Enhancements
===========================

Remote Steering/Lookat/Viewpointmarks
----------------------------------------

Analogous to my Navigator and DBUS modifications to MeshLab.

* https://bitbucket.org/scb-/meshlab/commits/b49566e78873bf2e4a5a03578a9dc85b617436b7

Need to send strings (DAENode identifiers and relative positions)
from a remote process to the running chroma-cam pygame application
and get those handled in its event loop, eg changing viewpoint.

An approach using UDP to pick up the message and post to
the pygame event queue.

* http://stackoverflow.com/questions/11361973/post-event-pygame

Maybe can do the IPC at a higher level with zeromq ?



GPU resident pixels
--------------------

No point pulling CUDA ray traced pixels back to host only to send them
back to GPU for display with OpenGL. Instead use shared PBOs.

Calculate ray directions in CUDA code
---------------------------------------

Currently have number of pixel sized arrays: position and direction.
Both unnecessary, can determine px,py from thread index and offset. Those
together with `tan(yfov/2)` can yield ray directions.

* http://www.igorsevo.com/File.ashx?type=res&resType=0&title=source+code%5cCUDAraytracer.cu


  



Quaternion Trackball Control
-----------------------------

* https://www.opengl.org/wiki/GluLookAt_code
* http://www.fastgraph.com/makegames/3drotation/


transformations
~~~~~~~~~~~~~~~

* http://www.lfd.uci.edu/~gohlke/code/transformations.py.html




  









