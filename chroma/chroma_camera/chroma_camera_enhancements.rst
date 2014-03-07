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






