G4DAEView Performance
=====================

Pushing to max-slots 100 bogs down performance
------------------------------------------------

Switching to glDrawElements rather than glMultiDrawArrays avoids
crashing and succeeds to animate slowy. Switching off geometry drawing improves 
performance substantially.  

Suspect bottleneck is GPU memory, the lack of it forcing 
to swap things in/out and killing performance.

Probably insufficient space to hold all in GPU memory at once:

#. geometry VBO
#. Chroma geometry 
#. photon VBO


Apple OpenGL Profiler
----------------------

* :google:`Apple OpenGL Profiler`

* see :doc:`/osx/graphicstools/graphicstools`




LINE_STRIP slow performance report
-----------------------------------

* http://lists.apple.com/archives/mac-opengl/2011/Sep/msg00007.html



