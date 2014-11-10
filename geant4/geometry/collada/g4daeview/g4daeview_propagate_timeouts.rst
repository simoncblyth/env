Propagate Timeouts
====================



Some photon files like mock001 succeed to propagate.



::

     File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daephotons.py", line 222, in propagate
        self.propagator.interop_propagate( vbo, max_steps=max_steps, max_slots=max_slots )
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daephotonspropagator.py", line 192, in interop_propagate
        self.propagate( vbo_dev_ptr, max_steps=max_steps, max_slots=max_slots )   
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daephotonspropagator.py", line 160, in propagate
        t = get_time()
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 453, in get_call_time
        end.synchronize()
    pycuda._driver.LaunchError: cuEventSynchronize failed: launch timeout
    PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    cuEventDestroy failed: launch timeout
    PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    cuEventDestroy failed: launch timeout
    PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    cuGLUnmapBufferObject failed: launch timeout
    (chroma_env)delta:g4daeview blyth$ 
    (chroma_env)delta:g4daeview blyth$ 
    (chroma_env)delta:g4daeview blyth$ g4daeview.sh --load mock007


