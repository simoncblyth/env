Propagate Timeouts
====================

Some photon files like mock001 succeed to propagate, some 
causing timeouts.

Objectives
----------

* robustify 
* getting sensor ids back into hit collections
* monitoring times


network setup for testing
--------------------------

::

      czmq-
      czmq-broker-local 

      g4daeview.sh --zmqendpoint=tcp://localhost:5002

      OR g4daechroma.sh

      mocknuwa-
      mocknuwa-runenv
      G4DAECHROMA_CLIENT_CONFIG=tcp://localhost:5001 mocknuwa


file based testing
-------------------

debug propagation with::

    daedirectpropagation.sh mock001


holding propagation
---------------------

::


   g4daeview.sh --load mock002 --nopropagate --geometry-regexp PmtHemiCathode
   udp.py --load mock002 
   udp.py --load mock003 
   udp.py --propagate


mock photons
-------------

Using the transform cache, samples of photons were prepared with 
directions oriented with respect to the PMTs. Eg bullseye photons.

To visualize initial photons load with `-P/--nopropagate` 

::

   g4daeview.sh --load mock002 --nopropagate --geometry-regexp PmtHemiCathode


::

   //transport->GetPhotons()->Save("mock002");  // ldir +y
   //transport->GetPhotons()->Save("mock003");  // ldir +x
   //transport->GetPhotons()->Save("mock004");  // ldir +z
   //transport->GetPhotons()->Save("mock005");  // lpos (0,0,100) ldir (0,0,-1)  try to shoot directly at PMT 
   //transport->GetPhotons()->Save("mock006");  // lpos (0,0,500) ldir (0,0,-1)  try to shoot directly at PMT 
   //transport->GetPhotons()->Save("mock007");  // lpos (0,0,1500) ldir (0,0,-1)  try to shoot directly at PMT 



timeouts
---------

pycuda errors that manifest as timeouts can be due to the GPU equivalent 
of a segfault which kills the context, and subsequently causes the 
timeout as the host has no context to talk to on device.

Are certain photon parameters causing "segfaults" on GPU ?


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


