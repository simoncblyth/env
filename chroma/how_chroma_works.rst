How Chroma Works ?
====================

A collection of lightbulb realizations, from perusing the source and
playing around running tests etc..

questions
----------

#. single detector material ? 

   * proper simulation needs to lookup material at generation point of optical photon 
     that gets generated as a result of physics processes happening to other particles 
     being tracked


chroma/generator/photon.py
-----------------------------

ZMQ sockets are used to communicate between the `G4ParallelGenerator`
*controller* and a handful of worker `G4GeneratorProcess` instances.
The `G4ParallelGenerator` provides the *photon_generator* 
used by `chroma.sim.Simulation`.

G4GeneratorProcess
~~~~~~~~~~~~~~~~~~~~

::

   08 class G4GeneratorProcess(multiprocessing.Process)
   09     def __init__(self, idnum, material, vertex_socket_address, photon_socket_address, seed=None):
   10         multiprocessing.Process.__init__(self)


Each instance corresponds to a separate process. 
While run this waits to receive a vertex (via ZMQ socket PULL) 
Vertices are passed to g4gen to generate photons which are 
populated into `ev.photon_beg` and pushed via ZMQ socket.

G4ParallelGenerator
~~~~~~~~~~~~~~~~~~~~~

::

     65 class G4ParallelGenerator(object):
     66     def __init__(self, nprocesses, material, base_seed=None):


Manages a handful of `G4GeneratorProcess` instances, ensuring distinct seeds, 
which it communicates to via zmq sockets. When *generate_events* is called with 
vertices these are pushed to the `G4GeneratorProcess` instances running in 
separate processes and the photon socket is iterated over in order to receive the resulting events 
with *ev.photon_beg* populated once G4 has completed their generation.


chroma/sim.py
---------------

Simulation
~~~~~~~~~~~~

::

    19 class Simulation(object):
    20     def __init__(self, detector, seed=None, cuda_device=None,
    21                  geant4_processes=4, nthreads_per_block=64, max_blocks=1024):
    ..
    36         if geant4_processes > 0:
    37             self.photon_generator = generator.photon.G4ParallelGenerator(geant4_processes, detector.detector_material, base_seed=self.seed)
    38         else:
    39             self.photon_generator = None






 
