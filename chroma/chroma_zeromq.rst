Chroma ZeroMQ
=================


Where does ZeroMQ server fit in ?
------------------------------------

Its communication glue between processes.::

    simon:chroma blyth$ find . -name '*.py' -exec grep -H zmq {} \;
    ./chroma/generator/photon.py:import zmq
    ./chroma/generator/photon.py:        context = zmq.Context()
    ./chroma/generator/photon.py:        vertex_socket = context.socket(zmq.PULL)
    ./chroma/generator/photon.py:        photon_socket = context.socket(zmq.PUSH)
    ./chroma/generator/photon.py:        self.zmq_context = zmq.Context()
    ./chroma/generator/photon.py:        self.vertex_socket = self.zmq_context.socket(zmq.PUSH)
    ./chroma/generator/photon.py:        self.photon_socket = self.zmq_context.socket(zmq.PULL)
    ./setup.py:    install_requires = ['uncertainties','pyzmq-static','spnav', 'pycuda', 
    simon:chroma blyth$ 

That misses **chroma_server**, which provides propagation on demand to remote clients::

    (chroma_env)delta:chroma blyth$ grep zmq ../bin/* 
    ../bin/chroma-server:import zmq
    ../bin/chroma-server:        self.context = zmq.Context()
    ../bin/chroma-server:        self.socket = self.context.socket(zmq.REP)


chroma/generator/photon.py
---------------------------


::

     65 class G4ParallelGenerator(object):
     66     def __init__(self, nprocesses, material, base_seed=None):
     67         self.material = material
     68         if base_seed is None:
     69             base_seed = np.random.randint(100000000)
     70         base_address = 'ipc:///tmp/chroma_'+str(uuid.uuid4())
     71         self.vertex_address = base_address + '.vertex'
     72         self.photon_address = base_address + '.photon'
     73         self.processes = [ G4GeneratorProcess(i, material, self.vertex_address, self.photon_address, seed=base_seed + i) for i in xrange(nprocesses) ]
     74 
     75         for p in self.processes:
     76             p.start()


::

    simon:chroma blyth$ find . -name '*.py' -exec grep -H G4Parallel  {} \;
    ./chroma/benchmark.py:g4generator = generator.photon.G4ParallelGenerator(4, water)
    ./chroma/generator/photon.py:class G4ParallelGenerator(object):
    ./chroma/sim.py:            self.photon_generator = generator.photon.G4ParallelGenerator(geant4_processes, detector.detector_material, base_seed=self.seed)
    ./test/test_generator_photon.py:class TestG4ParallelGenerator(unittest.TestCase):
    ./test/test_generator_photon.py:        gen = generator.photon.G4ParallelGenerator(1, water)
    ./test/test_generator_photon.py:        gen = generator.photon.G4ParallelGenerator(1, water)
    ./test/test_pdf.py:from chroma.generator.photon import G4ParallelGenerator
    ./test/test_pdf.py:        g4generator = G4ParallelGenerator(1, water)


chroma/sim.py
--------------


::

     19 class Simulation(object):
     20     def __init__(self, detector, seed=None, cuda_device=None,
     21                  geant4_processes=4, nthreads_per_block=64, max_blocks=1024):
     22         self.detector = detector
     23 
     24         self.nthreads_per_block = nthreads_per_block
     25         self.max_blocks = max_blocks
     26 
     27         if seed is None:
     28             self.seed = pick_seed()
     29         else:
     30             self.seed = seed
     31 
     32         # We have three generators to seed: numpy.random, GEANT4, and CURAND.
     33         # The latter two are done below.
     34         np.random.seed(self.seed)
     35 
     36         if geant4_processes > 0:
     37             self.photon_generator = generator.photon.G4ParallelGenerator(geant4_processes, detector.detector_material, base_seed=self.seed)
     38         else:
     39             self.photon_generator = None
     40 


chroma/io/root.py
-------------------


bin/chroma-server 
-----------------

* elegant simplicity, should be easy to integrate against 


#. loads geometry from STL file specified by cmdline arg
#. bind zmq server to address eg `tcp://*:5024` that:

   * listens for photon objs 
   * propagates them  and returns end photons to client

::

     11 class ChromaServer(object):
     12     '''A ZeroMQ socket server which listens for incoming Photons objects
     13     and replies with propagated Photons.
     14 
     15     :param address: Socket address on which to listen
     16     :param detector: Detector to progagate photons in
     17     '''
     18     def __init__(self, address, detector):
     19         # set up zeromq socket
     20         self.address = address
     21         self.context = zmq.Context()
     22         self.socket = self.context.socket(zmq.REP)
     23         self.socket.bind(address)
     24 
     25         # set up simulation
     26         self.detector = detector
     27         self.sim = Simulation(self.detector)
     28 
     29     def serve_forever(self):
     30         '''Listen for photons, propagate them, and return the final states.'''
     31         while True:
     32             photons_in = self.socket.recv_pyobj()
     33             print 'Processing', len(photons_in), 'photons'
     34 
     35             # propagate in chroma simulation
     36             photons_end = self.sim.simulate(photons_in, keep_photons_end=True).next()
     37 
     38             # return final photon vertices to client
     39             self.socket.send_pyobj(photons_end)
     40 


BUT: its expecting python objects




client usage of this server 
---------------------------

* https://learning-0mq-with-pyzmq.readthedocs.org/en/latest/pyzmq/patterns/client_server.html

Would expect some connection request code like::

    context = zmq.Context()
    print "Connecting to server..."
    socket = context.socket(zmq.REQ)
    socket.connect ("tcp://localhost:%s" % port)
    if len(sys.argv) > 2:
        socket.connect ("tcp://localhost:%s" % port1)


Used to communicate between chroma and g4py subprocesses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `G4ParallelGenerator` uses ZMQ to send vertices and get photons 
back from its `g4py` sub-processes


::

    simon:chroma blyth$ find . -name '*'  -exec grep -H zmq {} \;
    Binary file ./.hg/store/data/chroma/models/_colbert___high_res___brow.stl.bz2.d matches
    Binary file ./bin/.chroma-server.swp matches
    ./bin/chroma-server:import zmq
    ./bin/chroma-server:        self.context = zmq.Context()
    ./bin/chroma-server:        self.socket = self.context.socket(zmq.REP)
    ./chroma/generator/photon.py:import zmq
    ./chroma/generator/photon.py:        context = zmq.Context()
    ./chroma/generator/photon.py:        vertex_socket = context.socket(zmq.PULL)
    ./chroma/generator/photon.py:        photon_socket = context.socket(zmq.PUSH)
    ./chroma/generator/photon.py:        self.zmq_context = zmq.Context()
    ./chroma/generator/photon.py:        self.vertex_socket = self.zmq_context.socket(zmq.PUSH)
    ./chroma/generator/photon.py:        self.photon_socket = self.zmq_context.socket(zmq.PULL)
    Binary file ./chroma/models/Colbert_HighRes_Brow.stl.bz2 matches
    ./setup.py:    install_requires = ['uncertainties','pyzmq-static','spnav', 'pycuda', 
    simon:chroma blyth$ 




bin/chroma-sim
----------------

#. geometry loading from STL file specified on cmdline
#. particle gun specification via cmdline options
#. simulation

    #. generation using G4ParallelGenerator
    #. propagation to detectors

#. ROOT output of events 


