Forking Multiprocessing pycuda 
====================================

From `benchmark.py` a relevant comment::

     19 # Generator processes need to fork BEFORE the GPU context is setup
     20 g4generator = generator.photon.G4ParallelGenerator(4, water)

This presumably means all processes will use their own GPU context ?


crux of `chroma-sim`
---------------------

It appears that there is no provision for setting up dummy 
materials for model stl files (these are just lists of vertices and faces ?).


::

    detector = chroma.loader.load_geometry_from_string(args[0])

    pos = np.array([float(s) for s in options.pos.split(',')], dtype=float)
    dir = np.array([float(s) for s in options.dir.split(',')], dtype=float)

    ev_gen = constant_particle_gun(particle_name=options.particle, 
                                   pos=pos, dir=dir, ke=options.ke)

    sim = Simulation(detector, seed=options.seed, cuda_device=options.device,
                     geant4_processes=options.ngenerators)
    
    print 'RNG seed: %i' % sim.seed

    writer = root.RootWriter(options.output_filename)

    start = time.time()
    for i, ev in \
            enumerate(sim.simulate(itertools.islice(ev_gen, 
                                                    options.nevents),
                                   keep_photons_beg=options.save_photons_beg,
                                   keep_photons_end=options.save_photons_end)):
        print '\rEvent: %i' % (i+1),
        sys.stdout.flush()

        assert ev.nphotons > 0, 'Geant4 generated event with no photons!'

        writer.write_event(ev)
 



Observations
-------------

#. material definitions are needed
#. possible issue with multiprocessing and pycuda 


::

    (chroma_env)delta:bin blyth$ chroma-sim ../chroma/models/liberty.stl 
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 17060
    INFO:chroma:  vertices:  8573
    INFO:chroma:Loading BVH "default" for geometry from cache.
    Process G4GeneratorProcess-3:
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
    Process G4GeneratorProcess-2:
        self.run()
    Traceback (most recent call last):
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/photon.py", line 20, in run
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
        gen = g4gen.G4Generator(self.material, seed=self.seed)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 42, in __init__
    Process G4GeneratorProcess-1:
        self.world_material = self.create_g4material(material)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 59, in create_g4material
        g4material = G4Material('world_material', material.density * g / cm3,
    Traceback (most recent call last):
    AttributeError: 'NoneType' object has no attribute 'density'
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
        self.run()
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/photon.py", line 20, in run
        self.run()
    Process G4GeneratorProcess-4:
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/photon.py", line 20, in run
        gen = g4gen.G4Generator(self.material, seed=self.seed)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 42, in __init__
    Traceback (most recent call last):
        gen = g4gen.G4Generator(self.material, seed=self.seed)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 42, in __init__
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
        self.world_material = self.create_g4material(material)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 59, in create_g4material
        self.world_material = self.create_g4material(material)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 59, in create_g4material
        g4material = G4Material('world_material', material.density * g / cm3,
        g4material = G4Material('world_material', material.density * g / cm3,
    AttributeError: 'NoneType' object has no attribute 'density'
    AttributeError: 'NoneType' object has no attribute 'density'
        self.run()
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/photon.py", line 20, in run
        gen = g4gen.G4Generator(self.material, seed=self.seed)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 42, in __init__
        self.world_material = self.create_g4material(material)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 59, in create_g4material
        g4material = G4Material('world_material', material.density * g / cm3,
    AttributeError: 'NoneType' object has no attribute 'density'
    Traceback (most recent call last):
      File "/usr/local/env/chroma_env/bin/chroma-sim", line 8, in <module>
        execfile(__file__)
      File "/usr/local/env/chroma_env/src/chroma/bin/chroma-sim", line 115, in <module>
        main()
      File "/usr/local/env/chroma_env/src/chroma/bin/chroma-sim", line 91, in main
        geant4_processes=options.ngenerators)
      File "/usr/local/env/chroma_env/src/chroma/chroma/sim.py", line 41, in __init__
        self.context = gpu.create_cuda_context(cuda_device)
      File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/tools.py", line 129, in create_cuda_context
        cuda.init()
    pycuda._driver.RuntimeError: cuInit failed: no device
    Exception AttributeError: "'Simulation' object has no attribute 'context'" in <bound method Simulation.__del__ of <chroma.sim.Simulation object at 0x1184675d0>> ignored
    (chroma_env)delta:bin blyth$ :



Seen this before, it cuInit fails to find the device on the first try:: 

    (chroma_env)delta:bin blyth$ chroma-sim ../chroma/models/companioncube.stl 
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 17245
    INFO:chroma:  vertices:  8567
    INFO:chroma:Building new BVH using recursive grid algorithm.
    Expanding 170 parent nodes
    Merging 17245 nodes to 5451 parents
    Expanding 46 parent nodes
    Merging 5691 nodes to 1327 parents
    Merging 1373 nodes to 296 parents
    Merging 296 nodes to 56 parents
    Merging 56 nodes to 16 parents
    Merging 16 nodes to 4 parents
    Merging 4 nodes to 1 parent
    INFO:chroma:BVH generated in 0.6 seconds.
    INFO:chroma:Saving BVH (2c45445ab0a39d1ddd3c348400a7fd91:default) to cache.
    Process G4GeneratorProcess-1:
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
        self.run()
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/photon.py", line 20, in run
        gen = g4gen.G4Generator(self.material, seed=self.seed)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 42, in __init__
        self.world_material = self.create_g4material(material)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 59, in create_g4material
        g4material = G4Material('world_material', material.density * g / cm3,
    AttributeError: 'NoneType' object has no attribute 'density'
    Process G4GeneratorProcess-2:
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
        self.run()
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/photon.py", line 20, in run
        gen = g4gen.G4Generator(self.material, seed=self.seed)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 42, in __init__
        self.world_material = self.create_g4material(material)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 59, in create_g4material
    Process G4GeneratorProcess-3:
        g4material = G4Material('world_material', material.density * g / cm3,
    AttributeError: 'NoneType' object has no attribute 'density'
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
        self.run()
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/photon.py", line 20, in run
        gen = g4gen.G4Generator(self.material, seed=self.seed)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 42, in __init__
        self.world_material = self.create_g4material(material)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 59, in create_g4material
        g4material = G4Material('world_material', material.density * g / cm3,
    AttributeError: 'NoneType' object has no attribute 'density'
    Process G4GeneratorProcess-4:
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
        self.run()
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/photon.py", line 20, in run
        gen = g4gen.G4Generator(self.material, seed=self.seed)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 42, in __init__
        self.world_material = self.create_g4material(material)
      File "/usr/local/env/chroma_env/src/chroma/chroma/generator/g4gen.py", line 59, in create_g4material
        g4material = G4Material('world_material', material.density * g / cm3,
    AttributeError: 'NoneType' object has no attribute 'density'
    INFO:chroma:Optimization: Sufficient memory to move triangles onto GPU
    INFO:chroma:Optimization: Sufficient memory to move vertices onto GPU
    INFO:chroma:device usage:
    ----------
    nodes            24.7K 394.9K
    total                  394.9K
    ----------
    device total             2.1G
    device used            238.8M
    device free              1.9G

    RNG seed: 3020167515


