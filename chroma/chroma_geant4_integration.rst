Chroma Geant4 Integration
==========================

See also :doc:`geant4_background`

Chroma Use of G4
-----------------

Chroma uses G4 for photon generation.::

    simon:chroma blyth$ find ~/chroma/chroma  -name '*.py' -exec grep -H G4 {} \;
    /Users/blyth/chroma/chroma/benchmark.py:g4generator = generator.photon.G4ParallelGenerator(4, water)
    /Users/blyth/chroma/chroma/generator/g4gen.py:class G4Generator(object):
    /Users/blyth/chroma/chroma/generator/g4gen.py:        self.world = g4py.ezgeom.G4EzVolume('world')
    /Users/blyth/chroma/chroma/generator/g4gen.py:        self.world.PlaceIt(G4ThreeVector(0,0,0))
    /Users/blyth/chroma/chroma/generator/g4gen.py:        g4material = G4Material('world_material', material.density * g / cm3,
    /Users/blyth/chroma/chroma/generator/g4gen.py:            g4material.AddElement(G4Element.GetElement(element_name, True),
    /Users/blyth/chroma/chroma/generator/g4gen.py:        prop_table = G4MaterialPropertiesTable()
    /Users/blyth/chroma/chroma/generator/g4gen.py:                mass = G4ParticleTable.GetParticleTable().FindParticle(vertex.particle_name).GetPDGMass()
    /Users/blyth/chroma/chroma/generator/g4gen.py:                self.particle_gun.SetParticlePosition(G4ThreeVector(*pos)*mm)
    /Users/blyth/chroma/chroma/generator/g4gen.py:                self.particle_gun.SetParticleMomentumDirection(G4ThreeVector(*dir).unit())
    /Users/blyth/chroma/chroma/generator/g4gen.py:                    self.particle_gun.SetParticlePolarization(G4ThreeVector(*vertex.pol).unit())
    /Users/blyth/chroma/chroma/generator/photon.py:class G4GeneratorProcess(multiprocessing.Process):
    /Users/blyth/chroma/chroma/generator/photon.py:        gen = g4gen.G4Generator(self.material, seed=self.seed)
    /Users/blyth/chroma/chroma/generator/photon.py:class G4ParallelGenerator(object):
    /Users/blyth/chroma/chroma/generator/photon.py:        self.processes = [ G4GeneratorProcess(i, material, self.vertex_address, self.photon_address, seed=base_seed + i) for i in xrange(nprocesses) ]
    /Users/blyth/chroma/chroma/sim.py:            self.photon_generator = generator.photon.G4ParallelGenerator(geant4_processes, detector.detector_material, base_seed=self.seed)
    simon:chroma blyth$ 



Boost python C++ `_g4chroma`
-----------------------------

* `src/mute.cc` control G4 stdout
* `src/G4chroma.hh`
* `src/G4chroma.cc`    # ChromaPhysicsList and PhotonTrackingAction

::

    simon:chroma blyth$ find . -name '*.*' -exec grep -H _g4chroma {} \;
    ./chroma/generator/g4gen.py:from chroma.generator import _g4chroma
    ./chroma/generator/g4gen.py:        self.physics_list = _g4chroma.ChromaPhysicsList()
    ./chroma/generator/g4gen.py:        self.tracking_action = _g4chroma.PhotonTrackingAction()
    ./setup.py:        Extension('chroma.generator._g4chroma',
    ./src/G4chroma.cc:BOOST_PYTHON_MODULE(_g4chroma)


chroma/setup.py::

     59 setup(
     60     name = 'Chroma',
     61     version = '0.5',
     62     packages = find_packages(),
     63     include_package_data=True,
     64 
     65     scripts = ['bin/chroma-sim', 'bin/chroma-cam',
     66                'bin/chroma-geo', 'bin/chroma-bvh',
     67                'bin/chroma-server'],
     68     ext_modules = [
     69         Extension('chroma.generator._g4chroma',
     70                   ['src/G4chroma.cc'],
     71                   include_dirs=include_dirs,
     72                   extra_compile_args=geant4_cflags,
     73                   extra_link_args=geant4_libs+clhep_libs,
     74                   extra_objects=extra_objects,
     75                   libraries=libraries,
     76                   ),
     77         Extension('chroma.generator.mute',
     78                   ['src/mute.cc'],
     79                   include_dirs=include_dirs,
     80                   extra_compile_args=geant4_cflags,
     81                   extra_link_args=geant4_libs+clhep_libs,
     82                   extra_objects=extra_objects,
     83                   libraries=libraries),
     84         ],
     85 
     86     setup_requires = ['pyublas'],
     87     install_requires = ['uncertainties','pyzmq-static','spnav', 'pycuda',
     88                         'numpy>=1.6', 'pygame', 'nose', 'sphinx', 'unittest2'],
     89     test_suite = 'nose.collector',
     90 
     91 )






ChromaPhysicsList
~~~~~~~~~~~~~~~~~~~

::

     05 ChromaPhysicsList::ChromaPhysicsList():  G4VModularPhysicsList()
     06 {
     07   // default cut value  (1.0mm) 
     08   defaultCutValue = 1.0*mm;
     09 
     10   // General Physics
     11   RegisterPhysics( new G4EmPenelopePhysics(0) );
     12   // Optical Physics
     13   G4OpticalPhysics* opticalPhysics = new G4OpticalPhysics();
     14   RegisterPhysics( opticalPhysics );
     15 }


`chroma/generator/g4gen.py`::

     19 class G4Generator(object):
     20     def __init__(self, material, seed=None):
     21         """Create generator to produce photons inside the specified material.
     22 
     23            material: chroma.geometry.Material object with density, 
     24                      composition dict and refractive_index.
     25 
     26                      composition dictionary should be 
     27                         { element_symbol : fraction_by_weight, ... }.
     28 
     29            seed: int, *optional*
     30                Random number generator seed for HepRandom. If None, generator
     31                is not seeded.
     32         """
     33         if seed is not None:
     34             HepRandom.setTheSeed(seed)
     35 
     36         g4py.NISTmaterials.Construct()
     37         g4py.ezgeom.Construct()
     38         self.physics_list = _g4chroma.ChromaPhysicsList()
     39         gRunManager.SetUserInitialization(self.physics_list)



PhotonTrackingAction
~~~~~~~~~~~~~~~~~~~~~


One-by-one collection and G4 `fStopAndKill` of optical photons.

Boost python module `_g4chroma` implementation in C++ providing a G4UserTrackingAction *PhotonTrackingAction* 
that collects opticalphotons and provides accessors to them, and snuffs them out with *fStopAndKill* ::

    105 void PhotonTrackingAction::PreUserTrackingAction(const G4Track *track)
    106 {
    107   G4ParticleDefinition *particle = track->GetDefinition();
    108   if (particle->GetParticleName() == "opticalphoton") {
    109     pos.push_back(track->GetPosition()/mm);
    110     dir.push_back(track->GetMomentumDirection());
    111     pol.push_back(track->GetPolarization());
    112     wavelength.push_back( (h_Planck * c_light / track->GetKineticEnergy()) / nanometer );
    113     t0.push_back(track->GetGlobalTime() / ns);
    114     const_cast<G4Track *>(track)->SetTrackStatus(fStopAndKill);
    115   }
    116 }


Subsequently uses pyublas to give photon pos/dir/.. etc numpy arrays 
directly from the `tracking_action`

chroma/generator/g4gen.py::

     79     def _extract_photons_from_tracking_action(self, sort=True):
     80         n = self.tracking_action.GetNumPhotons()
     81         pos = np.zeros(shape=(n,3), dtype=np.float32)
     82         pos[:,0] = self.tracking_action.GetX()
     83         pos[:,1] = self.tracking_action.GetY()
     84         pos[:,2] = self.tracking_action.GetZ()
     ..
     07 
     08         return Photons(pos, dir, pol, wavelengths, t0)


chroma/generator/g4gen.py::

    110     def generate_photons(self, vertices, mute=False):
    111         """Use GEANT4 to generate photons produced by propagating `vertices`.
    112            
    113         Args:
    114             vertices: list of event.Vertex objects
    115                 List of initial particle vertices.
    116 
    117             mute: bool
    118                 Disable GEANT4 output to console during generation.  (GEANT4 can
    119                 be quite chatty.)
    120 
    121         Returns:
    122             photons: event.Photons
    123                 Photon vertices generated by the propagation of `vertices`.
    124         """
    125         if mute:
    126             g4mute()
    127 
    128         photons = None
    129 
    130         try:
    131             for vertex in vertices:
    132                 self.particle_gun.SetParticleByName(vertex.particle_name)
    133                 mass = G4ParticleTable.GetParticleTable().FindParticle(vertex.particle_name).GetPDGMass()
    134                 total_energy = vertex.ke*MeV + mass
    135                 self.particle_gun.SetParticleEnergy(total_energy)
    136 
    137                 # Must be float type to call GEANT4 code
    138                 pos = np.asarray(vertex.pos, dtype=np.float64)
    139                 dir = np.asarray(vertex.dir, dtype=np.float64)
    140 
    141                 self.particle_gun.SetParticlePosition(G4ThreeVector(*pos)*mm)
    142                 self.particle_gun.SetParticleMomentumDirection(G4ThreeVector(*dir).unit())
    143                 self.particle_gun.SetParticleTime(vertex.t0*ns)
    144 
    145                 if vertex.pol is not None:
    146                     self.particle_gun.SetParticlePolarization(G4ThreeVector(*vertex.pol).unit())
    147 
    148                 self.tracking_action.Clear()
    149                 gRunManager.BeamOn(1)
    150 
    151                 if photons is None:
    152                     photons = self._extract_photons_from_tracking_action()
    153                 else:
    154                     photons += self._extract_photons_from_tracking_action()
    155         finally:
    156             if mute:
    157                 g4unmute()
    158 
    159         return photons



* NB photons from all vertices passed are combined into a single `Photons` instance


Attempt to add a PhotonSteppingAction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Breaks g4py and even after remove the PhotonSteppingAction it stays broken!
Too many layers, such fragility puts me off g4py, boost_python::

    (chroma_env)simon:cfg4 blyth$ lldb $(which python) g4gen.py 
    Current executable set to '/usr/local/env/chroma_env/bin/python' (x86_64).
    (lldb) r
    Process 48280 launched: '/usr/local/env/chroma_env/bin/python' (x86_64)
    Process 48280 stopped
    * thread #1: tid = 0x1545fb, 0x0000000106dea134 _g4chroma.so`boost::python::objects::polymorphic_id_generator<ChromaPhysicsList>::execute(void*) [inlined] std::type_info::name(this=0x0000000000000032) const at typeinfo:98, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x3a)
        frame #0: 0x0000000106dea134 _g4chroma.so`boost::python::objects::polymorphic_id_generator<ChromaPhysicsList>::execute(void*) [inlined] std::type_info::name(this=0x0000000000000032) const at typeinfo:98
       95       _LIBCPP_INLINE_VISIBILITY
       96       const char* name() const _NOEXCEPT
       97   #ifndef _LIBCPP_NONUNIQUE_RTTI_BIT
    -> 98           {return __type_name;}
       99   #else
       100          {return reinterpret_cast<const char*>(__type_name & ~_LIBCPP_NONUNIQUE_RTTI_BIT);}
       101  #endif
    (lldb) bt
    * thread #1: tid = 0x1545fb, 0x0000000106dea134 _g4chroma.so`boost::python::objects::polymorphic_id_generator<ChromaPhysicsList>::execute(void*) [inlined] std::type_info::name(this=0x0000000000000032) const at typeinfo:98, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x3a)
      * frame #0: 0x0000000106dea134 _g4chroma.so`boost::python::objects::polymorphic_id_generator<ChromaPhysicsList>::execute(void*) [inlined] std::type_info::name(this=0x0000000000000032) const at typeinfo:98
        frame #1: 0x0000000106dea134 _g4chroma.so`boost::python::objects::polymorphic_id_generator<ChromaPhysicsList>::execute(void*) [inlined] boost::python::type_info::type_info(id=0x0000000000000032) at type_id.hpp:122
        frame #2: 0x0000000106dea134 _g4chroma.so`boost::python::objects::polymorphic_id_generator<ChromaPhysicsList>::execute(void*) [inlined] boost::python::type_info::type_info(__x=0x00007fff5fbfdba1, id=0x0000000000000032) at type_id.hpp:128
        frame #3: 0x0000000106dea134 _g4chroma.so`boost::python::objects::polymorphic_id_generator<ChromaPhysicsList>::execute(p_=<unavailable>) + 20 at inheritance.hpp:43
        frame #4: 0x000000010355903b libboost_python.dylib`boost::(anonymous namespace)::convert_type(void*, boost::python::type_info, boost::python::type_info, bool) + 395
        frame #5: 0x00000001035534df libboost_python.dylib`boost::python::objects::find_instance_impl(_object*, boost::python::type_info, bool) + 79
        frame #6: 0x000000010354ea67 libboost_python.dylib`boost::python::converter::get_lvalue_from_python(_object*, boost::python::converter::registration const&) + 23
        frame #7: 0x000000010630259e G4run.so`boost::python::detail::caller_arity<2u>::impl<void (G4RunManager::*)(G4VUserPhysicsList*), boost::python::default_call_policies, boost::mpl::vector3<void, G4RunManager&, G4VUserPhysicsList*> >::operator()(_object*, _object*) + 78
        frame #8: 0x000000010355611e libboost_python.dylib`boost::python::objects::function::call(_object*, _object*) const + 766
        frame #9: 0x000000010355826a libboost_python.dylib`boost::detail::function::void_function_ref_invoker0<boost::python::objects::(anonymous namespace)::bind_return, void>::invoke(boost::detail::function::function_buffer&) + 26
        frame #10: 0x000000010355dfc5 libboost_python.dylib`boost::python::handle_exception_impl(boost::function0<void>) + 85
        frame #11: 0x0000000103557c73 libboost_python.dylib`function_call + 83
        frame #12: 0x0000000100011ca3 Python`PyObject_Call + 99
        frame #13: 0x00000001000afd7d Python`PyEval_EvalFrameEx + 15981
        frame #14: 0x00000001000abc9d Python`PyEval_EvalCodeEx + 1725
        frame #15: 0x000000010003783c Python`function_call + 364
        frame #16: 0x0000000100011ca3 Python`PyObject_Call + 99
        frame #17: 0x000000010001ef96 Python`instancemethod_call + 182
        frame #18: 0x0000000100011ca3 Python`PyObject_Call + 99
        frame #19: 0x000000010006d5dd Python`slot_tp_init + 141
        frame #20: 0x0000000100067942 Python`type_call + 354
        frame #21: 0x0000000100011ca3 Python`PyObject_Call + 99
        frame #22: 0x00000001000afd7d Python`PyEval_EvalFrameEx + 15981
        frame #23: 0x00000001000abc9d Python`PyEval_EvalCodeEx + 1725
        frame #24: 0x00000001000ab5d6 Python`PyEval_EvalCode + 54
        frame #25: 0x00000001000d5194 Python`PyRun_FileExFlags + 164
        frame #26: 0x00000001000d4d11 Python`PyRun_SimpleFileExFlags + 769
        frame #27: 0x00000001000eaa5e Python`Py_Main + 3070
        frame #28: 0x00007fff87c165fd libdyld.dylib`start + 1
        frame #29: 0x00007fff87c165fd libdyld.dylib`start + 1
    (lldb) 



