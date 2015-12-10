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




