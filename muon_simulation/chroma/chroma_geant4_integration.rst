Chroma Geant4 Interaction
==========================

Boost python C++ `_g4chroma`
-----------------------------

* `src/mute.cc` control G4 stdout
* `src/G4chroma.hh`
* `src/G4chroma.cc`

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


::

    simon:chroma blyth$ find . -name '*.*' -exec grep -H _g4chroma {} \;
    ./chroma/generator/g4gen.py:from chroma.generator import _g4chroma
    ./chroma/generator/g4gen.py:        self.physics_list = _g4chroma.ChromaPhysicsList()
    ./chroma/generator/g4gen.py:        self.tracking_action = _g4chroma.PhotonTrackingAction()
    ./setup.py:        Extension('chroma.generator._g4chroma',
    ./src/G4chroma.cc:BOOST_PYTHON_MODULE(_g4chroma)



geometry from CUDA photon propagation, in `photon.h`::

    584 __device__ int
    585 propagate_at_surface(Photon &p, State &s, curandState &rng, Geometry *geometry,
    586                      bool use_weights=false)
    587 {
    588     Surface *surface = geometry->surfaces[s.surface_index];
    589 
    590     if (surface->model == SURFACE_COMPLEX)
    591         return propagate_complex(p, s, rng, surface, use_weights);
    592     else if (surface->model == SURFACE_WLS)
    593         return propagate_at_wls(p, s, rng, surface, use_weights);
    594     else {
    595         // use default surface model: do a combination of specular and
    596         // diffuse reflection, detection, and absorption based on relative
    597         // probabilties
    598 
    599         // since the surface properties are interpolated linearly, we are
    600         // guaranteed that they still sum to 1.0.
    601         float detect = interp_property(surface, p.wavelength, surface->detect);
    602         float absorb = interp_property(surface, p.wavelength, surface->absorb);
    603         float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    604         float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    605 
    606         float uniform_sample = curand_uniform(&rng);



::

    simon:cuda blyth$ grep __shared__ *.*
    bvh.cu:    __shared__ unsigned long long min_area[128];
    bvh.cu:    __shared__ unsigned long long adjacent_area;
    daq.cu:    __shared__ int photon_id;
    daq.cu:    __shared__ int triangle_id;
    daq.cu:    __shared__ int solid_id;
    daq.cu:    __shared__ int channel_index;
    daq.cu:    __shared__ unsigned int history;
    daq.cu:    __shared__ float photon_time;
    daq.cu:    __shared__ float weight;
    mesh.h:    __shared__ Geometry sg;
    pdf.cu:    __shared__ float distance_table[1000];
    pdf.cu:    __shared__ unsigned int *work_queue;
    pdf.cu:    __shared__ int queue_items;
    pdf.cu:    __shared__ int channel_id;
    pdf.cu:    __shared__ float channel_event_time;
    pdf.cu:    __shared__ int distance_table_len;
    pdf.cu:    __shared__ int offset;
    propagate.cu:    __shared__ unsigned int counter;
    propagate.cu:    __shared__ Geometry sg;
    render.cu:    __shared__ Geometry sg;
    simon:cuda blyth$ 


::

    simon:cuda blyth$ grep sync *.*      
    bvh.cu:    __syncthreads();
    bvh.cu:    __syncthreads();
    bvh.cu:    __syncthreads();
    daq.cu:    __syncthreads();
    mesh.h:    __syncthreads();
    pdf.cu:    __syncthreads();
    pdf.cu:    __syncthreads();
    pdf.cu:    __syncthreads();
    pdf.cu:    __syncthreads();
    propagate.cu:    __syncthreads();
    propagate.cu:    __syncthreads();
    propagate.cu:    __syncthreads();
    render.cu:    __syncthreads();
    simon:cuda blyth$ 



