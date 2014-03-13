Photon Propagation
===================

python side GPUPhotons
-----------------------

#. transports numpy arrays of photons to/from the GPU
#. compiles the `propagate.cu`
#. interesting comment regards single stepping the propagation, to understand what its upto 


`chroma/gpu/photon.py` GPUPhotons::

    096     @profile_if_possible
    097     def propagate(self, gpu_geometry, rng_states, nthreads_per_block=64,
    098                   max_blocks=1024, max_steps=10, use_weights=False,
    099                   scatter_first=0):
    100         """Propagate photons on GPU to termination or max_steps, whichever
    101         comes first.
    102 
    103         May be called repeatedly without reloading photon information if
    104         single-stepping through photon history.
    105 
    106         ..warning::
    107             `rng_states` must have at least `nthreads_per_block`*`max_blocks`
    108             number of curandStates.
    109         """
    110         nphotons = self.pos.size
    111         step = 0
    112         input_queue = np.empty(shape=nphotons+1, dtype=np.uint32)
    113         input_queue[0] = 0
    114         # Order photons initially in the queue to put the clones next to each other
    115         for copy in xrange(self.ncopies):
    116             input_queue[1+copy::self.ncopies] = np.arange(self.true_nphotons, dtype=np.uint32) + copy * self.true_nphotons
    117         input_queue_gpu = ga.to_gpu(input_queue)
    118         output_queue = np.zeros(shape=nphotons+1, dtype=np.uint32)
    119         output_queue[0] = 1
    120         output_queue_gpu = ga.to_gpu(output_queue)
    121 
    122         while step < max_steps:
    123             # Just finish the rest of the steps if the # of photons is low
    124             if nphotons < nthreads_per_block * 16 * 8 or use_weights:
    125                 nsteps = max_steps - step
    126             else:
    127                 nsteps = 1
    128 
    129             for first_photon, photons_this_round, blocks in \
    130                     chunk_iterator(nphotons, nthreads_per_block, max_blocks):
    131                 self.gpu_funcs.propagate(
                                 np.int32(first_photon), 
                                 np.int32(photons_this_round), 
                                 input_queue_gpu[1:], 
                                 output_queue_gpu, 
                                 rng_states, 
                                 self.pos, self.dir, self.wavelengths, self.pol, self.t, self.flags, 
                                 self.last_hit_triangles, 
                                 self.weights, 
                                 np.int32(nsteps),   ## CAUTION thats max_steps on cuda side
                                 np.int32(use_weights), 
                                 np.int32(scatter_first), 
                                 gpu_geometry.gpudata, 

                                 block=(nthreads_per_block,1,1), grid=(blocks, 1))
    ###
    ###      propagation chunks marshalled by passing first_photon/photons_this_round indices
    ###      that point into the photon arrays that were loaded at instantiation
    ###
    132 
    133             step += nsteps
    134             scatter_first = 0 # Only allow non-zero in first pass
    135 
    136             if step < max_steps:
    137                 temp = input_queue_gpu
    138                 input_queue_gpu = output_queue_gpu
    139                 output_queue_gpu = temp
    140                 # Assign with a numpy array of length 1 to silence
    141                 # warning from PyCUDA about setting array with different strides/storage orders.
    142                 output_queue_gpu[:1].set(np.ones(shape=1, dtype=np.uint32))
    143                 nphotons = input_queue_gpu[:1].get()[0] - 1

    ///         stick the surviving propagated photons in output_queue into input_queue  

    144 
    145         if ga.max(self.flags).get() & (1 << 31):
    146             print >>sys.stderr, "WARNING: ABORTED PHOTONS"
    147         cuda.Context.get_current().synchronize()




cuda entry points
-------------------

::

    simon:cuda blyth$ grep -l blockIdx *.*
    bvh.cu
    daq.cu
    hybrid_render.cu
    mesh.h
    pdf.cu
    propagate.cu
    random.h
    render.cu
    tools.cu
    transform.cu


cuda propagate
----------------

Entry point is **propagate**, communication via numpy arrays curtesy of pycuda. 


::

    (chroma_env)delta:chroma blyth$ find . -name '*.cu' -exec grep -H propagate {} \;
    ./chroma/cuda/hybrid_render.cu: command = propagate_to_boundary(p, s, rng);
    ./chroma/cuda/hybrid_render.cu:     command = propagate_at_surface(p, s, rng, g);
    ./chroma/cuda/hybrid_render.cu: propagate_at_boundary(p, s, rng);
    ./chroma/cuda/propagate.cu:propagate(int first_photon, int nthreads, unsigned int *input_queue,
    ./chroma/cuda/propagate.cu: command = propagate_to_boundary(p, s, rng, use_weights, scatter_first);
    ./chroma/cuda/propagate.cu:   command = propagate_at_surface(p, s, rng, g, use_weights);
    ./chroma/cuda/propagate.cu: propagate_at_boundary(p, s, rng);
    ./chroma/cuda/propagate.cu:} // propagate
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ find . -name '*.h' -exec grep -H propagate {} \;
    ./chroma/cuda/photon.h:enum { BREAK, CONTINUE, PASS }; // return value from propagate_to_boundary
    ./chroma/cuda/photon.h:int propagate_to_boundary(Photon &p, State &s, curandState &rng,
    ./chroma/cuda/photon.h:} // propagate_to_boundary
    ./chroma/cuda/photon.h:propagate_at_boundary(Photon &p, State &s, curandState &rng)
    ./chroma/cuda/photon.h:} // propagate_at_boundary
    ./chroma/cuda/photon.h:propagate_at_specular_reflector(Photon &p, State &s)
    ./chroma/cuda/photon.h:} // propagate_at_specular_reflector
    ./chroma/cuda/photon.h:propagate_at_diffuse_reflector(Photon &p, State &s, curandState &rng)
    ./chroma/cuda/photon.h:} // propagate_at_diffuse_reflector
    ./chroma/cuda/photon.h:propagate_complex(Photon &p, State &s, curandState &rng, Surface* surface, bool use_weights=false)
    ./chroma/cuda/photon.h:    // calculate s polarization fraction, identical to propagate_at_boundary
    ./chroma/cuda/photon.h:            return propagate_at_diffuse_reflector(p, s, rng);
    ./chroma/cuda/photon.h:            return propagate_at_specular_reflector(p, s);
    ./chroma/cuda/photon.h:} // propagate_complex
    ./chroma/cuda/photon.h:propagate_at_wls(Photon &p, State &s, curandState &rng, Surface *surface, bool use_weights=false)
    ./chroma/cuda/photon.h:            return propagate_at_specular_reflector(p, s);
    ./chroma/cuda/photon.h:            return propagate_at_diffuse_reflector(p, s, rng);
    ./chroma/cuda/photon.h:} // propagate_at_wls
    ./chroma/cuda/photon.h:propagate_at_surface(Photon &p, State &s, curandState &rng, Geometry *geometry,
    ./chroma/cuda/photon.h:        return propagate_complex(p, s, rng, surface, use_weights);
    ./chroma/cuda/photon.h:        return propagate_at_wls(p, s, rng, surface, use_weights);
    ./chroma/cuda/photon.h:            return propagate_at_diffuse_reflector(p, s, rng);
    ./chroma/cuda/photon.h:            return propagate_at_specular_reflector(p, s);
    ./chroma/cuda/photon.h:} // propagate_at_surface
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ 









* **self.flags** on corresponds to  **histories** array 
* **id** identifies the CUDA thread, corresponding to a single photon
* photon parameters indexed into the arrays with `photon_id` 


`chroma/cuda/propagate.cu`::

    112 __global__ void
    113 propagate(int first_photon, int nthreads, unsigned int *input_queue,
    114       unsigned int *output_queue, curandState *rng_states,
    115       float3 *positions, float3 *directions,
    116       float *wavelengths, float3 *polarizations,
    117       float *times, unsigned int *histories,
    118       int *last_hit_triangles, float *weights,
    119       int max_steps, int use_weights, int scatter_first,
    120       Geometry *g)
    121 {
    122     __shared__ Geometry sg;
    123 
    124     if (threadIdx.x == 0)
    125     sg = *g;
    //
    // only grab geometry for the first thread 
    // shared geometry between threads
    //
    126 
    127     __syncthreads();

    //
    //    https://devtalk.nvidia.com/default/topic/379871/cuda-programming-and-performance/semantics-of-__syncthreads/
    //
    //    What more do you want? __syncthreads() is you garden variety thread barrier.
    //    Any thread reaching the barrier waits until all of the other threads in that
    //    block also reach it. It is designed for avoiding race conditions when loading
    //    shared memory, and the compiler will not move memory reads/writes around a
    //    __syncthreads(). 
    //
    //    It is nothing more and nothing less. Unless you are writing to a shared memory
    //    location in thread i then reading that same location in thread j, you don't
    //    need __syncthreads().
    //
    //
    //    PRESUMABLY THAT ENSURES ALL THREADS/PHOTONS SEE THE SAME SHARED GEOMETRY,
    //    BY WAITING FOR THREAD 0 TO COMPLETE SETTING THAT UP BEFORE PROCEEDING
    //
    //    BUT NEED TO UNDERSTAND MORE CLEALY WHAT CONSTITUTES A BLOCK FOR 
    //    PYCUDA/CHROMA
    //

    128 
    129     int id = blockIdx.x*blockDim.x + threadIdx.x;
    //
    //  id points at the single photon to propagate in this parallel thread
    //
    130 
    131     if (id >= nthreads)
    132     return;
    133 
    134     g = &sg;
    135 
    136     curandState rng = rng_states[id];
    137 
    138     int photon_id = input_queue[first_photon + id];
    139 
    140     Photon p;
    141     p.position = positions[photon_id];
    142     p.direction = directions[photon_id];
    143     p.direction /= norm(p.direction);
    144     p.polarization = polarizations[photon_id];
    145     p.polarization /= norm(p.polarization);
    146     p.wavelength = wavelengths[photon_id];
    147     p.time = times[photon_id];
    148     p.last_hit_triangle = last_hit_triangles[photon_id];
    149     p.history = histories[photon_id];
    150     p.weight = weights[photon_id];
    151 
    152     if (p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT))
    153     return;
    ///              DEAD ALREADY, AS INDICATED BY THE HISTORY FLAGS
    154 
    155     State s;
    156 
    157     int steps = 0;
    158     while (steps < max_steps) {
    159     steps++;
    160 
    161     int command;
    162 
    163     // check for NaN and fail
    164     if (isnan(p.direction.x*p.direction.y*p.direction.z*p.position.x*p.position.y*p.position.z)) {
    165         p.history |= NO_HIT | NAN_ABORT;
    166         break;
    167     }
    168 
    169     fill_state(s, p, g);
    170 
    171     if (p.last_hit_triangle == -1)
    172         break;
    173 
    174     command = propagate_to_boundary(p, s, rng, use_weights, scatter_first);
    //
    //      propagate_* only changes p (?) refering to state s   
    //
    175     scatter_first = 0; // Only use the scatter_first value once
    176 
    177     if (command == BREAK)
    178         break;
    179 
    180     if (command == CONTINUE)
    181         continue;
    182 
    183     if (s.surface_index != -1) {
    184       command = propagate_at_surface(p, s, rng, g, use_weights);
    185 
    186         if (command == BREAK)
    187         break;
    188 
    189         if (command == CONTINUE)
    190         continue;
    191     }
    192 
    193     propagate_at_boundary(p, s, rng);
    194 
    195     } // while (steps < max_steps)
    196 
    197     rng_states[id] = rng;
    198     positions[photon_id] = p.position;
    199     directions[photon_id] = p.direction;
    200     polarizations[photon_id] = p.polarization;
    201     wavelengths[photon_id] = p.wavelength;
    202     times[photon_id] = p.time;
    203     histories[photon_id] = p.history;
    204     last_hit_triangles[photon_id] = p.last_hit_triangle;
    205     weights[photon_id] = p.weight;
    206 
    207     // Not done, put photon in output queue
    208     if ((p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT)) == 0) {
    //
    //       the photon lives on thanks to 
    //            RAYLEIGH_SCATTER REFLECT_DIFFUSE REFLECT_SPECULAR SURFACE_REEMIT SURFACE_TRANSMIT BULK_REEMIT   
    //
    //
    209     int out_idx = atomicAdd(output_queue, 1);
    210     output_queue[out_idx] = photon_id;
    //
    //     http://supercomputingblog.com/cuda/cuda-tutorial-4-atomic-operations/
    //
    //         This atomicAdd function can be called within a kernel. When a thread executes this operation, a memory address is read, 
    //         has the value of val added to it, and the result is written back to memory. 
    //         The original value of the memory at location ?address? is returned to the thread.
    //
    211     }
    212 } // propagate


`chroma/cuda/geometry_types.h`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     16 enum { SURFACE_DEFAULT, SURFACE_COMPLEX, SURFACE_WLS };
     17 
     18 struct Surface
     19 {
     20     float *detect;
     21     float *absorb;
     22     float *reemit;
     23     float *reflect_diffuse;
     24     float *reflect_specular;
     //
     // eta,k, only used in SURFACE_COMPLEX ?  
     //
     25     float *eta;
     26     float *k;
     27     float *reemission_cdf;
     28 
     29     unsigned int model;
     30     unsigned int n;
     31     unsigned int transmissive;
     32     float step;
     33     float wavelength0;
     34     float thickness;
     35 };


`chroma/cuda/photon.h`
~~~~~~~~~~~~~~~~~~~~~~~~

::

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
    607 
    608         if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD
    609         && absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) {
    610             // Prevent absorption and reweight accordingly
    611             float survive = 1.0f - absorb;
    612             absorb = 0.0f;
    613             p.weight *= survive;
    614 
    615             // Renormalize remaining probabilities
    616             detect /= survive;
    617             reflect_diffuse /= survive;
    618             reflect_specular /= survive;
    619         }
    620 
    621         if (use_weights && detect > 0.0f) {
    622             p.history |= SURFACE_DETECT;
    623             p.weight *= detect;
    624             return BREAK;
    625         }
    626 
    627         if (uniform_sample < absorb) {
    628             p.history |= SURFACE_ABSORB;
    629             return BREAK;
    630         }
    631         else if (uniform_sample < absorb + detect) {
    632             p.history |= SURFACE_DETECT;
    633             return BREAK;
    634         }
    635         else if (uniform_sample < absorb + detect + reflect_diffuse)
    636             return propagate_at_diffuse_reflector(p, s, rng);
    637         else
    638             return propagate_at_specular_reflector(p, s);
    639     }
    640 
    641 } // propagate_at_surface
    642 
    643 #endif
    644 
    ...
    342 __device__ int
    343 propagate_at_specular_reflector(Photon &p, State &s)
    344 {
    345     float incident_angle = get_theta(s.surface_normal, -p.direction);
    346     float3 incident_plane_normal = cross(p.direction, s.surface_normal);
    347     incident_plane_normal /= norm(incident_plane_normal);
    348 
    349     p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);
    350 
    351     p.history |= REFLECT_SPECULAR;
    352 
    353     return CONTINUE;
    354 } // propagate_at_specular_reflector
    355 
    356 __device__ int
    357 propagate_at_diffuse_reflector(Photon &p, State &s, curandState &rng)
    358 {
    359     float ndotv;
    360     do {
    361     p.direction = uniform_sphere(&rng);
    362     ndotv = dot(p.direction, s.surface_normal);
    363     if (ndotv < 0.0f) {
    364         p.direction = -p.direction;
    365         ndotv = -ndotv;
    366     }
    367     } while (! (curand_uniform(&rng) < ndotv) );
    368 
    369     p.polarization = cross(uniform_sphere(&rng), p.direction);
    370     p.polarization /= norm(p.polarization);
    371 
    372     p.history |= REFLECT_DIFFUSE;
    373 
    374     return CONTINUE;
    375 } // propagate_at_diffuse_reflector
    ...
    377 __device__ int
    378 propagate_complex(Photon &p, State &s, curandState &rng, Surface* surface, bool use_weights=false)
    379 {
    380     float detect = interp_property(surface, p.wavelength, surface->detect);
    381     float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    382     float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    383     float n2_eta = interp_property(surface, p.wavelength, surface->eta);
    384     float n2_k = interp_property(surface, p.wavelength, surface->k);
    385 
    386     // thin film optical model, adapted from RAT PMT optical model by P. Jones
    387     cuFloatComplex n1 = make_cuFloatComplex(s.refractive_index1, 0.0f);
    388     cuFloatComplex n2 = make_cuFloatComplex(n2_eta, n2_k);
    389     cuFloatComplex n3 = make_cuFloatComplex(s.refractive_index2, 0.0f);
    390 


* `chroma/doc/source/surface.rst`



surface_index
~~~~~~~~~~~~~~~~

::

    g4pb:cuda blyth$ pwd
    /usr/local/env/chroma/chroma/cuda
    g4pb:cuda blyth$ grep surface_index *.*
    hybrid_render.cu:       if (s.surface_index != -1) {
    photon.h:    int surface_index;
    photon.h:    s.surface_index = convert(0xFF & (material_code >> 8));
    photon.h:    Surface *surface = geometry->surfaces[s.surface_index];
    propagate.cu:   if (s.surface_index != -1) {
    g4pb:cuda blyth$ 








