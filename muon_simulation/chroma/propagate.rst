Propagate
==========

python side
-------------

::

    096     @profile_if_possible
    097     def propagate(self, gpu_geometry, rng_states, nthreads_per_block=64,
    098                   max_blocks=1024, max_steps=10, use_weights=False,
    099                   scatter_first=0):
    ...
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





cuda side
-----------

::

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
    // shared geometry between threads
    //
    126 
    127     __syncthreads();
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



Photon enum::

     47 enum
     48 {
     49     NO_HIT           = 0x1 << 0,
     50     BULK_ABSORB      = 0x1 << 1,
     51     SURFACE_DETECT   = 0x1 << 2,
     52     SURFACE_ABSORB   = 0x1 << 3,
     53     RAYLEIGH_SCATTER = 0x1 << 4,
     54     REFLECT_DIFFUSE  = 0x1 << 5,
     55     REFLECT_SPECULAR = 0x1 << 6,
     56     SURFACE_REEMIT   = 0x1 << 7,
     57     SURFACE_TRANSMIT = 0x1 << 8,
     58     BULK_REEMIT      = 0x1 << 9,
     59     NAN_ABORT        = 0x1 << 31
     60 }; // processes




