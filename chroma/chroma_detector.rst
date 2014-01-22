Chroma Modeling of Sensitive Detectors 
=========================================

.. contents:: :local:

Questions 
-----------

#. How hits are handled in Chroma ?

chroma/geometry.py
---------------------

::

        103 class Solid(object):
        104     """Solid object attaches materials, surfaces, and colors to each triangle
        105     in a Mesh object."""
        106     def __init__(self, mesh, material1=None, material2=None, surface=None, color=0x33ffffff):
        107         self.mesh = mesh


        /// collection of solids, maybe with a material flagged as a detector_material

        261 class Geometry(object):
        262     "Geometry object."
        263     def __init__(self, detector_material=None):
        264         self.detector_material = detector_material
        265         self.solids = []
        266         self.solid_rotations = []
        267         self.solid_displacements = []
        268         self.bvh = None


chroma/detector.py
-------------------

::

     05 class Detector(Geometry):
     06     '''A Detector is a subclass of Geometry that allows some Solids
     07     to be marked as photon detectors, which we will suggestively call
     08     "PMTs."  Each detector is imagined to be connected to an electronics
     09     channel that records a hit time and charge.
     10 
     11     Each PMT has two integers identifying it: a channel index and a
     12     channel ID.  When all of the channels in the detector are stored
     13     in a Numpy array, they will be stored in index order.  Channel
     14     indices star from zero and have no gaps.  Channel ID numbers are
     15     arbitrary integers that identify a PMT uniquely in a stable way.
     16     They are written out to disk when using the Chroma ROOT format,
     17     and are used when reading events back in to map channels back
     18     into the correct array index.
     19 
     20     For now, all the PMTs share a single set of time and charge
     21     distributions.  In the future, this will be generalized to
     22     allow per-channel distributions.
     23     '''
     24 
     25     def __init__(self, detector_material=None):
     26         Geometry.__init__(self, detector_material=detector_material)
     27 
     28         # Using numpy arrays here to allow for fancy indexing
     29         self.solid_id_to_channel_index = np.zeros(0, dtype=np.int32)
     30         self.channel_index_to_solid_id = np.zeros(0, dtype=np.int32)
     31 
     32         self.channel_index_to_channel_id = np.zeros(0, dtype=np.int32)
     33 
     34         # If the ID numbers are arbitrary, we can't treat them
     35         # as array indices, so have to use a dictionary
     36         self.channel_id_to_channel_index = {}
     37 
     38         # zero time and unit charge distributions
     39         self.time_cdf = (np.array([-0.01, 0.01]), np.array([0.0, 1.0]))
     40         self.charge_cdf = (np.array([0.99, 1.00]), np.array([0.0, 1.0]))



chroma/gpu/detector.py
-----------------------

::

     14 class GPUDetector(GPUGeometry):
     15     def __init__(self, detector, wavelengths=None, print_usage=False):
     16         GPUGeometry.__init__(self, detector, wavelengths=wavelengths, print_usage=False)
     17         self.solid_id_to_channel_index_gpu = \
     18             ga.to_gpu(detector.solid_id_to_channel_index.astype(np.int32))
     19         self.nchannels = detector.num_channels()
     20 
     21 
     22         self.time_cdf_x_gpu = ga.to_gpu(detector.time_cdf[0].astype(np.float32))
     23         self.time_cdf_y_gpu = ga.to_gpu(detector.time_cdf[1].astype(np.float32))
     24 
     25         self.charge_cdf_x_gpu = ga.to_gpu(detector.charge_cdf[0].astype(np.float32))
     26         self.charge_cdf_y_gpu = ga.to_gpu(detector.charge_cdf[1].astype(np.float32))
     27 
     28         detector_source = get_cu_source('detector.h')
     29         detector_struct_size = characterize.sizeof('Detector', detector_source)
     30         self.detector_gpu = make_gpu_struct(detector_struct_size,
     31                                             [self.solid_id_to_channel_index_gpu,
     32                                              self.time_cdf_x_gpu,
     33                                              self.time_cdf_y_gpu,
     34                                              self.charge_cdf_x_gpu,
     35                                              self.charge_cdf_y_gpu,
     36                                              np.int32(self.nchannels),
     37                                              np.int32(len(detector.time_cdf[0])),
     38                                              np.int32(len(detector.charge_cdf[0])),
     39                                              np.float32(detector.charge_cdf[0][-1] / 2**16)])


Crucial connection between solids and channels, handled in **solid_id_to_channel_index[solid_id]**.
This distinquishes sensitive solids (PMTs).::

    simon:chroma blyth$ find . -name '*.*' -exec grep -H solid_id_to_channel_index {} \;
    ./cuda/daq.cu:      int channel_index = detector->solid_id_to_channel_index[solid_id];
    ./cuda/daq.cu:      channel_index = detector->solid_id_to_channel_index[solid_id];
    ./cuda/detector.h:    int *solid_id_to_channel_index;
    ./detector.py:        self.solid_id_to_channel_index = np.zeros(0, dtype=np.int32)
    ./detector.py:        self.solid_id_to_channel_index.resize(solid_id+1)
    ./detector.py:        self.solid_id_to_channel_index[solid_id] = -1 # solid maps to no channel
    ./detector.py:        self.solid_id_to_channel_index[solid_id] = channel_index
    ./gpu/daq.py:        self.solid_id_to_channel_index_gpu = gpu_detector.solid_id_to_channel_index_gpu
    ./gpu/detector.py:        self.solid_id_to_channel_index_gpu = \
    ./gpu/detector.py:            ga.to_gpu(detector.solid_id_to_channel_index.astype(np.int32))
    ./gpu/detector.py:                                            [self.solid_id_to_channel_index_gpu,




chroma/gpu/daq.py
-------------------

::

     60     def acquire(self, gpuphotons, rng_states, nthreads_per_block=64, max_blocks=1024, start_photon=None, nphotons=None, weight=1.0):
     61         if start_photon is None:
     62             start_photon = 0
     63         if nphotons is None:
     64             nphotons = len(gpuphotons.pos) - start_photon
     65 
     66         if self.ndaq == 1:
     67             for first_photon, photons_this_round, blocks in \
     68                     chunk_iterator(nphotons, nthreads_per_block, max_blocks):
     69                 self.gpu_funcs.run_daq(rng_states, np.uint32(0x1 << 2),
     70                                        np.int32(start_photon+first_photon), np.int32(photons_this_round), gpuphotons.t,
     71                                        gpuphotons.flags, gpuphotons.last_hit_triangles, gpuphotons.weights,
     72                                        self.solid_id_map_gpu,
     73                                        self.detector_gpu,
     74                                        self.earliest_time_int_gpu,
     75                                        self.channel_q_int_gpu, self.channel_history_gpu,
     76                                        np.float32(weight),
     77                                        block=(nthreads_per_block,1,1), grid=(blocks,1))




chroma/cuda/photon.h 
---------------------

::

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

::

    In [16]: np.uint32(0x1 << 2)
    Out[16]: 4




chroma/cuda/daq.cu
--------------------

* how do the *sample_cdf* compare with those from Geant4 ?

Sequence::

   photon_id > triangle_id > solid_id > channel_index 


::

     35 __global__ void
     36 run_daq(curandState *s, unsigned int detection_state,
     37     int first_photon, int nphotons, float *photon_times,
     38     unsigned int *photon_histories, int *last_hit_triangles,
     39     float *weights,
     40     int *solid_map,
     41     Detector *detector,
     42     unsigned int *earliest_time_int,
     43     unsigned int *channel_q_int, unsigned int *channel_histories,
     44     float global_weight)
     45 {
     46 
     47     int id = threadIdx.x + blockDim.x * blockIdx.x;
     48 
     49     if (id < nphotons) {
     50     curandState rng = s[id];
     51     int photon_id = id + first_photon;
     52     int triangle_id = last_hit_triangles[photon_id];
     53 
     54     if (triangle_id > -1) {
     55         int solid_id = solid_map[triangle_id];
     56         unsigned int history = photon_histories[photon_id];
     57         int channel_index = detector->solid_id_to_channel_index[solid_id];
     58 
     59         if (channel_index >= 0 && (history & detection_state)) {                  // SURFACE_DETECT flagged in history   
     60 
     61         float weight = weights[photon_id] * global_weight;
     62         if (curand_uniform(&rng) < weight) {
     63             float time = photon_times[photon_id] +
     64             sample_cdf(&rng, detector->time_cdf_len,
     65                    detector->time_cdf_x, detector->time_cdf_y);
     66             unsigned int time_int = float_to_sortable_int(time);
     67 
     68             float charge = sample_cdf(&rng, detector->charge_cdf_len,
     69                       detector->charge_cdf_x,
     70                       detector->charge_cdf_y);
     71             unsigned int charge_int = roundf(charge / detector->charge_unit);
     72 
     73             atomicMin(earliest_time_int + channel_index, time_int);
     74             atomicAdd(channel_q_int + channel_index, charge_int);
     75             atomicOr(channel_histories + channel_index, history);
     76         } // if weighted photon contributes
     77 
     78         } // if photon detected by a channel
     79 
     80     } // if photon terminated on surface
     81 
     82     s[id] = rng;
     83    
     84     }
     85    
     86 }



