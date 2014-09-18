Chroma Sim Usage
===================

Simplify
-----------

#. Most of `chroma/sim.py` not relevant to simply propagating, 
   its ripe for drastic simplification perhaps into a new `chroma/propagate.py`

#. will need my flexible CUDA launch stuff too



chroma-sim
-------------

`chroma/bin/chroma-sim`::

    082     detector = chroma.loader.load_geometry_from_string(args[0])
     84     pos = np.array([float(s) for s in options.pos.split(',')], dtype=float)
     85     dir = np.array([float(s) for s in options.dir.split(',')], dtype=float)
     87     ev_gen = constant_particle_gun(particle_name=options.particle,
     88                                    pos=pos, dir=dir, ke=options.ke)
     90     sim = Simulation(detector, seed=options.seed, cuda_device=options.device,
     91                      geant4_processes=options.ngenerators)
     95     writer = root.RootWriter(options.output_filename)
     97     start = time.time()
     98     for i, ev in \
     99             enumerate(sim.simulate(itertools.islice(ev_gen,
    100                                                     options.nevents),
    101                                    keep_photons_beg=options.save_photons_beg,
    102                                    keep_photons_end=options.save_photons_end)):
    104         sys.stdout.flush()
    106         assert ev.nphotons > 0, 'Geant4 generated event with no photons!'
    108         writer.write_event(ev)
    109     print
    111     writer.close()


Simulation
------------

`chroma/chroma/sim.py`::

     19 class Simulation(object):
     20     def __init__(self, detector, seed=None, cuda_device=None,
     21                  geant4_processes=4, nthreads_per_block=64, max_blocks=1024):
     22         self.detector = detector
     ..
     43         if hasattr(detector, 'num_channels'):
     44             self.gpu_geometry = gpu.GPUDetector(detector)
     45             self.gpu_daq = gpu.GPUDaq(self.gpu_geometry)
     46             self.gpu_pdf = gpu.GPUPDF()
     47             self.gpu_pdf_kernel = gpu.GPUKernelPDF()
     48         else:
     49             self.gpu_geometry = gpu.GPUGeometry(detector)
     ..
     55     def simulate(self, iterable, keep_photons_beg=False,
     56                  keep_photons_end=False, run_daq=True, max_steps=100):
     ..
     73         for ev in iterable:
     74             gpu_photons = gpu.GPUPhotons(ev.photons_beg)
     75 
     76             gpu_photons.propagate(self.gpu_geometry, self.rng_states,
     77                                   nthreads_per_block=self.nthreads_per_block,
     78                                   max_blocks=self.max_blocks,
     79                                   max_steps=max_steps)
     80 
     81             ev.nphotons = len(ev.photons_beg.pos)
     82 
     83             if not keep_photons_beg:
     84                 ev.photons_beg = None
     85 
     86             if keep_photons_end:
     87                 ev.photons_end = gpu_photons.get()
     88 
     89             # Skip running DAQ if we don't have one
     90             if hasattr(self, 'gpu_daq') and run_daq:
     //  zero arrays 
     91                 self.gpu_daq.begin_acquire()
     92                 self.gpu_daq.acquire(gpu_photons, self.rng_states, nthreads_per_block=self.nthreads_per_block, max_blocks=self.max_blocks)
     93                 gpu_channels = self.gpu_daq.end_acquire()
     94                 ev.channels = gpu_channels.get()
     95 
     96             yield ev


GPUDaq
--------

#. time_int ?  digitization ?

`chroma/chroma/gpu/daq.py`::

    037 class GPUDaq(object):
     38     def __init__(self, gpu_detector, ndaq=1):
     39         self.earliest_time_gpu = ga.empty(gpu_detector.nchannels*ndaq, dtype=np.float32)
     40         self.earliest_time_int_gpu = ga.empty(gpu_detector.nchannels*ndaq, dtype=np.uint32)
     41         self.channel_history_gpu = ga.zeros_like(self.earliest_time_int_gpu)
     42         self.channel_q_int_gpu = ga.zeros_like(self.earliest_time_int_gpu)
     43         self.channel_q_gpu = ga.zeros(len(self.earliest_time_int_gpu), dtype=np.float32)
     44         self.detector_gpu = gpu_detector.detector_gpu
     45         self.solid_id_map_gpu = gpu_detector.solid_id_map
     46         self.solid_id_to_channel_index_gpu = gpu_detector.solid_id_to_channel_index_gpu
     47 
     48         self.module = get_cu_module('daq.cu', options=cuda_options,
     49                                     include_source_directory=True)
     50         self.gpu_funcs = GPUFuncs(self.module)
     51         self.ndaq = ndaq
     52         self.stride = gpu_detector.nchannels
     53 
     54     def begin_acquire(self, nthreads_per_block=64):
     55         self.gpu_funcs.reset_earliest_time_int(
                                   np.float32(1e9), 
                                   np.int32(len(self.earliest_time_int_gpu)), 
                                   self.earliest_time_int_gpu, 
                                         block=(nthreads_per_block,1,1), 
                                          grid=(len(self.earliest_time_int_gpu)//nthreads_per_block+1,1)
                                         )
     56         self.channel_q_int_gpu.fill(0)
     57         self.channel_q_gpu.fill(0)
     58         self.channel_history_gpu.fill(0)
     59 
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
     78         else:
     ...
     91         cuda.Context.get_current().synchronize()
     92
     93     def end_acquire(self, nthreads_per_block=64):
     94         self.gpu_funcs.convert_sortable_int_to_float(
                      np.int32(len(self.earliest_time_int_gpu)), 
                      self.earliest_time_int_gpu, 
                      self.earliest_time_gpu, 
                          block=(nthreads_per_block,1,1), 
                           grid=(len(self.earliest_time_int_gpu)//nthreads_per_block+1,1))
     95 
     96         self.gpu_funcs.convert_charge_int_to_float(
                             self.detector_gpu, 
                             self.channel_q_int_gpu, 
                             self.channel_q_gpu, 
                                  block=(nthreads_per_block,1,1), 
                                   grid=(len(self.channel_q_int_gpu)//nthreads_per_block+1,1))
     97 
     98         cuda.Context.get_current().synchronize()
     99 
    100         return GPUChannels(self.earliest_time_gpu, self.channel_q_gpu, self.channel_history_gpu, self.ndaq, self.stride)
    101 



    009 class GPUChannels(object):
     10     def __init__(self, t, q, flags, ndaq=1, stride=None):
     11         self.t = t
     12         self.q = q
     13         self.flags = flags
     14         self.ndaq = ndaq
     15         if stride is None:
     16             self.stride = len(t)
     17         else:
     18             self.stride = stride
     19 
     20     def iterate_copies(self):
     21         for i in xrange(self.ndaq):
     22             yield GPUChannels(self.t[i*self.stride:(i+1)*self.stride],
     23                               self.q[i*self.stride:(i+1)*self.stride],
     24                               self.flags[i*self.stride:(i+1)*self.stride])
     25 
     26     def get(self):
     27         t = self.t.get()
     28         q = self.q.get()
     29 
     30         # For now, assume all channels with small
     31         # enough hit time were hit.
     //
     //    NO q cut 
     //
     32         return event.Channels(t<1e8, t, q, self.flags.get())
     33 
     34     def __len__(self):
     35         return self.t.size



`chroma/chroma/event.py`::

    ///
    ///   t<1e8 is numpy.where list of indices of channels with such earliest times
    ///
    260 class Channels(object):
    261     def __init__(self, hit, t, q, flags=None):
    262         '''Create a list of n channels.  All channels in the detector must 
    263         be included, regardless of whether they were hit.
    264 
    265            hit: numpy.ndarray(dtype=bool, shape=n)
    266              Hit state of each channel.
    267 
    268            t: numpy.ndarray(dtype=numpy.float32, shape=n)
    269              Hit time of each channel. (ns)
    270 
    271            q: numpy.ndarray(dtype=numpy.float32, shape=n)
    272              Integrated charge from hit.  (units same as charge 
    273              distribution in detector definition)
    274         '''
    275         self.hit = hit
    276         self.t = t
    277         self.q = q
    278         self.flags = flags
    279 
    280     def hit_channels(self):
    281         '''Extract a list of hit channels.
    282         
    283         Returns: array of hit channel IDs, array of hit times, array of charges on hit channels
    284         '''
    285         return self.hit.nonzero(), self.t[self.hit], self.q[self.hit]


`chroma/chroma/cuda/daq.cu`::

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
     59         if (channel_index >= 0 && (history & detection_state)) {
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

 
* run_daq: time and charge sample_cdf applied to raw time and charge 

  * time is offset by the sample_cdf random draw (detector/electronics jitter ?  )
  * charge is purely sample_cdf random draw ?
  * MAYBE need to skip this randomization, done later by ElecSim in DetSim case ?


`chroma/chroma/cuda/detector.h`::

     04 struct Detector
      5 {
      6     // Order in decreasing size to avoid alignment problems
      7     int *solid_id_to_channel_index;
      8 
      9     float *time_cdf_x;
     10     float *time_cdf_y;
     11 
     12     float *charge_cdf_x;
     13     float *charge_cdf_y;
     14 
     15     int nchannels;
     16     int time_cdf_len;
     17     int charge_cdf_len;
     18     float charge_unit;
     19     // Convert charges to/from quantized integers with
     20     // q_int = (int) roundf(q / charge_unit )
     21     // q = q_int * charge_unit
     22 };


       


 
