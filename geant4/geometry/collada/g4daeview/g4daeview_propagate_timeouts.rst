Propagate Timeouts
====================

Objectives
----------

* robustify propagation

* bookkeeping

  * local worker lockfile to prevent two propagators running simultaneously on same node
  * python metadata dict containing propagator name and parameters : could be the lockfile 
  * how to do this remotely? multipart zmq message with 2nd frame containing metadata ?

    * probably json https://github.com/vivkin/gason as easy to 
      translate from python dict and parsers available in every language : speed
      is not a great concern as not much metadata 

  * sqlite based recording of 

    * propagation parameters and times
    * photon/hits/hitlist digests 

  * exercise bookkeeping by checking dependency on params like max_steps and cohort size 

* hit checking 

  * flag analysis : NAN_ABORT trace
  * many slot 22 : suggests truncation : check just storage truncation, not step truncation
  * correlate the photons with the hits 
  * hit time distributions
  * hit wavelength distributions


mock001 : normal incidence NAN_ABORT
--------------------------------------

Perfect normal incidence resulted in an NAN_ABORT. Observed
with a vertical photon incident on the horizontal AD table, 
on which the ADs sit. Worked around by special casing normal incidence.


mock001 : axis aligned photons issue
---------------------------------------

Need to login GPU machine in ">console" only  mode in order to use the debugger.
Most convenient to do that over ssh, for multiple connections and copy and pasting.

::

    (chroma_env)delta:g4daeview blyth$ g4daechroma.sh --cuda-gdb 


Suspect a handful of stuck threads::

    (cuda-gdb) info threads
      4 Thread 0x261b of process 44071  0x00007fff9292664a in -[PAStackshot initWithGlobalTrace].tracebuf () from /usr/lib/system/libsystem_kernel.dylib
      3 Thread 0x240b of process 44071  0x00007fff9292664a in -[PAStackshot initWithGlobalTrace].tracebuf () from /usr/lib/system/libsystem_kernel.dylib
      2 Thread 0x23f7 of process 44071  0x00007fff92921a1a in -[PAStackshot initWithGlobalTrace].tracebuf () from /usr/lib/system/libsystem_kernel.dylib
    * 1 Thread 0x2203 of process 44071  0x00007fff89fc1b2d in pthread_threadid_np () from /usr/lib/system/libsystem_pthread.dylib
    (cuda-gdb) kill
    Kill the program being debugged? (y or n) y
    [Termination of CUDA Kernel 9 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 8 (reduce_kernel_stage2<<<(1,1,1),(512,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 7 (reduce_kernel_stage1<<<(3,1,1),(512,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 5 (init_rng<<<(1025,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 4 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 3 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 2 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 1 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 0 (write_size<<<(1,1,1),(1,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 10 (propagate_hit<<<(2,1,1),(64,1,1)>>>) on Device 0]
    [Termination of CUDA Kernel 6 (propagate_hit<<<(66,1,1),(64,1,1)>>>) on Device 0]
    (cuda-gdb) 


Suspicion turned out to be correct, axis aligned (actually z-aligned vertical photons) 
were not reaching intersection.  They were just continuously testing against boxes, this was due
to some infinities meaning that the x,y info was effectively not used.  Workaround 
was to special case axis aligned photons.



Code Improvements
-------------------

* provide high level API for ease of use

  * single header "G4DAEChroma.hh"
  * all normal operations through "chroma" instance ?
  * reposition photons and hits outside transport 


Dependencies
-------------

* eliminate cnpy to avoid dependency duplication 

  * need to migrate G4DAETransformCache to numpy.hpp 

* cordon ROOT + ZMQRoot + ChromaPhotonList usage behind a definition






DONE
------

* reproducibility/seeding : reproducibility established by reset_rng_states that does reset for every propagation 
* compare vbo to non-vbo propagation : match made
* move propagation relevant constants into daedirectconfig.py so both propagations use the same 
* getting sensor ids back to caller
* is the transform to local yielding expected coordinates ?  yep small



::

    # vbo propagation 
    # photon_id / slot / history / pmtid 

    In [23]: h[:,3,:4].view(np.int32)
    Out[23]: 
    array([[      62,       22,       68, 16844311],
           [      74,       22,       84, 16844311],
           [      82,       22,       68, 16844810],
           [      93,       22,      580, 16843276],
           [     164,       22,       68, 16843021],
           [     170,       11,       68, 16844291],
           [     194,        5,      516, 16844033],
           [     248,        6,      516, 16843800],


Issues
--------

timeouts
~~~~~~~~~

Some photon lists like mock001 succeed to propagate, 
some like mock007 causing timeouts.

pycuda errors that manifest as timeouts can be due to the GPU equivalent 
of a segfault which kills the context, and subsequently causes the 
timeout as the host has no context to talk to on device.

Are certain photon parameters causing "segfaults" on GPU ?

::

     File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daephotons.py", line 222, in propagate
        self.propagator.interop_propagate( vbo, max_steps=max_steps, max_slots=max_slots )
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daephotonspropagator.py", line 192, in interop_propagate
        self.propagate( vbo_dev_ptr, max_steps=max_steps, max_slots=max_slots )   
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daephotonspropagator.py", line 160, in propagate
        t = get_time()
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 453, in get_call_time
        end.synchronize()
    pycuda._driver.LaunchError: cuEventSynchronize failed: launch timeout
    PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    cuEventDestroy failed: launch timeout
    PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    cuEventDestroy failed: launch timeout
    PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    cuGLUnmapBufferObject failed: launch timeout
    (chroma_env)delta:g4daeview blyth$ 
    (chroma_env)delta:g4daeview blyth$ 
    (chroma_env)delta:g4daeview blyth$ g4daeview.sh --load mock007



mock photons
-------------

Using the transform cache, samples of photons were prepared with 
directions oriented with respect to the PMTs. Eg bullseye photons.

To visualize initial photons load with `-P/--nopropagate` 

::

   g4daeview.sh --load mock002 --nopropagate --geometry-regexp PmtHemiCathode


::

   //transport->GetPhotons()->Save("mock002");  // ldir +y
   //transport->GetPhotons()->Save("mock003");  // ldir +x
   //transport->GetPhotons()->Save("mock004");  // ldir +z
   //transport->GetPhotons()->Save("mock005");  // lpos (0,0,100) ldir (0,0,-1)  try to shoot directly at PMT 
   //transport->GetPhotons()->Save("mock006");  // lpos (0,0,500) ldir (0,0,-1)  try to shoot directly at PMT 
   //transport->GetPhotons()->Save("mock007");  // lpos (0,0,1500) ldir (0,0,-1)  try to shoot directly at PMT 



vbo vs non-vbo hit count difference
--------------------------------------

::

    In [36]: h.shape
    Out[36]: (146, 4, 4)

    In [37]: h = ph("h1")

    In [38]: h.shape
    Out[38]: (33, 4, 4)


mocknuwa propagation testing over network
--------------------------------------------

While running::

    # non-vbo  propagation using propagate_hit.cu gpu/photon_hit.py GPUPhotonsHit 
    g4daechroma.sh        

    # vbo propagation with the GUI 
    g4daeview.sh --live   
    g4daeview.sh --zmqendpoint=tcp://localhost:5002

    # the broker
    czmq-;czmq-broker-local    

Provoke a propagation with::

    mocknuwa.sh 1

file based propagation testing
--------------------------------

debug propagation with::

    daedirectpropagation.sh mock001

visualize initial positions by holding propagation
----------------------------------------------------

::


   g4daeview.sh --load mock002 --nopropagate --geometry-regexp PmtHemiCathode
   udp.py --load mock002 
   udp.py --load mock003 
   udp.py --propagate




vbo propagation
-----------------

Kernel invoked from interop_propagate  `daephotons.py`::

    182     def propagate(self, max_steps=100):
    ...
    216         vbo = self.renderer.pbuffer   
    217         
    218         self.propagator.update_constants()
    219         
    220         if not self.config.args.propagate:
    221             log.warn("propagation is inhibited by config: -P/--nopropagate ")
    222         else:
    223             log.warn("propagation proceeding")
    224             self.propagator.interop_propagate( vbo, max_steps=max_steps, max_slots=max_slots )
    225         pass
    226     
    227         propagated = vbo.read()


kernel call `daephotonspropagator.py`::

    .92     def propagate(self,
     93                   vbo_dev_ptr,
     94                   max_steps=100,
     95                   max_slots=30,
     96                   use_weights=False,
     97                   scatter_first=0):
     98         """
    ...
    145                     grid=(blocks, 1)
    146                     args = ( np.int32(first_photon),
    147                              np.int32(photons_this_round),
    148                              self.input_queue_gpu[1:].gpudata,
    149                              self.output_queue_gpu.gpudata,
    150                              self.ctx.rng_states,
    151                              vbo_dev_ptr,
    152                              np.int32(nsteps),
    153                              np.int32(max_slots),
    154                              np.int32(use_weights),
    155                              np.int32(scatter_first),
    156                              self.ctx.gpu_geometry.gpudata)
    157 
    158                     get_time = self.kernel.prepared_timed_call( grid, block, *args )


`cuda/propagate_vbo.cu`::

    488 __global__ void
    489 propagate_vbo( int first_photon,
    490                int nthreads,
    491                unsigned int *input_queue,
    492                unsigned int *output_queue,
    493                curandState *rng_states,
    494                float4 *vbo,
    495                int max_steps,
    496                int max_slots,
    497                int use_weights,
    498                int scatter_first,
    499                Geometry *g)
    500 {


Hmm, can i access the maps from the Geometry struct GPU side ? Nope not there::

     54 struct Geometry
     55 {
     56     float3 *vertices;
     57     uint3 *triangles;
     58     unsigned int *material_codes;
     59     unsigned int *colors;
     60     uint4 *primary_nodes;
     61     uint4 *extra_nodes;
     62     Material **materials;
     63     Surface **surfaces;
     64     float3 world_origin;
     65     float world_scale;
     66     int nprimary_nodes;
     67 };

      4 struct Detector
      5 {
      6     // Order in decreasing size to avoid alignment problems
      7     int *solid_id_to_channel_index;




non-vbo propagation
---------------------

Must use GPUDetector (not GPUGeometry) to have the mapping arrays.

`gpu/detector.py`::

     16 class GPUDetector(GPUGeometry):
     17     def __init__(self, detector, wavelengths=None, print_usage=False):
     18         GPUGeometry.__init__(self, detector, wavelengths=wavelengths, print_usage=False)
     19 
     20         self.solid_id_to_channel_index_gpu = \
     21             ga.to_gpu(detector.solid_id_to_channel_index.astype(np.int32))
     22         self.solid_id_to_channel_id_gpu = \
     23             ga.to_gpu(detector.solid_id_to_channel_id.astype(np.int32))
     24 


`gpu/photon_hit.py`::

    176         solid_id_map_gpu = gpu_geometry.solid_id_map
    177         solid_id_to_channel_id_gpu = gpu_geometry.solid_id_to_channel_id_gpu
    178 
    ...
    197                     grid = (blocks, 1)
    198                     args = (
    199                         np.int32(first_photon),
    200                         np.int32(photons_this_round),
    201                         self.input_queue_gpu[1:].gpudata,
    202                         self.output_queue_gpu.gpudata,
    203                         rng_states,
    204                         self.pos.gpudata,
    205                         self.dir.gpudata,
    206                         self.wavelengths.gpudata,
    207                         self.pol.gpudata,
    208                         self.t.gpudata,
    209                         self.flags.gpudata,
    210                         self.last_hit_triangles.gpudata,
    211                         self.weights.gpudata,
    212                         np.int32(nsteps),
    213                         np.int32(use_weights),
    214                         np.int32(scatter_first),
    215                         gpu_geometry.gpudata,
    216                         solid_id_map_gpu.gpudata,
    217                         solid_id_to_channel_id_gpu.gpudata,
    218                             )
    219                     get_time = self.propagate_hit_kernel.prepared_timed_call( grid, block, *args )
    220                     t = get_time()



`cuda/propagate_hit.cu`::

    118 // iiPPPPPPPPPPPiiiP
    119 
    120 __global__ void
    121 propagate_hit(
    122       int first_photon,
    123       int nthreads,
    124       unsigned int *input_queue,
    125       unsigned int *output_queue,
    126       curandState *rng_states,
    127       float3 *positions,
    128       float3 *directions,
    129       float *wavelengths,
    130       float3 *polarizations,
    131       float *times,
    132       unsigned int *histories,
    133       int *last_hit_triangles,
    134       float *weights,
    135       int max_steps,
    136       int use_weights,
    137       int scatter_first,
    138       Geometry *g,
    139       int* solid_map,
    140       int* solid_id_to_channel_id )
    141 {
    ...
    233     if ((p.history & SURFACE_DETECT) != 0) {
    234 
    235         //
    236         // kludgy mis-use of lht for outputting 
    237         // various things like 
    238         //       solid_id:    index like, zero based
    239         //       channel_id:  the pmtid, encoding site/ad/ring/...
    240         //
    241         int triangle_id = last_hit_triangles[photon_id];
    242         if (triangle_id > -1) {
    243             int solid_id = solid_map[triangle_id];
    244             int channel_id = solid_id_to_channel_id[solid_id];
    245             last_hit_triangles[photon_id] = channel_id ;
    246         } else {
    247             last_hit_triangles[photon_id] = -2 ;
    248         }



threading sensor ids back to caller (vbo)
----------------------------------------------

::

    In [7]: h = ph("h1")

    In [8]: a = h[:,3,0].view(np.int32)

    In [9]: b = h[:,3,1].view(np.int32)

    In [10]: c = h[:,3,2].view(np.int32)
        
    In [11]: a[a != 0]
    Out[11]: 
    array([ 750,  276,  816,  342,  486,  702, 1044,  936,  696,  696, 1050,
           1194,  372,  390,  756, 1086,  762, 1134,  786,  726, 1026,  408,
            912,   48,  102,   78,  756,  942,  954, 1164,  108,  876, 1092,
            702,  504,  414,  702,  498,  522,  546,  768,  324, 1086, 1008,

            ...

    In [13]: np.set_printoptions(formatter={'int':hex})

    In [14]: b[b != 0]
    Out[14]: 
    array([0x1010516, 0x101020f, 0x1010609, 0x1010302, 0x1010402, 0x101050e,
           0x1010717, 0x1010705, 0x101050d, 0x101050d, 0x1010718, 0x1010818,
           0x1010307, 0x101030a, 0x1010517, 0x1010806, 0x1010518, 0x101080e,
           0x1010604, 0x1010512, 0x1010714, 0x101030d, 0x1010701, 0x1010101,
           ...
 

    In [16]: np.set_printoptions(formatter={'int':None})

    In [17]: c[c != 0]
    Out[17]: array([888, 888, 888, ..., 888, 888, 888], dtype=int32)



threading sensor ids back to caller (non-vbo)
----------------------------------------------

::

    In [12]: a = ph("1")

    In [13]: h = ph("h1")

    In [14]: a.shape
    Out[14]: (4165, 4, 4)

    In [15]: h.shape
    Out[15]: (52, 4, 4)

    In [16]: np.set_printoptions(formatter={'int':hex})

    In [17]: h[:,3,3]
    Out[17]: 
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32)

    In [18]: h[:,3,3].view(np.int32)
    Out[18]: 
    array([0x1010516, 0x1010302, 0x1010402, 0x1010717, 0x1010718, 0x1010517,
           0x1010518, 0x1010701, 0x1010106, 0x1010706, 0x1010708, 0x101010b,
           0x101050e, 0x101040c, 0x1010601, 0x1010201, 0x101020d, 0x101020d,
           0x1010502, 0x1010209, 0x101070d, 0x1010602, 0x1010715, 0x1010108,
           0x1010407, 0x1010418, 0x101040b, 0x101060c, 0x1010709, 0x1010409,
           0x101050d, 0x101050d, 0x1010613, 0x1010707, 0x1010516, 0x101020d,
           0x1010201, 0x1010308, 0x101040f, 0x101010e, 0x1010109, 0x1010417,
           0x101050c, 0x1010309, 0x1010213, 0x101050c, 0x1010402, 0x101040e,
           0x1010716, 0x1010315, 0x101010f, 0x1010416], dtype=int32)


Hmm for comparison need photon index in the hits array





