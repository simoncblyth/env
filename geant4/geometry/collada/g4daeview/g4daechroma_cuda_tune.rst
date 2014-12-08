G4DAEChroma CUDA tune
========================


Small Problem Size
-------------------

For small problem size there is clear pattern of improving performance 
as increase `threads_per_block` from too small 32 out to 256 
(about half the work size) where a minimum is reached.
Beyond the minimum there is only a very slight degradation
as increase further.

Does that mean having two active blocks is the most efficient config ?  


::

    In [2]: scatter("select threads_per_block,tottime from log, ctrl on log.ctrl_id = ctrl.id where log.id>307;")


    sqlite> select batch.nwork, threads_per_block,tottime from log, ctrl, batch  on log.ctrl_id = ctrl.id and log.batch_id = batch.id where log.id>307;
    nwork       threads_per_block  tottime   
    ----------  -----------------  ----------
    445         32                 0.105339  
    445         64                 0.069023  
    445         96                 0.066592  
    445         128                0.059896  
    445         160                0.052635  
    445         192                0.049819  
    445         224                0.046049  
    445         256                0.045372  
    445         288                0.045148  
    445         320                0.045277  
    445         352                0.045376  
    445         384                0.045921  
    445         416                0.045918  
    445         448                0.045956  
    445         480                0.045953  
    445         512                0.045972  
    sqlite> 


Large problem size
---------------------

Seems pretty much flat across threads_per_block from 32 to 512.
Attempts to push beyond 512 cause crashes.


::

    sqlite> select batch.nwork, threads_per_block,tottime from log, ctrl, batch  on log.ctrl_id = ctrl.id and log.batch_id = batch.id where batch.nwork > 3200  ;
    nwork       threads_per_block  tottime   
    ----------  -----------------  ----------
    4585        64                 0.359394  
    4585        128                0.351535  
    4585        192                0.369758  
    4585        256                0.371362  
    4585        320                0.376655  
    4585        384                0.377518  
    4585        448                0.378556  
    4585        512                0.384698  

    3201        64                 0.227585  
    3201        128                0.220926  
    3201        192                0.224383  
    3201        256                0.228088  
    3201        320                0.221011  
    3201        384                0.279568  
    3201        448                0.222894  
    3201        512                0.226172  
    sqlite> 





Variation of time with workload
---------------------------------

Time per 1000 work items is slightly increasing as increase workload, 
and would correspond to 80s for 1M : that cannot be correct. 
Perhaps as are in straggler mode, with small workloads so far.

::

    sqlite> select batch.nwork, threads_per_block,tottime, tottime/batch.nwork*1000 from log, ctrl, batch  on log.ctrl_id = ctrl.id and log.batch_id = batch.id where batch.nwork > 100 and threads_per_block = 256 order by batch.nwork  ;
    nwork       threads_per_block  tottime     tottime/batch.nwork*1000
    ----------  -----------------  ----------  ------------------------
    233         256                0.064202    0.275545064377682       
    445         256                0.045276    0.101743820224719       
    445         256                0.045372    0.101959550561798       
    1869        256                0.143624    0.0768453718566078      
    1888        256                0.141632    0.0750169491525424      
    2025        256                0.124308    0.0613866666666667      
    2053        256                0.132408    0.0644948855333658      
    2053        256                0.132172    0.0643799318071115      
    2463        256                0.173663    0.0705087291920422      
    2553        256                0.259664    0.101709361535448       
    2779        256                0.208306    0.0749571788413098      
    2979        256                0.247615    0.0831201745552199      
    3095        256                0.268925    0.0868901453957997      
    3095        256                0.269127    0.0869554119547657      
    3159        256                0.241424    0.0764241848686293      
    3201        256                0.228088    0.0712552327397688      
    4585        256                0.371362    0.0809949836423119      
    sqlite> 



Larger than 512 ?
-------------------


768 and 1024 are giving error in `upload_queues`::

      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daedirectpropagator.py", line 61, in propagate
        self.chroma.parameters)
      File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/photon_hit.py", line 234, in propagate_hit
        self.upload_queues( nwork )
      File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/photon_hit.py", line 175, in upload_queues
        self.input_queue_gpu = ga.to_gpu(input_queue)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/gpuarray.py", line 865, in to_gpu
        result = GPUArray(ary.shape, ary.dtype, allocator, strides=ary.strides)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/gpuarray.py", line 187, in __init__
        self.gpudata = self.allocator(self.size * self.dtype.itemsize)
    pycuda._driver.LaunchError: cuMemAlloc failed: launch timeout
    _finish_up : cuda cleanup 
    PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    cuModuleUnload failed: launch timeout


Straggler Mode
----------------

For higher workloads chroma does single stepping with multiple kernel 
launches. The small workloads have been checking with so far fit into 
small remainder stragglers and are all done in a single launch.
 

::


    239         small_remainder = nthreads_per_block * 16 * 8
    240         block=(nthreads_per_block,1,1)
    241 
    242         results = {}
    243         results['name'] = "propagate_hit"
    244         results['nphotons'] = nphotons
    245         results['nwork'] = nwork
    246         results['nsmall'] = small_remainder
    247         results['COLUMNS'] = "name:s,nphotons:i,nwork:i,nsmall:i"
    ...
    254 
    255         while step < max_steps:
    256             npass += 1
    257             if nwork < small_remainder or use_weights:
    258                 nsteps = max_steps - step
    259                 log.debug("increase nsteps for stragglers: small_remainder %s nwork %s nsteps %s max_steps %s " % (small_remainder, nwork, nsteps, max_steps))
    260             else:
    261                 nsteps = 1 # Just finish the rest of the steps if the # of photons is low
    262             pass
    263             log.info("nwork %s step %s max_steps %s nsteps %s " % (nwork, step,max_steps, nsteps) )
    264 
    265             abort = False
    266             for first_photon, photons_this_round, blocks in chunk_iterator(nwork, nthreads_per_block, max_blocks):
    267                 if abort:
    268                     nabort += 1
    269                 else:
    270                     grid = (blocks, 1)
    271                     args = (
    272                         np.int32(first_photon),
    273                         np.int32(photons_this_round),
    274                         self.input_queue_gpu[1:].gpudata,




Propagate Queues
------------------






#. propagation of `first_photon + blockIdx.x*blockDim.x + threadIdx.x`  ie the launch set starting at `first_photon`


::

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
    142     __shared__ Geometry sg;
    143 
    144     if (threadIdx.x == 0)
    145     sg = *g;
    146 
    147     __syncthreads();
    148 
    149     int id = blockIdx.x*blockDim.x + threadIdx.x;
    150 
    151     if (id >= nthreads)
    152     return;
    153 
    154     g = &sg;
    155 
    156     curandState rng = rng_states[id];
    157 
    158     int photon_id = input_queue[first_photon + id];
    159 
    160     Photon p;
    161     p.position = positions[photon_id];

    ...

    230     // Not done, put photon in output queue
    231     if ((p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT)) == 0) 
    232     {  
    233         int out_idx = atomicAdd(output_queue, 1);  // atomic add 1 to slot zero value, returns non-incremented original value, pulling a queue ticket 
    234         output_queue[out_idx] = photon_id;
    235     }



