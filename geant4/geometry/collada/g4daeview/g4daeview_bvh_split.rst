BVH Splits
===========

Suspect the splits to be the culprit for the factor 4 slowdown 
of propagation that sometimes occurs.

TODO: find way to record the "split" event in metadata::

    2014-11-20 19:47:44,927 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:66  using seed 0 
    2014-11-20 19:47:44,927 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:73  ***** post G4DAEChroma ctor
    2014-11-20 19:47:44,927 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:50  start polling responder: DAEDirectResponder connect tcp://127.0.0.1:5002  
    2014-11-20 19:47:44,927 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:54  polling 0 
    2014-11-20 19:47:52,657 INFO    env.geant4.geometry.collada.g4daeview.daedirectresponder:47  DAEDirectResponder request (4165, 4, 4) 
    2014-11-20 19:47:52,657 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:43  handler got obj (cpl or npl)
    2014-11-20 19:47:52,657 INFO    env.geant4.geometry.collada.g4daeview.daedirectpropagator:54  ctrl {u'reset_rng_states': 1, u'nthreads_per_block': 64, u'seed': 0, u'max_blocks': 1024, u'max_steps': 30, u'COLUMNS': u'max_blocks:i,max_steps:i,nthreads_per_block:i,reset_rng_states:i,seed:i'} 
    2014-11-20 19:47:52,657 WARNING env.geant4.geometry.collada.g4daeview.daedirectpropagator:62  reset_rng_states
    2014-11-20 19:47:52,657 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:122 _set_rng_states
    2014-11-20 19:47:53,005 INFO    chroma.gpu.geometry :171 Splitting BVH between GPU and CPU memory at node 100
    2014-11-20 19:47:53,030 INFO    chroma.gpu.geometry :201 device usage:
    ----------
    nodes           100.0    1.6K
    total                    1.6K
    ----------
    device total             2.1G
    device used              1.9G
    device free            292.1M

    2014-11-20 19:47:53,033 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:117 _get_rng_states
    2014-11-20 19:47:53,033 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:72  setup_rng_states using seed 0 




