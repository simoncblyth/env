
VBO Propagation Call Sequence needs simplification now that using NPL
========================================================================

::

    File "/usr/local/env/chroma_env/lib/python2.7/site-packages/glumpy/window/event.py", line 349, in dispatch_event
        if handler(*args):
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daeinteractivityhandler.py", line 212, in on_external_npl
        self.scene.external_npl(npl)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daescene.py", line 119, in external_npl
        self.event.external_npl( npl )
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daeevent.py", line 199, in external_npl
        self.external_npl_base( npl )
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daeeventbase.py", line 25, in external_npl_base
        self.setup_npl(npl) 
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daeeventbase.py", line 66, in setup_npl
        self.setup_photons( photons ) 
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daeevent.py", line 213, in setup_photons
        self.setup_photons_base( photons )
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daeeventbase.py", line 74, in setup_photons_base
        self.dphotons.photons = photons   ## this setter triggers propagation  
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daephotons.py", line 325, in _set_photons
        lastpropagated = self.propagate()
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daephotons.py", line 227, in propagate
        self.propagator.interop_propagate( vbo, max_slots=max_slots ) 
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daephotonspropagator.py", line 205, in interop_propagate
        self.propagate( vbo_dev_ptr, max_slots=max_slots)   
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daephotonspropagator.py", line 160, in propagate
        self.ctx.rng_states, 
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daechromacontext.py", line 264, in _get_rng_states
        self._rng_states = self.setup_rng_states()
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daechromacontext.py", line 225, in setup_rng_states
        seed = self.gpu_seed 
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daechromacontext.py", line 245, in _get_gpu_seed
        assert 0, "use setter first"
    AssertionError: use setter first

