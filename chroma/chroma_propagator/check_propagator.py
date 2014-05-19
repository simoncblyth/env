#!/usr/bin/env python
"""

Selecting photons ending with inf wavelength, they all include
BULK_REEMIT in their histories.

inf wavelengths 1397 : array([  29,   44,   50, ..., 4149, 4153, 4162]) 
history for inf wavelength photons
[0x202] 642 ( 0.46): BULK_REEMIT,BULK_ABSORB 
[0x242] 639 ( 0.46): REFLECT_SPECULAR,BULK_REEMIT,BULK_ABSORB 
[0x252] 58 ( 0.04): REFLECT_SPECULAR,BULK_REEMIT,BULK_ABSORB,RAYLEIGH_SCATTER 
[0x262] 28 ( 0.02): REFLECT_SPECULAR,BULK_REEMIT,REFLECT_DIFFUSE,BULK_ABSORB 
[0x212] 10 ( 0.01): BULK_REEMIT,BULK_ABSORB,RAYLEIGH_SCATTER 
[0x222] 10 ( 0.01): BULK_REEMIT,REFLECT_DIFFUSE,BULK_ABSORB 
[0x272] 6 ( 0.00): REFLECT_SPECULAR,BULK_REEMIT,REFLECT_DIFFUSE,BULK_ABSORB,RAYLEIGH_SCATTER 
[0x232] 2 ( 0.00): BULK_REEMIT,REFLECT_DIFFUSE,BULK_ABSORB,RAYLEIGH_SCATTER 
[0x240] 1 ( 0.00): REFLECT_SPECULAR,BULK_REEMIT 
[0x250] 1 ( 0.00): REFLECT_SPECULAR,BULK_REEMIT,RAYLEIGH_SCATTER 
(chroma_env)delta:chroma_propagator blyth$ 


The list of inf wavelength photons  precisely matches those with BULK_REEMIT
flag set.

Probably materials are lacking::

        self.set('reemission_cdf', 0)



"""
import logging
log = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    import ROOT
    ROOT.gSystem.Load("$LOCAL_BASE/env/chroma/ChromaPhotonList/lib/libChromaPhotonList")
    from env.chroma.ChromaPhotonList.cpl import load_cpl

    from env.geant4.geometry.collada.g4daeview.daeconfig import DAEConfig
    config = DAEConfig(__doc__)
    config.init_parse()
    config.report()
    config.args.with_chroma = True 

    from env.geant4.geometry.collada.g4daeview.daegeometry import DAEGeometry
    geometry = DAEGeometry(config.args.geometry, config)
    geometry.flatten()

    from env.geant4.geometry.collada.g4daeview.daechromacontext import DAEChromaContext
    dcc = DAEChromaContext(config, geometry.make_chroma_geometry() )

    path = config.resolve_event_path("1")
    cpl = load_cpl(path,config.args.key)

    from chroma.event import Photons
    photons = Photons.from_cpl(cpl, extend=True)   
    photons.dump()

    propagator = dcc.propagator
    photons2 = propagator.propagate( photons, max_steps=100)
    photons2.dump()








