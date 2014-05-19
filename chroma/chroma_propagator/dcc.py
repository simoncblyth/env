#!/usr/bin/env python
"""
"""
import os, logging
log = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    import ROOT
    ROOT.gSystem.Load("$LOCAL_BASE/env/chroma/ChromaPhotonList/lib/libChromaPhotonList")

    from env.chroma.ChromaPhotonList.cpl import load_cpl, save_cpl, random_cpl
    from env.geant4.geometry.collada.g4daeview.daeconfig import DAEConfig
    from env.geant4.geometry.collada.g4daeview.daegeometry import DAEGeometry
    from env.geant4.geometry.collada.g4daeview.daechromacontext import DAEChromaContext

    config = DAEConfig(__doc__)
    config.init_parse()
    config.report()
    config.args.with_chroma = True 

    geometry = DAEGeometry(config.args.geometry, config)
    geometry.flatten()

    chroma_geometry = geometry.make_chroma_geometry() 
    dcc = DAEChromaContext(config, chroma_geometry )







