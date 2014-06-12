#!/usr/bin/env python
"""
Treating a DAEPhotons like a DAEMesh 
"""
from daephotonsdata import DAEPhotonsData
from daegeometry import DAEMesh 

if __name__ == '__main__':
    import os
    import ROOT
    ROOT.gSystem.Load("$LOCAL_BASE/env/chroma/ChromaPhotonList/lib/libChromaPhotonList")
    from env.chroma.ChromaPhotonList.cpl import load_cpl, save_cpl, random_cpl
    path = os.environ['DAE_PATH_TEMPLATE'] % {'arg':"1"} 

    cpl = load_cpl(path,'CPL')
    photons = DAEPhotons.from_cpl(cpl)
    mesh = DAEMesh(photons.vertices)
    print mesh


    








