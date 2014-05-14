#!/usr/bin/env python
"""
Treating a DAEChromaPhotonListBase like a geometrical mesh 
"""
from daechromaphotonlistbase import DAEChromaPhotonListBase
from daegeometry import DAEMesh 

if __name__ == '__main__':
    import os
    import ROOT
    ROOT.gSystem.Load("$LOCAL_BASE/env/chroma/ChromaPhotonList/lib/libChromaPhotonList")
    from env.chroma.ChromaPhotonList.cpl import load_cpl, save_cpl, random_cpl
    path = os.environ['DAE_PATH_TEMPLATE'] % {'arg':"1"} 
    cpl = load_cpl(path,'CPL')

    dcpl = DAEChromaPhotonListBase(cpl)
    mdcpl = DAEMesh(dcpl.pos)
    print mdcpl


    








