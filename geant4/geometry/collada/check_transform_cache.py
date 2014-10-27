#!/usr/bin/env python
"""

::

    In [20]: np.where( idmap.a['id'] > 0 )
    Out[20]: (array([ 3199,  3200,  3201, ..., 11422, 11423, 11424]),)

    In [21]: de = np.where( idmap.a['id'] > 0 )[0]

    In [22]: de
    Out[22]: array([ 3199,  3200,  3201, ..., 11422, 11423, 11424])


Multiple volumes all land in same DE with ?::

    In [24]: idmap.a[de][0]
    Out[24]: (3199, 16843009, '1010101', '(8842.5,532069,599609)', '(3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17)', '/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt')

    In [25]: idmap.a[de][1]
    Out[25]: (3200, 16843009, '1010101', '(8842.5,532069,599609)', '(3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17)', '/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum')

    In [26]: idmap.a[de][2]
    Out[26]: (3201, 16843009, '1010101', '(8842.5,532069,599609)', '(3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17)', '/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode')

    In [27]: idmap.a[de][3]
    Out[27]: (3202, 16843009, '1010101', '(8842.5,532069,599540)', '(3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17)', '/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom')

    In [28]: idmap.a[de][4]
    Out[28]: (3203, 16843009, '1010101', '(8842.5,532069,599690)', '(3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17)', '/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode')

    In [29]: idmap.a[de][5]
    Out[29]: (3205, 16843010, '1010102', '(8842.5,668528,441547)', '(5.04009e-17,0.567844,-0.823136)(-3.47693e-17,0.823136,0.567844)(1,0,6.12303e-17)', '/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmt')


Change default key to volume index rather than sd index as more convenient for comparisons::

    (chroma_env)delta:G4DAEChromaTest blyth$ G4DAEChromaTest
    geokey DAE_NAME_DYB_GDML geopath /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml 
    G4DAETransformCache::Load [/usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml.cache/key.npy] 
    G4DAETransformCache::Load [/usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml.cache/data.npy] 
     key       3200 tr (8842.5,532069,599609) (9.80427e-17,0.761538,-0.64812)(7.29457e-17,0.64812,0.761538)(1,-1.21941e-16,7.99242e-18)
     idx          1 tr (8842.5,532069,599609) (9.80427e-17,0.761538,-0.64812)(7.29457e-17,0.64812,0.761538)(1,-1.21941e-16,7.99242e-18)
     key       3206 tr (8842.5,668528,441547) (8.61823e-17,0.567844,-0.823136)(8.36624e-17,0.823136,0.567844)(1,-1.17804e-16,2.34326e-17)
     idx          2 tr (8842.5,668528,441547) (8.61823e-17,0.567844,-0.823136)(8.36624e-17,0.823136,0.567844)(1,-1.17804e-16,2.34326e-17)
     key       3212 tr (8842.5,759428,253553) (7.19524e-17,0.335452,-0.942057)(9.09442e-17,0.942057,0.335452)(1,-1.09811e-16,3.72759e-17)
     idx          3 tr (8842.5,759428,253553) (7.19524e-17,0.335452,-0.942057)(9.09442e-17,0.942057,0.335452)(1,-1.09811e-16,3.72759e-17)
     key       3218 tr (8842.5,798573,48438.3) (5.63227e-17,0.0801989,-0.996779)(9.42949e-17,0.996779,0.0801989)(1,-9.85082e-17,4.85789e-17)


    In [32]: np.set_printoptions(suppress=True, precision=4)

    In [34]: a = np.load("/usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml.cache/data.npy")

    In [35]: a[0]
    Out[35]: 
    array([[      0.    ,       0.7615,      -0.6481,    8842.5   ],
           [      0.    ,       0.6481,       0.7615,  532069.326 ],
           [      1.    ,      -0.    ,       0.    ,  599608.6129],
           [      0.    ,       0.    ,       0.    ,       1.    ]])

    In [36]: k = np.load("/usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml.cache/key.npy")

    In [37]: k[0]
    Out[37]: 3200


    In [40]: idmap.a[k[0]]
    Out[40]: (3200, 16843009, '1010101', '(8842.5,532069,599609)', '(3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17)', '/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum')

    In [41]: idmap.a[k[1]]
    Out[41]: (3206, 16843010, '1010102', '(8842.5,668528,441547)', '(5.04009e-17,0.567844,-0.823136)(-3.47693e-17,0.823136,0.567844)(1,0,6.12303e-17)', '/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum')

    In [42]: idmap.a[k[2]]
    Out[42]: (3212, 16843011, '1010103', '(8842.5,759428,253553)', '(5.76825e-17,0.335452,-0.942057)(-2.05398e-17,0.942057,0.335452)(1,6.16298e-33,6.12303e-17)', '/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum')

    In [43]: a[0]
    Out[43]: 
    array([[      0.    ,       0.7615,      -0.6481,    8842.5   ],
           [      0.    ,       0.6481,       0.7615,  532069.326 ],
           [      1.    ,      -0.    ,       0.    ,  599608.6129],
           [      0.    ,       0.    ,       0.    ,       1.    ]])

    In [44]: a[1]
    Out[44]: 
    array([[      0.    ,       0.5678,      -0.8231,    8842.5   ],
           [      0.    ,       0.8231,       0.5678,  668528.0071],
           [      1.    ,      -0.    ,       0.    ,  441546.9754],
           [      0.    ,       0.    ,       0.    ,       1.    ]])

    In [45]: a[2]
    Out[45]: 
    array([[      0.    ,       0.3355,      -0.9421,    8842.5   ],
           [      0.    ,       0.9421,       0.3355,  759427.6093],
           [      1.    ,      -0.    ,       0.    ,  253553.0521],
           [      0.    ,       0.    ,       0.    ,       1.    ]])




"""
import os
import numpy as np

np.set_printoptions(suppress=True, precision=3 )

from idmap import IDMap

path = os.environ['IDMAP']
idmap = IDMap(path, old=False)


gdml = "/usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml"
dd = np.load(gdml + ".cache/data.npy")
kk = np.load(gdml + ".cache/key.npy")

for i,k in enumerate(kk):
   m = idmap.a[k]
   d = dd[i]
   print m 
   print d




