DetSim Parallel Cerenkov and Scintillation
============================================

* Can the photon loops be parallelized by moving
  generation onto the GPU ?

* This would largely avoid transport overheads.


Everything that is constant from the point of view of the 
photon cohort needs to be collected into the "G4DAEStep" 
object. Although some things could potentially 
be looked up from material props on GPU no point doing that 
when are constant for all photons, just do it once
and present as parameter to the kernel launch.

::

    m_chroma->CollectCerenkovStep( 
                    G4Step& step,
                   
                )




Validating GPU generated photons
-----------------------------------

Scintillation photons::

    In [1]: t = tt(1)

    In [2]: t.shape
    Out[2]: (2652646, 4, 4)

    plt.hist(t[:,0,3], bins=100, range=(0,100))     # time distrib, smooth

    plt.hist(t[:,1,3], bins=100 )   # distinct coarsely binned structure of underlying distrib apparent ?


Cerenkov wavelength distrib very flat ? 

Need to collect geant4 originals in same 
format to allow direct comparison.


Properties
----------

::

    delta:~ blyth$ export-
    delta:~ blyth$ export-export
    delta:~ blyth$ find $DAE_NAME_DYB_CHROMACACHE -name reemission_cdf.npy | grep Gd
    /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae.29c299d81706c62884caf5c3dbdea5c1/chroma_geometry/chroma.detector:Detector:0x11ca48510/unique_materials/003/chroma.geometry:Material:__dd__Materials__GdDopedLS0xc2a8ed0/reemission_cdf.npy
    delta:~ blyth$ 




Lookups for Cerenkov
---------------------

::

    In [1]: ri = np.load("./chroma.detector:Detector:0x11ca48510/unique_materials/000/chroma.geometry:Material:__dd__Materials__LiquidScintillator0xc2308d0/refractive_index.npy")

    In [2]: ri
    Out[2]: 
    array([[  79.99 ,    1.454],
           [ 120.023,    1.454],
           [ 129.99 ,    1.554],
           [ 139.984,    1.664],
           [ 149.975,    1.783],
           [ 159.98 ,    1.793],
           [ 169.981,    1.554],
           [ 179.974,    1.527],
           [ 189.985,    1.618],
           [ 199.975,    1.618],
           [ 300.   ,    1.526],
           [ 404.7  ,    1.499],
           [ 435.8  ,    1.495],
           [ 486.001,    1.492],
           [ 546.001,    1.486],
           [ 589.002,    1.484],
           [ 690.701,    1.48 ],
           [ 799.898,    1.478]], dtype=float32)





Copy over to DetSimChroma and change class names
-------------------------------------------------

::

    [blyth@ntugrid5 Simulation]$ cp DetSim/src/DsG4Scintillation.cc DetSimChroma/src/DsChromaG4Scintillation.cc
    [blyth@ntugrid5 Simulation]$ cp DetSim/src/DsG4Scintillation.h  DetSimChroma/src/DsChromaG4Scintillation.h
    [blyth@ntugrid5 Simulation]$ cp DetSim/src/DsG4Cerenkov.cc    DetSimChroma/src/DsChromaG4Cerenkov.cc
    [blyth@ntugrid5 Simulation]$ cp DetSim/src/DsG4Cerenkov.h     DetSimChroma/src/DsChromaG4Cerenkov.h
    [blyth@ntugrid5 Simulation]$ cp DetSim/src/DsPhysConsOptical.cc DetSimChroma/src/DsChromaPhysConsOptical.cc
    [blyth@ntugrid5 Simulation]$ cp DetSim/src/DsPhysConsOptical.h DetSimChroma/src/DsChromaPhysConsOptical.h

::

    [blyth@ntugrid5 Simulation]$ perl -pi -e 's,DsG4Scintillation,DsChromaG4Scintillation,g' DetSimChroma/src/DsChromaG4Scintillation.h 
    [blyth@ntugrid5 Simulation]$ perl -pi -e 's,DsG4Scintillation,DsChromaG4Scintillation,g' DetSimChroma/src/DsChromaG4Scintillation.cc
    [blyth@ntugrid5 Simulation]$ perl -pi -e 's,DsG4Scintillation,DsChromaG4Scintillation,g' DetSimChroma/src/DsChromaPhysConsOptical.cc

::

    [blyth@ntugrid5 Simulation]$ perl -pi -e 's,DsG4Cerenkov,DsChromaG4Cerenkov,g' DetSimChroma/src/DsChromaG4Cerenkov.cc
    [blyth@ntugrid5 Simulation]$ perl -pi -e 's,DsG4Cerenkov,DsChromaG4Cerenkov,g' DetSimChroma/src/DsChromaG4Cerenkov.h
    [blyth@ntugrid5 Simulation]$ perl -pi -e 's,DsG4Cerenkov,DsChromaG4Cerenkov,g' DetSimChroma/src/DsChromaPhysConsOptical.cc


Also need this header only class::

    [blyth@ntugrid5 Simulation]$ cp DetSim/src/DsPhotonTrackInfo.h DetSimChroma/src/DsChromaPhotonTrackInfo.h


    [blyth@ntugrid5 src]$ perl -pi -e 's,DsPhotonTrackInfo,DsChromaPhotonTrackInfo,g' DsChromaG4Scintillation.cc
    [blyth@ntugrid5 src]$ perl -pi -e 's,DsPhotonTrackInfo,DsChromaPhotonTrackInfo,g' DsChromaG4Cerenkov.cc
    [blyth@ntugrid5 src]$ 


    [blyth@ntugrid5 src]$ perl -pi -e 's,DsPhysConsOptical,DsChromaPhysConsOptical,g' DsChromaPhysConsOptical.cc
    [blyth@ntugrid5 src]$ perl -pi -e 's,DsPhysConsOptical,DsChromaPhysConsOptical,g' DsChromaPhysConsOptical.h


    [blyth@ntugrid5 Simulation]$ cp DetSim/src/DsG4OpRayleigh.h DetSimChroma/src/DsChromaG4OpRayleigh.h
    [blyth@ntugrid5 Simulation]$ cp DetSim/src/DsG4OpRayleigh.cc DetSimChroma/src/DsChromaG4OpRayleigh.cc
    [blyth@ntugrid5 Simulation]$ cp DetSim/src/DsG4OpBoundaryProcess.h DetSimChroma/src/DsChromaG4OpBoundaryProcess.h
    [blyth@ntugrid5 Simulation]$ cp DetSim/src/DsG4OpBoundaryProcess.cc DetSimChroma/src/DsChromaG4OpBoundaryProcess.cc
    [blyth@ntugrid5 Simulation]$ 
    [blyth@ntugrid5 Simulation]$ perl -pi -e 's,DsG4OpRayleigh,DsChromaG4OpRayleigh,g' DetSimChroma/src/DsChromaG4OpRayleigh.h DetSimChroma/src/DsChromaG4OpRayleigh.cc
    [blyth@ntugrid5 Simulation]$ perl -pi -e 's,DsG4OpRayleigh,DsChromaG4OpRayleigh,g' DetSimChroma/src/DsChromaPhysConsOptical.cc
    [blyth@ntugrid5 Simulation]$ perl -pi -e 's,DsG4OpBoundaryProcess,DsChromaG4OpBoundaryProcess,g' DetSimChroma/src/DsChromaPhysConsOptical.cc DetSimChroma/src/DsChromaG4OpBoundaryProcess.cc DetSimChroma/src/DsChromaG4OpBoundaryProcess.h
    [blyth@ntugrid5 Simulation]$ 




Material Properties for Scintillation/Cerenkov GPU generation
---------------------------------------------------------------

::

    delta:~ blyth$ collada_to_chroma.sh 
    INFO:env.geant4.geometry.collada.idmap:np.genfromtxt /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.idmap 
    INFO:env.geant4.geometry.collada.idmap:found 685 unique ids 
    INFO:env.geant4.geometry.collada.g4daenode:idmap exists /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.idmap entries 12230 
    INFO:env.geant4.geometry.collada.g4daenode:index linking DAENode with boundgeom 12230 volumes 
    INFO:env.geant4.geometry.collada.g4daenode:linking DAENode with idmap 12230 identifiers 
    INFO:env.geant4.geometry.collada.g4daenode:add_sensitive_surfaces matid __dd__Materials__Bialkali qeprop EFFICIENCY 
    INFO:env.geant4.geometry.collada.g4daenode:sensitize 684 nodes with matid __dd__Materials__Bialkali and channel_id > 0, uniques 684 
    INFO:env.geant4.geometry.collada.collada_to_chroma:convert_opticalsurfaces
    INFO:env.geant4.geometry.collada.collada_to_chroma:convert_opticalsurfaces creates 44 from 726  
    WARNING:env.geant4.geometry.collada.collada_to_chroma:setting parent_material to __dd__Materials__Vacuum0xbf9fcc0 as parent is None for node top.0 
    INFO:env.geant4.geometry.collada.collada_to_chroma:channel_count (nodes with channel_id > 0) : 6888  uniques 684 
    INFO:env.geant4.geometry.collada.collada_to_chroma:convert_geometry DONE timing_report: 
    INFO:env.base.timing:timing_report
    ColladaToChroma 
    __init__                       :      0.000          1      0.000 
    convert_flatten                :      2.429          1      2.429 
    convert_geometry_traverse      :      4.475          1      4.475 
    convert_make_maps              :      0.000          1      0.000 
    convert_materials              :      0.009          1      0.009 
    convert_opticalsurfaces        :      0.233          1      0.233 
    INFO:env.geant4.geometry.collada.collada_to_chroma:dropping into IPython.embed() try: cg.<TAB> 
    Python 2.7.8 (default, Jul 13 2014, 17:11:32) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: gdls
    Out[1]: <chroma.geometry.Material at 0x10dd0cc50>

    In [3]: self = cc

    In [5]: collada = self.nodecls.orig

    In [6]: collada.materials
    Out[6]: 
    [<Material id=__dd__Materials__PPE0xc12f008 effect=__dd__Materials__PPE_fx_0xc12f008>,
     <Material id=__dd__Materials__MixGas0xc21d930 effect=__dd__Materials__MixGas_fx_0xc21d930>,
     <Material id=__dd__Materials__Air0xc032550 effect=__dd__Materials__Air_fx_0xc032550>,
     <Material id=__dd__Materials__Bakelite0xc2bc240 effect=__dd__Materials__Bakelite_fx_0xc2bc240>,
     <Material id=__dd__Materials__Foam0xc558e28 effect=__dd__Materials__Foam_fx_0xc558e28>,
     <Material id=__dd__Materials__Aluminium0xc542070 effect=__dd__Materials__Aluminium_fx_0xc542070>,
     <Material id=__dd__Materials__Iron0xc542700 effect=__dd__Materials__Iron_fx_0xc542700>,
     <Material id=__dd__Materials__GdDopedLS0xc2a8ed0 effect=__dd__Materials__GdDopedLS_fx_0xc2a8ed0>,
     <Material id=__dd__Materials__Acrylic0xc02ab98 effect=__dd__Materials__Acrylic_fx_0xc02ab98>,
     <Material id=__dd__Materials__Teflon0xc129f90 effect=__dd__Materials__Teflon_fx_0xc129f90>,
     <Material id=__dd__Materials__LiquidScintillator0xc2308d0 effect=__dd__Materials__LiquidScintillator_fx_0xc2308d0>,
     <Material id=__dd__Materials__Bialkali0xc2f2428 effect=__dd__Materials__Bialkali_fx_0xc2f2428>,
     <Material id=__dd__Materials__OpaqueVacuum0xbf5d600 effect=__dd__Materials__OpaqueVacuum_fx_0xbf5d600>,
     <Material id=__dd__Materials__Vacuum0xbf9fcc0 effect=__dd__Materials__Vacuum_fx_0xbf9fcc0>,
     <Material id=__dd__Materials__Pyrex0xc1005e0 effect=__dd__Materials__Pyrex_fx_0xc1005e0>,
     <Material id=__dd__Materials__UnstStainlessSteel0xc5c11e8 effect=__dd__Materials__UnstStainlessSteel_fx_0xc5c11e8>,
     <Material id=__dd__Materials__PVC0xc25cfe8 effect=__dd__Materials__PVC_fx_0xc25cfe8>,
     <Material id=__dd__Materials__StainlessSteel0xc2adc00 effect=__dd__Materials__StainlessSteel_fx_0xc2adc00>,
     <Material id=__dd__Materials__ESR0xbf9f438 effect=__dd__Materials__ESR_fx_0xbf9f438>,
     <Material id=__dd__Materials__Nylon0xc3aa360 effect=__dd__Materials__Nylon_fx_0xc3aa360>,
     <Material id=__dd__Materials__MineralOil0xbf5c830 effect=__dd__Materials__MineralOil_fx_0xbf5c830>,
     <Material id=__dd__Materials__BPE0xc0ad360 effect=__dd__Materials__BPE_fx_0xc0ad360>,
     <Material id=__dd__Materials__Ge_680xc2d7e60 effect=__dd__Materials__Ge_68_fx_0xc2d7e60>,
     <Material id=__dd__Materials__Co_600xc3cf0c0 effect=__dd__Materials__Co_60_fx_0xc3cf0c0>,
     <Material id=__dd__Materials__C_130xc3d0ab0 effect=__dd__Materials__C_13_fx_0xc3d0ab0>,
     <Material id=__dd__Materials__Silver0xc3d1370 effect=__dd__Materials__Silver_fx_0xc3d1370>,
     <Material id=__dd__Materials__Nitrogen0xc031fd0 effect=__dd__Materials__Nitrogen_fx_0xc031fd0>,
     <Material id=__dd__Materials__Water0xc176e30 effect=__dd__Materials__Water_fx_0xc176e30>,
     <Material id=__dd__Materials__NitrogenGas0xc17d300 effect=__dd__Materials__NitrogenGas_fx_0xc17d300>,
     <Material id=__dd__Materials__IwsWater0xc288f98 effect=__dd__Materials__IwsWater_fx_0xc288f98>,
     <Material id=__dd__Materials__ADTableStainlessSteel0xc177178 effect=__dd__Materials__ADTableStainlessSteel_fx_0xc177178>,
     <Material id=__dd__Materials__Tyvek0xc246ca0 effect=__dd__Materials__Tyvek_fx_0xc246ca0>,
     <Material id=__dd__Materials__OwsWater0xbf90c10 effect=__dd__Materials__OwsWater_fx_0xbf90c10>,
     <Material id=__dd__Materials__DeadWater0xbf8a548 effect=__dd__Materials__DeadWater_fx_0xbf8a548>,
     <Material id=__dd__Materials__RadRock0xcd2f508 effect=__dd__Materials__RadRock_fx_0xcd2f508>,
     <Material id=__dd__Materials__Rock0xc0300c8 effect=__dd__Materials__Rock_fx_0xc0300c8>]

    In [7]: collada.materials[7]
    Out[7]: <Material id=__dd__Materials__GdDopedLS0xc2a8ed0 effect=__dd__Materials__GdDopedLS_fx_0xc2a8ed0>

    In [8]: collada.materials[7].extra
    Out[8]: <MaterialProperties keys=['SLOWTIMECONSTANT', 'GammaFASTTIMECONSTANT', 'ReemissionSLOWTIMECONSTANT', 'REEMISSIONPROB', 'AlphaFASTTIMECONSTANT', 'ReemissionFASTTIMECONSTANT', 'SLOWCOMPONENT', 'YIELDRATIO', 'FASTCOMPONENT', 'RINDEX', 'NeutronFASTTIMECONSTANT', 'ReemissionYIELDRATIO', 'RAYLEIGH', 'NeutronYIELDRATIO', 'GammaYIELDRATIO', 'SCINTILLATIONYIELD', 'AlphaYIELDRATIO', 'RESOLUTIONSCALE', 'GammaSLOWTIMECONSTANT', 'AlphaSLOWTIMECONSTANT', 'NeutronSLOWTIMECONSTANT', 'ABSLENGTH', 'FASTTIMECONSTANT'] >

    In [9]: 

    In [11]: collada.materials[7].extra.properties
    Out[11]: 
    {'ABSLENGTH': array([[  79.9898,    0.001 ],
           [ 120.0235,    0.001 ],
           [ 199.9746,    0.001 ],
           ..., 
           [ 897.916 ,  328.4   ],
           [ 898.8925,  306.2   ],
           [ 899.8711,  299.6   ]]),
     'AlphaFASTTIMECONSTANT': array([[ 0.0012,  1.    ],
           [-0.0012,  1.    ]]),
     'AlphaSLOWTIMECONSTANT': array([[  0.0012,  35.    ],
           [ -0.0012,  35.    ]]),
     'AlphaYIELDRATIO': array([[ 0.0012,  0.65  ],
           [-0.0012,  0.65  ]]),
     'FASTCOMPONENT': array([[  79.9898,    0.    ],
           [ 120.0235,    0.    ],
           [ 199.9746,    0.    ],
           ..., 
           [ 599.0011,    0.0017],
           [ 600.0012,    0.0018],
           [ 799.8984,    0.    ]]),
     'FASTTIMECONSTANT': array([[ 0.0012,  3.64  ],
           [-0.0012,  3.64  ]]),
     'GammaFASTTIMECONSTANT': array([[ 0.0012,  7.    ],
           [-0.0012,  7.    ]]),
     'GammaSLOWTIMECONSTANT': array([[  0.0012,  31.    ],
           [ -0.0012,  31.    ]]),
     'GammaYIELDRATIO': array([[ 0.0012,  0.805 ],
           [-0.0012,  0.805 ]]),
     'NeutronFASTTIMECONSTANT': array([[ 0.0012,  1.    ],
           [-0.0012,  1.    ]]),
     'NeutronSLOWTIMECONSTANT': array([[  0.0012,  34.    ],
           [ -0.0012,  34.    ]]),
     'NeutronYIELDRATIO': array([[ 0.0012,  0.65  ],
           [-0.0012,  0.65  ]]),
     'RAYLEIGH': array([[     79.9898,     850.    ],
           [    120.0235,     850.    ],
           [    199.9746,     850.    ],
           ..., 
           [    589.8394,  170000.    ],
           [    699.9223,  300000.    ],
           [    799.8984,  500000.    ]]),
     'REEMISSIONPROB': array([[  79.9898,    0.4   ],
           [ 120.0235,    0.4   ],
           [ 159.9797,    0.4   ],
           ..., 
           [ 575.8273,    0.0587],
           [ 712.6064,    0.    ],
           [ 799.8984,    0.    ]]),
     'RESOLUTIONSCALE': array([[ 0.0012,  1.    ],
           [-0.0012,  1.    ]]),
     'RINDEX': array([[  79.9898,    1.4536],
           [ 120.0235,    1.4536],
           [ 129.9898,    1.5545],
           ..., 
           [ 589.0016,    1.4842],
           [ 690.7008,    1.48  ],
           [ 799.8984,    1.4781]]),
     'ReemissionFASTTIMECONSTANT': array([[ 0.0012,  1.5   ],
           [-0.0012,  1.5   ]]),
     'ReemissionSLOWTIMECONSTANT': array([[ 0.0012,  1.5   ],
           [-0.0012,  1.5   ]]),
     'ReemissionYIELDRATIO': array([[ 0.0012,  1.    ],
           [-0.0012,  1.    ]]),
     'SCINTILLATIONYIELD': array([[     0.0012,  11522.    ],
           [    -0.0012,  11522.    ]]),
     'SLOWCOMPONENT': array([[  79.9898,    0.    ],
           [ 120.0235,    0.    ],
           [ 199.9746,    0.    ],
           ..., 
           [ 599.0011,    0.0017],
           [ 600.0012,    0.0018],
           [ 799.8984,    0.    ]]),
     'SLOWTIMECONSTANT': array([[  0.0012,  12.2   ],
           [ -0.0012,  12.2   ]]),
     'YIELDRATIO': array([[ 0.0012,  0.86  ],
           [-0.0012,  0.86  ]])}

    In [12]: 





    In [12]: collada.materials[7].extra.properties['SLOWCOMPONENT']
    Out[12]: 
    array([[  79.9898,    0.    ],
           [ 120.0235,    0.    ],
           [ 199.9746,    0.    ],
           ..., 
           [ 599.0011,    0.0017],
           [ 600.0012,    0.0018],
           [ 799.8984,    0.    ]])

    In [13]: collada.materials[7].extra.properties['FASTCOMPONENT']
    Out[13]: 
    array([[  79.9898,    0.    ],
           [ 120.0235,    0.    ],
           [ 199.9746,    0.    ],
           ..., 
           [ 599.0011,    0.0017],
           [ 600.0012,    0.0018],
           [ 799.8984,    0.    ]])

    In [14]: collada.materials[7].extra.properties['REEMISSIONPROB']
    Out[14]: 
    array([[  79.9898,    0.4   ],
           [ 120.0235,    0.4   ],
           [ 159.9797,    0.4   ],
           ..., 
           [ 575.8273,    0.0587],
           [ 712.6064,    0.    ],
           [ 799.8984,    0.    ]])

    In [15]: 


    In [15]: np.allclose( collada.materials[7].extra.properties['SLOWCOMPONENT'], collada.materials[7].extra.properties['FASTCOMPONENT'] )
    Out[15]: True



