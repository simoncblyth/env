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



