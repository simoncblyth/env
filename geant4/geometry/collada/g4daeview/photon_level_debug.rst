Photon Level Debug
===================

Issues
---------


spagetti style and surface step selection incompatible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other styles more resonable, spagetti is finnicky as its LINE_STRIP based. So trying
to hide steps by sending off to infinity creates funny renders.::

    udp.py --surface RSOil



Pushing Up the Slots causes BUS error (AVOIDED)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

max slots of 100 or 50 load noodle style photon render, 
but on switching to movie style get a bus error:: 

    *** Break *** bus error
     Generating stack trace...
     0x00007fff8a06c3a9 in glMultiDrawArrays_ACC_Exec (in GLEngine) + 565
     0x00000001073706e7 in ffi_call_unix64 (in _ctypes.so) + 79
     0x00007fff59315790 in <unknown function>
    -------------------------------------------------------------------
    PyCUDA ERROR: The context stack was not empty upon module cleanup.
    -------------------------------------------------------------------
    A context was still active when the context stack was being
    cleaned up. At this point in our execution, CUDA may already
    have been deinitialized, so there is no way we can finish
    cleanly. The program will be aborted now.
    Use Context.pop() to avoid this problem.
    -------------------------------------------------------------------
    /Users/blyth/env/bin/g4daeview.sh: line 64: 61267 Abort trap: 6           g4daeview.py $*
    delta:e blyth$ 
    delta:e blyth$ 
    delta:e blyth$ g4daeview.sh  --load 1 --wipepropagate --debugkernel --debugphoton 0 --max-slots 30


Guessed causes ?

#. drawing time exceeds some OpenGL timeout ?

#. moving to draw rather than multidraw succeeds to animate even max_slots 100 but very slowly,
   assuming due to photon-x-100 VBO memory size pushes geometry VBO out the door



Single Photon Debug
~~~~~~~~~~~~~~~~~~~~~

::

    udp.py --style spagetti,confetti,noodles,movie-extra --sid 3354



Photon Picking 
~~~~~~~~~~~~~~~

Works OK in movie mode. Not so well in noddles, confetti modes.
Because is looking at the nearest positioned photon at a particular time
using slot -1. 



Make Bialkali absorb rate ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    014-07-07 16:39:49,336 env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:209 summary for pid 4041 
    p_flags[4041]                      p_lht[4041]                                          
               [[ 228.349   228.349 ]  [[  1237      3      5      0] GdDopedLS  Acrylic    
                [ 228.349   234.2408]   [   949      5     11      1] Acrylic    LiquidScin 
                [ 228.349   234.3079]   [   615     11      5      2] LiquidScin Acrylic    
                [ 228.349   237.0455]   [   327      5      6      3] Acrylic    MineralOil 
                [ 228.349   237.1461]   [363865      6     15      4] MineralOil Pyrex      
                [ 228.349   238.4965]   [364826     15      9      5] Pyrex      Vacuum     
                [ 228.349   238.5126]   [365759      9     12      6] Vacuum     Bialkali   
    R_SPECULAR  [ 228.349   307.1889]   [635507      5      6      7] Acrylic    MineralOil 
    R_SPECULAR  [ 228.349   307.1889]   [635507      5      6      7] Acrylic    MineralOil 
    R_SPECULAR  [ 228.349   307.1889]]  [635507      5      6      7]] Acrylic    MineralOil
    p_post[4041]                                            p_dirw[4041]



Quite a lot of many step histories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Study these, are they just due to lots of specular reflections ?

Visual distinction for zombie photons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Truncation transitions result in zombie/drunken photons
linearly interpolating around. Need some visual distinction to
make clear not real propagation paths. 

Difficult to zoom into closeups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

eg to see photons crossing acrylic
This is due to view jumping when bookmarking 

Photon Appearance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Photons appearing behind volumes, visible due to transparent effect
  can look orange (other) when actually red (absorption).  
  Make geometry invisible with `I` to check actual color.

* Constant pixel size blobs can make the position of the photon deceptive.
  TODO: blob size variation with distance 

* Difficult to zoom in on a picked photon.
  TODO: allow viewpoint creation based on a picked photon position 


ESR Step start Asymmetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selecting ESR see many more from bottom ESR than from top ?
This is true with "noodle" style, but not "movie-extra". 

This asymmetry problem (not in movie-extra) remains 
following the wider presentation mode discrepancy fix below.


Material Step Starts
~~~~~~~~~~~~~~~~~~~~~~

Selecting steps that start in particular materials mostly gives
understandable results, with steps in expected places in geometry.

But some materials lead to seemingly wrong results, eg for steps
starting in Acrylic having entries along muon path.

NB to test this in animation modes be sure to push out the 
time otherwise the zombies could be misleading.::

   udp.py --timerange 0,5000 --time 5000

After that the liquids look OK, but still problems with Acrylic and ESR.


Discrepancy between presentation modes (FIXED)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dots and lines were not lining up. 
Fixed by restoration of VBO slot contiguity (at expense of always duplicating the last slot)::

    udp.py --style noodles,movie-extra --time 60  # red (absorbed) are mostly matching at 60ns
    udp.py --style confetti,noodles

Is this a problem with firsts, counts following moving last to -2::

    gl.glMultiDrawArrays( mode, firsts, counts, drawcount )

Yep. Drawing needs contiguity.


Material Selection not compatible with Spagetti style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using spagetti with material selection results in 
lines going off to infinity. This is how photon selection is implemented
but thats not compatible with LINE_STRIP drawing.


Timeconstant for reemitted photons ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Infinite wavelength for reemitted photons (FIXED)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FIXED by providing a `reemission_cdf` obtained by `np.cumsum` of the FASTCOMPONENT property.

This is following the spirit of whats done by `void DsG4Scintillation::BuildThePhysicsTable()` 
whether its close enough remains to be determined.

::

    delta:~ blyth$ collada_to_chroma.sh
    INFO:env.geant4.geometry.collada.collada_to_chroma:daeload path /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae 
    WARNING:env.geant4.geometry.collada.collada_to_chroma:setting parent_material to __dd__Materials__Vacuum0xbf9fcc0 as parent is None for node top.0 
    INFO:env.geant4.geometry.collada.collada_to_chroma:dropping into IPython.embed() try: g.<TAB> 
    Python 2.7.6 (default, Nov 18 2013, 15:12:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: g.
    g.add_solid            g.colors               g.flatten              g.material2_index      g.solid_displacements  g.solid_rotations      g.surface_index        g.unique_surfaces
    g.bvh                  g.detector_material    g.material1_index      g.mesh                 g.solid_id             g.solids               g.unique_materials     

    In [1]: g.material1_index
    Out[1]: array([13, 13, 13, ..., 34, 34, 34], dtype=int32)

    In [2]: map(len,[g.material1_index,g.material2_index,g.surface_index,g.unique_materials,g.unique_surfaces])
    Out[2]: [2448160, 2448160, 2448160, 36, 35]


Clicking a green photon at random, see the usual infinite wavelength::

    2014-07-02 16:20:52,345 env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:192 summary for pid 1337 
    p_flags[1337]                                   p_lht[1337]                    
    [[         0 1093812380 1093812380          2]  [[  1217   1337      1      0] 
     [       512 1093812380 1093812380          2]   [  1254   1337      2      1] 
     [       512 1093812380 1100465654          3]   [   990   1337      3      2] 
     [       512 1093812380 1100505046          5]   [   634   1337      4      3] 
     [       512 1093812380 1101658367         15]   [631702   1337      5      4] 
     [       512 1093812380 1101704640          3]   [632006   1337      6      5] 
     [       512 1093812380 1101730630          0]   [632304   1337      7      6] 
     [       512 1093812380 1101730719         13]   [632331   1337      8      7] 
     [       514 1093812380 1101730719         13]   [    -1   1337      8      8] 
     [       514 1093812380 1101730719         13]]  [    -1   1337      8      8]]
    p_post[1337]                                            p_dirw[1337]                            p_polw[1337]                        p_ccol[1337]       
    [[ -18229.1035 -799469.375    -7061.5503      11.1408]  [[ -0.0442   0.9029   0.4276  88.9868]  [[ 0.9737 -0.0568  0.2204  1.    ]  [[ 1.  1.  1.  1.] 
     [ -18229.1035 -799469.375    -7061.5503      11.1408]   [  0.0269   0.1337  -0.9907      inf]   [ 0.8209  0.5626  0.0982  1.    ]   [ 0.  1.  0.  1.] 
     [ -18186.377  -799257.0625   -8635.          18.9717]   [  0.0267   0.1328  -0.9908      inf]   [ 0.1955  0.9713  0.1355  1.    ]   [ 0.  1.  0.  1.] 
     [ -18185.9727 -799255.0625   -8650.          19.0468]   [  0.0269   0.1337  -0.9907      inf]   [ 0.1955  0.9712  0.1363  1.    ]   [ 0.  1.  0.  1.] 
     [ -18173.9707 -799195.4375   -9092.          21.2466]   [  0.0267   0.1328  -0.9908      inf]   [ 0.1955  0.9713  0.1355  1.    ]   [ 0.  1.  0.  1.] 
     [ -18173.4844 -799193.       -9110.          21.3348]   [  0.0262   0.13    -0.9912      inf]   [ 0.1956  0.9717  0.1326  1.    ]   [ 0.  1.  0.  1.] 
     [ -18173.2227 -799191.6875   -9119.9004      21.3844]   [  0.0389   0.1934  -0.9804      inf]   [ 0.1934  0.9611  0.1972  1.    ]   [ 0.  1.  0.  1.] 
     [ -18173.2207 -799191.6875   -9119.9502      21.3846]   [  0.0389   0.1934  -0.9803      inf]   [ 0.1934  0.9611  0.1973  1.    ]   [ 0.  1.  0.  1.] 
     [ -18173.2207 -799191.6875   -9119.9502      21.3846]   [  0.0389   0.1934  -0.9803      inf]   [ 0.1934  0.9611  0.1973  1.    ]   [ 1.  0.  0.  1.] 
     [ -18199.0137 -799319.875    -8169.6206      16.6555]]  [  0.0269   0.1337  -0.9907      inf]]  [ 0.8209  0.5626  0.0982  1.    ]]  [ 0.  1.  0.  1.]]
    t_post[1337]                                          t_dirw[1337]                      t_polw[1337]                      t_ccol[1337]     
    [ -18199.0137 -799319.875    -8169.6206      16.6555] [ 0.0269  0.1337 -0.9907     inf] [ 0.8209  0.5626  0.0982  1.    ] [ 0.  1.  0.  1.]
    2014-07-02 16:20:52,350 env.geant4.geometry.collada.g4daeview.daephotons:108 clicked_point (-18198.09676577193, -799326.9836636602, -8180.765649884277) => index 1337 


Rerun with that photon in debug::

    g4daeview.sh --with-chroma --load 1 --debugkernel --debugphoton 1337 --wipepropagate

    materials 2:GdDopedLS 
              3:Acrylic  
              5:LiquidScintillator 
             13:ESR
             15:MineralOil
              0:Air

Looks like GdDopedLS has a reemission probability of 0.4 and no wavelength distribution to back it up::

    2014-07-02 17:15:22,228 env.geant4.geometry.collada.g4daeview.daechromacontext:59  setup_rng_states using seed 0 
    [  1]   1337 material_code 33816320 inner 2 outer 3 si -1 ri1 1.453600 ri2 1.462000 abs 0.001000 sca 850.000000 rem 0.400000 ncdf -0.000008 w0 60.000000 st 20.000000 cdf lo/up 0.000000 0.000000 
    [  2]   1337 material_code 33816320 inner 2 outer 3 si -1 ri1 1.478100 ri2 1.487800 abs 3358.373535 sca 500000.000000 rem 0.000000 ncdf -0.000008 w0 60.000000 st 20.000000 cdf lo/up 0.000000 0.000000 
    [  3]   1337 material_code 50724608 inner 3 outer 5 si -1 ri1 1.487800 ri2 1.478100 abs 8000.000000 sca 500000.000000 rem 0.000000 ncdf -0.000008 w0 60.000000 st 20.000000 cdf lo/up 0.000000 0.000000 
    [  4]   1337 material_code 84147968 inner 5 outer 3 si -1 ri1 1.478100 ri2 1.487800 abs 3236.346924 sca 500000.000000 rem 0.000000 ncdf -0.000008 w0 60.000000 st 20.000000 cdf lo/up 0.000000 0.000000 
    [  5]   1337 material_code 51379968 inner 3 outer 15 si -1 ri1 1.456400 ri2 1.487800 abs 2672.763672 sca 500000.000000 rem 0.000000 ncdf -0.000008 w0 60.000000 st 20.000000 cdf lo/up 0.000000 0.000000 
    [  6]   1337 material_code 261888 inner 0 outer 3 si -1 ri1 1.487800 ri2 1.000270 abs 8000.000000 sca 500000.000000 rem 0.000000 ncdf -0.000008 w0 60.000000 st 20.000000 cdf lo/up 0.000000 0.000000 
    [  7]   1337 material_code 218169088 inner 13 outer 0 si -1 ri1 1.000270 ri2 1.000000 abs 10000000.000000 sca 1000000.000000 rem 0.000000 ncdf -0.000008 w0 60.000000 st 20.000000 cdf lo/up 0.000000 0.000000 
    [  8]   1337 material_code 218169088 inner 13 outer 0 si -1 ri1 1.000000 ri2 1.000270 abs 0.001000 sca 1000000.000000 rem 0.000000 ncdf -0.000008 w0 60.000000 st 20.000000 cdf lo/up 0.000000 0.000000 
    FILL_STATE       START    [  1337] slot  1 steps  1 lht   1217 tpos   11.141  -18229.10 -799469.38   -7061.55    w   88.99   dir    -0.04     0.90     0.43 pol    0.974   -0.057    0.220 
    TO_BOUNDARY      CONTINUE [  1337] slot -1 steps  1 lht   1217 tpos   11.141  -18229.10 -799469.38   -7061.55    w     inf   dir     0.03     0.13    -0.99 pol    0.821    0.563    0.098 BULK_REEMIT 
    FILL_STATE       CONTINUE [  1337] slot  2 steps  2 lht   1254 tpos   11.141  -18229.10 -799469.38   -7061.55    w     inf   dir     0.03     0.13    -0.99 pol    0.821    0.563    0.098 BULK_REEMIT 
    TO_BOUNDARY      PASS     [  1337] slot -1 steps  2 lht   1254 tpos   18.972  -18186.38 -799257.06   -8635.00    w     inf   dir     0.03     0.13    -0.99 pol    0.821    0.563    0.098 BULK_REEMIT 
    AT_BOUNDARY      CONTINUE [  1337] slot -1 steps  2 lht   1254 tpos   18.972  -18186.38 -799257.06   -8635.00    w     inf   dir     0.03     0.13    -0.99 pol    0.195    0.971    0.135 BULK_REEMIT 
    FILL_STATE       PASS     [  1337] slot  3 steps  3 lht    990 tpos   18.972  -18186.38 -799257.06   -8635.00    w     inf   dir     0.03     0.13    -0.99 pol    0.195    0.971    0.135 BULK_REEMIT 
    TO_BOUNDARY      PASS     [  1337] slot -1 steps  3 lht    990 tpos   19.047  -18185.97 -799255.06   -8650.00    w     inf   dir     0.03     0.13    -0.99 pol    0.195    0.971    0.135 BULK_REEMIT 
    AT_BOUNDARY      CONTINUE [  1337] slot -1 steps  3 lht    990 tpos   19.047  -18185.97 -799255.06   -8650.00    w     inf   dir     0.03     0.13    -0.99 pol    0.195    0.971    0.136 BULK_REEMIT 
    FILL_STATE       PASS     [  1337] slot  4 steps  4 lht    634 tpos   19.047  -18185.97 -799255.06   -8650.00    w     inf   dir     0.03     0.13    -0.99 pol    0.195    0.971    0.136 BULK_REEMIT 
    TO_BOUNDARY      PASS     [  1337] slot -1 steps  4 lht    634 tpos   21.247  -18173.97 -799195.44   -9092.00    w     inf   dir     0.03     0.13    -0.99 pol    0.195    0.971    0.136 BULK_REEMIT 
    AT_BOUNDARY      CONTINUE [  1337] slot -1 steps  4 lht    634 tpos   21.247  -18173.97 -799195.44   -9092.00    w     inf   dir     0.03     0.13    -0.99 pol    0.195    0.971    0.135 BULK_REEMIT 
    FILL_STATE       PASS     [  1337] slot  5 steps  5 lht 631702 tpos   21.247  -18173.97 -799195.44   -9092.00    w     inf   dir     0.03     0.13    -0.99 pol    0.195    0.971    0.135 BULK_REEMIT 
    TO_BOUNDARY      PASS     [  1337] slot -1 steps  5 lht 631702 tpos   21.335  -18173.48 -799193.00   -9110.00    w     inf   dir     0.03     0.13    -0.99 pol    0.195    0.971    0.135 BULK_REEMIT 
    AT_BOUNDARY      CONTINUE [  1337] slot -1 steps  5 lht 631702 tpos   21.335  -18173.48 -799193.00   -9110.00    w     inf   dir     0.03     0.13    -0.99 pol    0.196    0.972    0.133 BULK_REEMIT 
    FILL_STATE       PASS     [  1337] slot  6 steps  6 lht 632006 tpos   21.335  -18173.48 -799193.00   -9110.00    w     inf   dir     0.03     0.13    -0.99 pol    0.196    0.972    0.133 BULK_REEMIT 
    TO_BOUNDARY      PASS     [  1337] slot -1 steps  6 lht 632006 tpos   21.384  -18173.22 -799191.69   -9119.90    w     inf   dir     0.03     0.13    -0.99 pol    0.196    0.972    0.133 BULK_REEMIT 
    AT_BOUNDARY      CONTINUE [  1337] slot -1 steps  6 lht 632006 tpos   21.384  -18173.22 -799191.69   -9119.90    w     inf   dir     0.04     0.19    -0.98 pol    0.193    0.961    0.197 BULK_REEMIT 
    FILL_STATE       PASS     [  1337] slot  7 steps  7 lht 632304 tpos   21.384  -18173.22 -799191.69   -9119.90    w     inf   dir     0.04     0.19    -0.98 pol    0.193    0.961    0.197 BULK_REEMIT 
    TO_BOUNDARY      PASS     [  1337] slot -1 steps  7 lht 632304 tpos   21.385  -18173.22 -799191.69   -9119.95    w     inf   dir     0.04     0.19    -0.98 pol    0.193    0.961    0.197 BULK_REEMIT 
    AT_BOUNDARY      CONTINUE [  1337] slot -1 steps  7 lht 632304 tpos   21.385  -18173.22 -799191.69   -9119.95    w     inf   dir     0.04     0.19    -0.98 pol    0.193    0.961    0.197 BULK_REEMIT 
    FILL_STATE       PASS     [  1337] slot  8 steps  8 lht 632331 tpos   21.385  -18173.22 -799191.69   -9119.95    w     inf   dir     0.04     0.19    -0.98 pol    0.193    0.961    0.197 BULK_REEMIT 
    TO_BOUNDARY      BREAK    [  1337] slot -1 steps  8 lht     -1 tpos   21.385  -18173.22 -799191.69   -9119.95    w     inf   dir     0.04     0.19    -0.98 pol    0.193    0.961    0.197 BULK_REEMIT BULK_ABSORB 
    2014-07-02 17:15:23,323 env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:273 write_propagated /usr/local/env/tmp/1/propagated-0.npz 


NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.cc::

    577             else {
    578                 // reemission, the sample method need modification
    579                 G4double CIIvalue = G4UniformRand()*
    580                     ScintillationIntegral->GetMaxValue();
    581                 if (CIIvalue == 0.0) {
    582                     // return unchanged particle and no secondaries  
    583                     aParticleChange.SetNumberOfSecondaries(0);
    584                     return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    585                 }
    586                 sampledEnergy=
    587                     ScintillationIntegral->GetEnergy(CIIvalue);
    588                 if (verboseLevel>1) {
    589                     G4cout << "oldEnergy = " <<aTrack.GetKineticEnergy() << G4endl;
    590                     G4cout << "reemittedSampledEnergy = " << sampledEnergy
    591                            << "\nreemittedCIIvalue =        " << CIIvalue << G4endl;
    592                 }
    593             }
    594 
    595             // Generate random photon direction





Disappearing/Reappearing Photon 3126 : FIXED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* FIXED by modifying present_vbo to allow straddling to the last photon.

Disappearance is much less common now, but some cases remain::

    delta:1 blyth$ daephotonsanalyzer.sh propagated-0.npz 
    2014-07-01 12:49:19,354 env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:350 creating DAEPhotonsAnalyzer for propagated-0.npz 
    2014-07-01 12:49:19,355 env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:191 load propagated from propagated-0.npz 
    2014-07-01 12:49:19,379 env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:382 dropping into IPython.embed() try: z.<TAB> 
    ... 

    In [1]: z.p_flags[3126]
    Out[1]: 
    array([[         0,          0,          0,          0],
           [        32,          0,          0,          4],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [        34,          0,          0,         12],
           [        34, 1101913600, 1107947402,          0]], dtype=uint32)

    In [2]: STATUS_DONE
    Out[2]: 12

    In [3]: REFLECT_DIFFUSE
    Out[3]: 32

    In [4]: REFLECT_DIFFUSE|BULK_ABSORB
    Out[4]: 34

::

    In [1]: z.p_lht[3126]
    Out[1]: 
    array([[2382597,    3126,       1,       0],
           [2165175,    3126,       2,       1],
           [      0,       0,       0,       0],
           [      0,       0,       0,       0],
           [      0,       0,       0,       0],
           [      0,       0,       0,       0],
           [      0,       0,       0,       0],
           [      0,       0,       0,       0],
           [     -1,    3126,       2,       2],
           [     -1,    3126,       2,       2]], dtype=int32)




The photon is invisible between 31.411 and 34.4839.  Fail to staddle ?::

    In [6]: z.p_post[3126]
    Out[6]: 
    array([[ -19966.8516, -796813.3125,   -7034.7739,      21.7334],
           [ -22015.8867, -796247.3125,   -6789.8774,      31.411 ],
           [      0.    ,       0.    ,       0.    ,       0.    ],
           [      0.    ,       0.    ,       0.    ,       0.    ],
           [      0.    ,       0.    ,       0.    ,       0.    ],
           [      0.    ,       0.    ,       0.    ,       0.    ],
           [      0.    ,       0.    ,       0.    ,       0.    ],
           [      0.    ,       0.    ,       0.    ,       0.    ],
           [ -21424.3594, -796217.1875,   -6569.8042,      34.4839],
           [      0.    ,       0.    ,       0.    ,       0.    ]], dtype=float32)


::

    g4daeview.sh --with-chroma --load 1 --wipepropagate --debugkernel --debugphoton 3126

::

    FILL_STATE       START    [  3126] slot  0 steps  1 lht 2382597 tpos   21.733  -19966.85 -796813.31   -7034.77    w  383.00   dir    -0.96     0.26     0.11 pol   -0.284   -0.933   -0.220 
    TO_BOUNDARY      PASS     [  3126] slot -1 steps  1 lht 2382597 tpos   31.411  -22015.89 -796247.31   -6789.88    w  383.00   dir    -0.96     0.26     0.11 pol   -0.284   -0.933   -0.220 
    AT_SURFACE       CONTINUE [  3126] slot -1 steps  1 lht 2382597 tpos   31.411  -22015.89 -796247.31   -6789.88    w  383.00   dir     0.94     0.05     0.35 pol   -0.350    0.221    0.910 REFLECT_DIFFUSE 
    FILL_STATE       CONTINUE [  3126] slot  1 steps  2 lht 2165175 tpos   31.411  -22015.89 -796247.31   -6789.88    w  383.00   dir     0.94     0.05     0.35 pol   -0.350    0.221    0.910 REFLECT_DIFFUSE 
    TO_BOUNDARY      BREAK    [  3126] slot -1 steps  2 lht     -1 tpos   34.484  -21424.36 -796217.19   -6569.80    w  383.00   dir     0.94     0.05     0.35 pol   -0.350    0.221    0.910 REFLECT_DIFFUSE BULK_ABSORB 





Missing NO_HIT : FIXED
~~~~~~~~~~~~~~~~~~~~~~~~~

Formerly (before moved to max_slots-2 for final position, for truncation amelioration) 
had some appararently direct from the Geant4(muon) NO_HIT(grey) photons appearing outside AD
in line with muon direction at 20-30ns

* where did they go ?
* reverting to old way to study them, see that are slot-0 (visible in confetti-0)

* the reason is that the last_offset in present_vbo has to be changed to pick 
  up the new last slot rather than dynamically setting the last slot

::

    delta:1 blyth$ daephotonsanalyzer.sh propagated-0.npz 

    In [14]: no_hits = np.where( z.propagated['flags'][::-10,0] == 1 )[0]
    In [16]: no_hits
    Out[16]: 
    array([ 818,  846,  865,  890,  927,  949,  988, 1015, 1028, 1061, 1141,
           1158, 1160, 1196, 1248])


::

    In [31]: z.propagated['position_time'][::10][4164-no_hits]
    Out[31]: 
    array([[ -20837.0723, -795441.1875,   -7052.3433,      27.145 ],
           [ -20685.9727, -795674.1875,   -7053.2344,      26.2188],
           [ -20553.4551, -795878.5   ,   -7054.0117,      25.4065],
           [ -20486.6914, -796003.4375,   -7059.9165,      24.9435],
           [ -20346.3223, -796198.0625,   -7055.0967,      24.1361],
           [ -20244.8359, -796354.625 ,   -7055.6226,      23.5137],
           [ -20119.9609, -796547.25  ,   -7056.1987,      22.748 ],
           [ -19886.707 , -796628.6875,   -7042.4688,      22.1542],
           [ -19982.6934, -796758.5   ,   -7057.3345,      21.9085],
           [ -19897.7383, -796890.0625,   -7057.2769,      21.3854],
           [ -19671.6348, -797238.6875,   -7058.2666,      19.9992],
           [ -19638.5586, -797291.6875,   -7058.1128,      19.791 ],
           [ -19636.4805, -797296.1875,   -7056.1191,      19.7753],
           [ -19571.9023, -797392.5   ,   -7058.5796,      19.3877],
           [ -19457.2754, -797569.3125,   -7058.8467,      18.6849]], dtype=float32)


Dropouts : 91 long bouncers out of 4165 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed by saving STATUS_ENQUEUE in addition to STATUS_DONE

  * but the enqueing is not causing a re-propagate ?


::

    In [3]: z.last_flags
    Out[3]: 
    array([[ 65,   0,   0,  12],
           [  2,   0,   0,  12],
           [  2,   0,   0,  12],
           ..., 
           [578,   0,   0,  12],
           [514,   0,   0,  12],
           [514,   0,   0,  12]], dtype=uint32)

    In [4]: z.last_flags[:,3]
    Out[4]: array([12, 12, 12, ..., 12, 12, 12], dtype=uint32)

    In [5]: np.where( z.last_flags[:,3] != 12 )
    Out[5]: 
    (array([ 111,  117,  208,  302,  415,  572,  660,  701,  720,  765,  769,
            773,  809,  842,  952,  962, 1072, 1078, 1118, 1178, 1305, 1519,
           1585, 1592, 1608, 1615, 1650, 1709, 1753, 1856, 1873, 1876, 1880,
           1949, 1997, 2003, 2012, 2053, 2106, 2186, 2191, 2216, 2236, 2288,
           2300, 2309, 2377, 2422, 2439, 2445, 2455, 2547, 2555, 2623, 2666,
           2669, 2791, 2860, 3017, 3024, 3158, 3192, 3212, 3244, 3288, 3293,
           3332, 3371, 3399, 3453, 3468, 3496, 3521, 3545, 3559, 3688, 3690,
           3811, 3831, 3835, 3890, 3938, 3940, 3950, 3970, 4033, 4041, 4062,
           4068, 4112, 4155]),)

    In [6]: 

    In [6]: np.where( z.last_post[:,3] < 0.001 )
    Out[6]: 
    (array([ 111,  117,  208,  302,  415,  572,  660,  701,  720,  765,  769,
            773,  809,  842,  952,  962, 1072, 1078, 1118, 1178, 1305, 1519,
           1585, 1592, 1608, 1615, 1650, 1709, 1753, 1856, 1873, 1876, 1880,
           1949, 1997, 2003, 2012, 2053, 2106, 2186, 2191, 2216, 2236, 2288,
           2300, 2309, 2377, 2422, 2439, 2445, 2455, 2547, 2555, 2623, 2666,
           2669, 2791, 2860, 3017, 3024, 3158, 3192, 3212, 3244, 3288, 3293,
           3332, 3371, 3399, 3453, 3468, 3496, 3521, 3545, 3559, 3688, 3690,
           3811, 3831, 3835, 3890, 3938, 3940, 3950, 3970, 4033, 4041, 4062,
           4068, 4112, 4155]),)


Hmm 91 not filled::

    In [7]: not_done = np.where( z.last_flags[:,3] != 12 )[0]

    In [11]: len(not_done)
    Out[11]: 91

    In [8]: z.last_flags[not_done]
    Out[8]: 
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],


After fix to save STATUE_ENQUEUE, they are filled but not done::

    In [6]: z.last_flags[not_done]
    Out[6]: 
    array([[ 80,   0,   0,  11],
           [ 64,   0,   0,  11],
           [ 64,   0,   0,  11],
           [ 64,   0,   0,  11],
           [ 64,   0,   0,  11],
           [ 64,   0,   0,  11],
           [ 64,   0,   0,  11],
           [ 64,   0,   0,  11],
           [ 64,   0,   0,  11],
           [ 80,   0,   0,  11],
           [ 64,   0,   0,  11],
           [ 96,   0,   0,  11],
           [ 64,   0,   0,  11],
           [608,   0,   0,  11],
           [ 80,   0,   0,  11],
           [ 64,   0,   0,  11],
           [ 64,   0,   0,  11],
           [112,   0,   0,  11],
           [576,   0,   0,  11],
           [ 96,   0,   0,  11],


Just not saved::

    In [17]: z.propagated['position_time'][1110:1120]
    Out[17]: 
    array([[ -17035.1074, -801313.25  ,   -7065.2979,       4.5006],
           [ -17035.1074, -801313.25  ,   -7065.2979,       4.5006],
           [ -16996.2812, -801357.9375,   -7056.0366,       4.7988],
           [ -16984.5684, -801371.4375,   -7053.2563,       4.8891],
           [ -16815.4824, -801566.    ,   -7012.2808,       6.1676],
           [ -16981.1055, -801368.5   ,   -6971.5107,       7.4459],
           [ -16992.5918, -801354.8125,   -6968.7422,       7.5364],
           [ -17268.082 , -801026.3125,   -6902.0137,       9.695 ],
           [      0.    ,       0.    ,       0.    ,       0.    ],
           [      0.    ,       0.    ,       0.    ,       0.    ]], dtype=float32)


::

    In [20]: z.propagated['flags'][1110:1120]
    Out[20]: 
    array([[         0,          0,          0,          0],
           [         0,          0,          0,          4],
           [         0,          0,          0,          4],
           [         0,          0,          0,          4],
           [        64,          0,          0,          4],
           [        64,          0,          0,          4],
           [        64,          0,          0,          4],
           [        64,          0,          0,          4],           64 REFLECT_SPECULAR, 4 STATUS_FILL_STATE
           [         0,          0,          0,          0],
           [        80, 1083180326, 1128190499,          0]], dtype=uint32)


::

    In [21]: z.propagated['flags'][1170:1180]
    Out[21]: 
    array([[         0,          0,          0,          0],
           [         0,          0,          0,          4],
           [         0,          0,          0,          4],
           [         0,          0,          0,          4],
           [        64,          0,          0,          4],
           [        64,          0,          0,          4],
           [        64,          0,          0,          4],
           [        64,          0,          0,          4],
           [         0,          0,          0,          0],
           [        64, 1083301678, 1123152860,          0]], dtype=uint32)


Maybe its an 8 slot bug, nope its due to 100 step truncation, STATUS_ENQUEUE was not being written:: 

    In [22]: z.propagated['last_hit_triangle'][1170:1180]
    Out[22]: 
    array([[    -1,      0,      0,      0],
           [   576,    117,      1,      1],
           [   288,    117,      2,      2],
           [616675,    117,      3,      3],
           [   288,    117,      4,      4],
           [   576,    117,      5,      5],
           [   909,    117,      6,      6],
           [  1197,    117,      7,      7],
           [     0,      0,      0,      0],
           [625654,    117,    100,    101]], dtype=int32)

::

    In [18]: z.propagated['position_time'][1120:1130]
    Out[18]: 
    array([[ -17015.4941, -801317.4375,   -7084.8896,       4.505 ],
           [ -17015.4941, -801317.4375,   -7084.8896,       4.505 ],
           [ -17170.748 , -800957.0625,   -6044.4136,      10.0594],
           [ -17174.9473, -800947.625 ,   -6018.001 ,      10.2017],
           [ -17242.541 , -800790.6875,   -5565.    ,      12.6317],
           [ -17242.5488, -800790.6875,   -5564.9502,      12.632 ],
           [ -17243.6074, -800788.25  ,   -5557.8618,      12.6698],
           [ -17245.8926, -800782.9375,   -5542.4385,      12.7525],
           [ -17328.8535, -800590.375 ,   -4987.998 ,      15.7173],
           [      0.    ,       0.    ,       0.    ,       0.    ]], dtype=float32)



::

    In [11]: z.last_post[:,3].min()
    Out[11]: 2.3316712

    In [12]: z.last_post[:,3].max()
    Out[12]: 1371.0537

    In [13]: z.time_range
    Out[13]: [0.0, 1371.0537]

    In [14]: z.t0
    Out[14]: 
    array([    1.4179,     2.3273,     2.3649, ...,   863.4072,   865.5709,
            1356.45  ], dtype=float32)

    In [15]: z.t0.min()
    Out[15]: 1.4178798

    In [16]: z.tf.min()
    Out[16]: 2.3316712






Fixed Issues
-------------

Both the below were caused by interpolation bug 

#. photon visualization disappearance, even with eg `--mode 7` to exclude truncated
#. non-sensical discontinuities in propagation history animation  


Repeatability/Seeding Doubts
------------------------------

Seed values are controlled by `--seed x` which now defaults to 0 (formerly None which corresponds to 
a time and process id based seed).

Repeatability is checked using `--debugpropagation` option, now on by default.
The check in `DAEPhotonsAnalyzer` is performed on writing `propagated-<seed>.npz` when
a prior file exists.


Techniques
------------

daephotonsanalyzer.sh
~~~~~~~~~~~~~~~~~~~~~~~~

Use `--debugpropagate` to write files `propagated-<seed>.npz` into the directory corresponding to event path.
This is done after performing propagations, which happen as event files are loaded  eg::

    g4daeview.sh --with-chroma --load 1 --debugpropagate

These files contain numpy arrays of the VBO content.
Such files can be interactively examined using `daephotonsanalyzer.sh`::

    delta:~ blyth$ daephotonsanalyzer.sh propagated-0.npz 
    2014-06-27 18:14:09,645 env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:236 creating DAEPhotonsAnalyzer for propagated-0.npz 
    2014-06-27 18:14:09,670 env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:241 dropping into IPython.embed() try: z.<TAB> 
    ...

    In [1]: z.flags
    Out[1]: array([ 65,   2,   2, ..., 578, 514, 514], dtype=uint32)

    In [2]: len(z.flags)
    Out[2]: 4165

    In [3]: len(z.propagated)
    Out[3]: 41650

    In [4]: a = z.propagated['position_time']

    In [9]: a[60:70,:]   # with max_slots=10 position_time for photon_id = 6 
    Out[9]: 
    array([[ -16823.5898, -801640.625 ,   -7065.897 ,       2.5105],
           [ -16901.7969, -801623.9375,   -7041.4619,       2.9237],
           [ -17071.3887, -801951.4375,   -6928.5552,       4.83  ],
           [ -17469.5137, -801868.0625,   -6804.0322,       6.9324],
           [ -17962.4277, -802183.5625,   -6624.877 ,       9.9572],
           [ -18238.0645, -801937.    ,   -6511.6592,      11.8687],
           [ -18533.707 , -802130.625 ,   -6404.1758,      13.6942],
           [ -18308.5176, -801930.    ,   -6764.2158,      16.0154],
           [ -18306.3887, -801928.    ,   -6767.6338,      16.0304],
           [      0.    ,       0.    ,       0.    ,       0.    ]], dtype=float32)



truncation
~~~~~~~~~~~~

VBO slots are restricted via `max_slots` (eg 10) which is often less than `max_steps` (eg 100). But the tail flags 
written in 



debugphoton
~~~~~~~~~~~~~

Using `--debugkernel --debugphoton 6` dumps the steps of the propagation for photon_id 6, note that the positions/times match the above read from VBO::

    delta:~ blyth$ g4daeview.sh --with-chroma --load 1 --debugkernel --debugphoton 6 --pid 6 


::

    2014-06-27 18:23:50,079 env.geant4.geometry.collada.g4daeview.daechromacontext:59  setup_rng_states using seed 0 
    FILL_STATE       START    [     6] slot  0 steps  1 lht 621543 tpos    2.510  -16823.59 -801640.62   -7065.90    w  383.88   dir    -0.94     0.20     0.29 pol   -0.121   -0.956    0.266 
    TO_BOUNDARY      PASS     [     6] slot -1 steps  1 lht 621543 tpos    2.924  -16901.80 -801623.94   -7041.46    w  383.88   dir    -0.94     0.20     0.29 pol   -0.121   -0.956    0.266 
    AT_SURFACE       CONTINUE [     6] slot -1 steps  1 lht 621543 tpos    2.924  -16901.80 -801623.94   -7041.46    w  383.88   dir    -0.44    -0.85     0.29 pol   -0.121   -0.956    0.266 REFLECT_SPECULAR 
    FILL_STATE       CONTINUE [     6] slot  1 steps  2 lht    214 tpos    2.924  -16901.80 -801623.94   -7041.46    w  383.88   dir    -0.44    -0.85     0.29 pol   -0.121   -0.956    0.266 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps  2 lht    214 tpos    4.830  -17071.39 -801951.44   -6928.56    w  383.88   dir    -0.44    -0.85     0.29 pol   -0.121   -0.956    0.266 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [     6] slot -1 steps  2 lht    214 tpos    4.830  -17071.39 -801951.44   -6928.56    w  383.88   dir    -0.94     0.20     0.29 pol    0.138    0.968   -0.208 REFLECT_SPECULAR 
    FILL_STATE       PASS     [     6] slot  2 steps  3 lht 621451 tpos    4.830  -17071.39 -801951.44   -6928.56    w  383.88   dir    -0.94     0.20     0.29 pol    0.138    0.968   -0.208 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps  3 lht 621451 tpos    6.932  -17469.51 -801868.06   -6804.03    w  383.88   dir    -0.94     0.20     0.29 pol    0.138    0.968   -0.208 REFLECT_SPECULAR 
    AT_SURFACE       CONTINUE [     6] slot -1 steps  3 lht 621451 tpos    6.932  -17469.51 -801868.06   -6804.03    w  383.88   dir    -0.81    -0.52     0.29 pol    0.138    0.968   -0.208 REFLECT_SPECULAR 
    FILL_STATE       CONTINUE [     6] slot  3 steps  4 lht    211 tpos    6.932  -17469.51 -801868.06   -6804.03    w  383.88   dir    -0.81    -0.52     0.29 pol    0.138    0.968   -0.208 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps  4 lht    211 tpos    9.957  -17962.43 -802183.56   -6624.88    w  383.88   dir    -0.81    -0.52     0.29 pol    0.138    0.968   -0.208 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [     6] slot -1 steps  4 lht    211 tpos    9.957  -17962.43 -802183.56   -6624.88    w  383.88   dir    -0.71     0.64     0.29 pol    0.603    0.770   -0.208 REFLECT_SPECULAR 
    FILL_STATE       PASS     [     6] slot  4 steps  5 lht 621031 tpos    9.957  -17962.43 -802183.56   -6624.88    w  383.88   dir    -0.71     0.64     0.29 pol    0.603    0.770   -0.208 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps  5 lht 621031 tpos   11.869  -18238.06 -801937.00   -6511.66    w  383.88   dir    -0.71     0.64     0.29 pol    0.603    0.770   -0.208 REFLECT_SPECULAR 
    AT_SURFACE       CONTINUE [     6] slot -1 steps  5 lht 621031 tpos   11.869  -18238.06 -801937.00   -6511.66    w  383.88   dir    -0.80    -0.52     0.29 pol    0.603    0.770   -0.208 REFLECT_SPECULAR 
    FILL_STATE       CONTINUE [     6] slot  5 steps  6 lht    210 tpos   11.869  -18238.06 -801937.00   -6511.66    w  383.88   dir    -0.80    -0.52     0.29 pol    0.603    0.770   -0.208 REFLECT_SPECULAR 
    TO_BOUNDARY      CONTINUE [     6] slot -1 steps  6 lht     -1 tpos   13.694  -18533.71 -802130.62   -6404.18    w  383.88   dir     0.48     0.43    -0.77 pol    0.565    0.817    0.118 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    FILL_STATE       CONTINUE [     6] slot  6 steps  7 lht 370007 tpos   13.694  -18533.71 -802130.62   -6404.18    w  383.88   dir     0.48     0.43    -0.77 pol    0.565    0.817    0.118 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps  7 lht 370007 tpos   16.015  -18308.52 -801930.00   -6764.22    w  383.88   dir     0.48     0.43    -0.77 pol    0.565    0.817    0.118 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [     6] slot -1 steps  7 lht 370007 tpos   16.015  -18308.52 -801930.00   -6764.22    w  383.88   dir     0.47     0.45    -0.76 pol   -0.303    0.893    0.334 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    FILL_STATE       PASS     [     6] slot  7 steps  8 lht 372085 tpos   16.015  -18308.52 -801930.00   -6764.22    w  383.88   dir     0.47     0.45    -0.76 pol   -0.303    0.893    0.334 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps  8 lht 372085 tpos   16.030  -18306.39 -801928.00   -6767.63    w  383.88   dir     0.47     0.45    -0.76 pol   -0.303    0.893    0.334 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [     6] slot -1 steps  8 lht 372085 tpos   16.030  -18306.39 -801928.00   -6767.63    w  383.88   dir     0.55     0.08    -0.83 pol   -0.094    0.995    0.037 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    FILL_STATE       PASS     [     6] slot  8 steps  9 lht 372228 tpos   16.030  -18306.39 -801928.00   -6767.63    w  383.88   dir     0.55     0.08    -0.83 pol   -0.094    0.995    0.037 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps  9 lht 372228 tpos   16.031  -18306.35 -801928.00   -6767.69    w  383.88   dir     0.55     0.08    -0.83 pol   -0.094    0.995    0.037 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [     6] slot -1 steps  9 lht 372228 tpos   16.031  -18306.35 -801928.00   -6767.69    w  383.88   dir     0.47     0.44    -0.76 pol   -0.288    0.894    0.342 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    FILL_STATE       PASS     [     6] slot  9 steps 10 lht 370727 tpos   16.031  -18306.35 -801928.00   -6767.69    w  383.88   dir     0.47     0.44    -0.76 pol   -0.288    0.894    0.342 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps 10 lht 370727 tpos   16.031  -18306.28 -801927.94   -6767.80    w  383.88   dir     0.47     0.44    -0.76 pol   -0.288    0.894    0.342 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [     6] slot -1 steps 10 lht 370727 tpos   16.031  -18306.28 -801927.94   -6767.80    w  383.88   dir    -0.18     0.97     0.15 pol   -0.530   -0.229    0.816 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    FILL_STATE       PASS     [     6] slot 10 steps 11 lht 372228 tpos   16.031  -18306.28 -801927.94   -6767.80    w  383.88   dir    -0.18     0.97     0.15 pol   -0.530   -0.229    0.816 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps 11 lht 372228 tpos   16.032  -18306.30 -801927.81   -6767.78    w  383.88   dir    -0.18     0.97     0.15 pol   -0.530   -0.229    0.816 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [     6] slot -1 steps 11 lht 372228 tpos   16.032  -18306.30 -801927.81   -6767.78    w  383.88   dir    -0.33     0.86     0.38 pol    0.441    0.497   -0.747 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    FILL_STATE       PASS     [     6] slot 11 steps 12 lht 372085 tpos   16.032  -18306.30 -801927.81   -6767.78    w  383.88   dir    -0.33     0.86     0.38 pol    0.441    0.497   -0.747 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps 12 lht 372085 tpos   16.032  -18306.32 -801927.75   -6767.76    w  383.88   dir    -0.33     0.86     0.38 pol    0.441    0.497   -0.747 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [     6] slot -1 steps 12 lht 372085 tpos   16.032  -18306.32 -801927.75   -6767.76    w  383.88   dir    -0.19     0.97     0.15 pol    0.517    0.228   -0.825 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    FILL_STATE       PASS     [     6] slot 12 steps 13 lht 370007 tpos   16.032  -18306.32 -801927.75   -6767.76    w  383.88   dir    -0.19     0.97     0.15 pol    0.517    0.228   -0.825 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps 13 lht 370007 tpos   16.054  -18307.16 -801923.38   -6767.07    w  383.88   dir    -0.19     0.97     0.15 pol    0.517    0.228   -0.825 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [     6] slot -1 steps 13 lht 370007 tpos   16.054  -18307.16 -801923.38   -6767.07    w  383.88   dir    -0.20     0.97     0.17 pol    0.528    0.249   -0.812 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    FILL_STATE       PASS     [     6] slot 13 steps 14 lht    330 tpos   16.054  -18307.16 -801923.38   -6767.07    w  383.88   dir    -0.20     0.97     0.17 pol    0.528    0.249   -0.812 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps 14 lht    330 tpos   17.370  -18359.22 -801666.25   -6722.09    w  383.88   dir    -0.20     0.97     0.17 pol    0.528    0.249   -0.812 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [     6] slot -1 steps 14 lht    330 tpos   17.370  -18359.22 -801666.25   -6722.09    w  383.88   dir    -0.19     0.97     0.17 pol   -0.829   -0.248    0.500 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    FILL_STATE       PASS     [     6] slot 14 steps 15 lht    618 tpos   17.370  -18359.22 -801666.25   -6722.09    w  383.88   dir    -0.19     0.97     0.17 pol   -0.829   -0.248    0.500 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps 15 lht    618 tpos   17.465  -18362.79 -801648.06   -6718.98    w  383.88   dir    -0.19     0.97     0.17 pol   -0.829   -0.248    0.500 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [     6] slot -1 steps 15 lht    618 tpos   17.465  -18362.79 -801648.06   -6718.98    w  383.88   dir    -0.19     0.97     0.17 pol   -0.829   -0.250    0.500 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    FILL_STATE       PASS     [     6] slot 15 steps 16 lht    949 tpos   17.465  -18362.79 -801648.06   -6718.98    w  383.88   dir    -0.19     0.97     0.17 pol   -0.829   -0.250    0.500 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    TO_BOUNDARY      CONTINUE [     6] slot -1 steps 16 lht    949 tpos   17.574  -18366.97 -801626.94   -6715.35    w     inf   dir     0.63     0.69     0.36 pol    0.671   -0.716    0.190 RAYLEIGH_SCATTER REFLECT_SPECULAR BULK_REEMIT 
    FILL_STATE       CONTINUE [     6] slot 16 steps 17 lht    951 tpos   17.574  -18366.97 -801626.94   -6715.35    w     inf   dir     0.63     0.69     0.36 pol    0.671   -0.716    0.190 RAYLEIGH_SCATTER REFLECT_SPECULAR BULK_REEMIT 
    TO_BOUNDARY      BREAK    [     6] slot -1 steps 17 lht     -1 tpos   17.671  -18354.58 -801613.44   -6708.33    w     inf   dir     0.63     0.69     0.36 pol    0.671   -0.716    0.190 RAYLEIGH_SCATTER REFLECT_SPECULAR BULK_REEMIT BULK_ABSORB 



history selection
~~~~~~~~~~~~~~~~~~

::

   udp.py --bits RAYLEIGH_SCATTER,REFLECT_SPECULAR,BULK_REEMIT,BULK_ABSORB --cohort 0,10,-1   
   # born within first 10ns that undergo all those processes


Restrict to photons with n-step histories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Avoid uncertainties from truncation effects by keeping n below max_slots-1.::

   --mode 7 --max-slots 10

Restrict birth time range, allowing to examine cohorts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Otherwise photons keep springing into life.::

   --cohort 0,10,-1   # ns 

   udp.py --cohort 2,3,-1 --style spagetti   

   udp.py --cohort 2.5,2.6,1 --style spagetti   # selects a 6 bouncer, between the PMTs

      #
      # interactive changing cohort in spagetti mode, allows to select single photons 
      # flags/history menu selection indicates it to be REFLECT_SPECULAR,BULK_ABSORB
      #
      # animation fails to visualize it ? current psave approach missing specular bouncers ?



cohort mode, third value in cohort string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Positive cohort mode dumps photon_id from the kernel::

    udp.py --cohort 0,10,1

   

::


    I: photon_id 6 tail_birth 2.510489 tail_death 17.670887  cohort 0.000000 10.000000 1.000000 
    I: photon_id 279 tail_birth 5.828637 tail_death 83.182884  cohort 0.000000 10.000000 1.000000 
    I: photon_id 541 tail_birth 7.159081 tail_death 45.278973  cohort 0.000000 10.000000 1.000000 
    I: photon_id 412 tail_birth 6.597654 tail_death 92.039955  cohort 0.000000 10.000000 1.000000 
    I: photon_id 157 tail_birth 4.990300 tail_death 30.397882  cohort 0.000000 10.000000 1.000000 
    I: photon_id 898 tail_birth 9.194763 tail_death 29.307714  cohort 0.000000 10.000000 1.000000 
    I: photon_id 916 tail_birth 9.298509 tail_death 35.309608  cohort 0.000000 10.000000 1.000000 
    I: photon_id 920 tail_birth 9.309920 tail_death 102.759193  cohort 0.000000 10.000000 1.000000 
    I: photon_id 816 tail_birth 8.671006 tail_death 33.654274  cohort 0.000000 10.000000 1.000000 
    I: photon_id 938 tail_birth 9.390456 tail_death 25.577848  cohort 0.000000 10.000000 1.000000 
    I: photon_id 949 tail_birth 9.440248 tail_death 74.828758  cohort 0.000000 10.000000 1.000000 
    I: photon_id 738 tail_birth 8.296719 tail_death 75.682594  cohort 0.000000 10.000000 1.000000 
    I: photon_id 766 tail_birth 8.447924 tail_death 45.957516  cohort 0.000000 10.000000 1.000000 
    I: photon_id 731 tail_birth 8.250953 tail_death 38.883736  cohort 0.000000 10.000000 1.000000 


::

    udp.py --cohort 2.51,2.52,1.   # down to single photon_id 6 

::

    udp.py --mode 0 --style confetti

    ## despite animation not working, using time reveal --mode 0 and confetti style allows to see the direction, bounce times



photon highlighting
~~~~~~~~~~~~~~~~~~~~~

Highlight a single photon by increasing presentation point size::

    udp.py --pid 938



style playoff
~~~~~~~~~~~~~~~

::

    udp.py --style confetti,spagetti,movie-extra --cohort 0,10,-1 --pid 541 --bits RAYLEIGH_SCATTER,REFLECT_SPECULAR,BULK_REEMIT,BULK_ABSORB


       ## bizarre off-the-cliff and jump around as go beyond 19ns in pid 541
       ## THIS WAS THE SMOKING GUN THAT REVEALED THE INTERPOLATION BUG
   







