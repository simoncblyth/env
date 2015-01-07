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

Cerenkov Refs
---------------

* http://large.stanford.edu/courses/2014/ph241/alaeian2/
* http://math.ucr.edu/home/baez/physics/Relativity/SpeedOfLight/cherenkov.html
* https://thespectrumofriemannium.wordpress.com/tag/tamm-frank-formula/

Rejection Sampling
-------------------

* http://www.youtube.com/watch?v=wRdjeCtc8d0&feature=youtube_gdata_player


Check Generation Consistency
-----------------------------

Three files to check for each process (Scintillation and Cerenkov):

* CERENKOV, the step file
* GOPCERENKOV, g4 optical photons generated from those steps
* OPCERENKOV, chroma optical photons generated from the steps on GPU

* SCINTILLATION, the step file
* GOPSCINTILLATION, g4 optical photons generated from those steps
* OPSCINTILLATION, chroma optical photons generated from the steps on GPU


No longer need to manually copy any files, 
the GPU generated photons are now "sidesaved" under 
types opcerenkov and opscintillation
and the Geant4 equivalents are "onlycopied" from `DsChromaEventAction::EndOfEventAction` 
via metadata control such as `ctrl:noreturn:1` and `ctrl:onlycopy:1` applied to processing of all species. 


Revisit With New Type Names
-----------------------------

::

    In [1]: cf('wavelength', tag=1, typs='gopcerenkov opcerenkovgen')  
     
     ## improved by shifting low wavelength to match G4 change 60:801:20 to 80:801:20

    In [3]: cf('3xyzw', tag=1, typs='gopcerenkov opcerenkovgen', legend=False)


    In [1]: cf('3xyzw', tag=1, typs='gopscintillation opscintillationgen', legend=False)

    In [2]: cf('wavelength', tag=1, typs='gopscintillation opscintillationgen', log=True)






Count Checking
----------------

* Cerenkov counts are matching only as ApplyWaterQE killing (via continue in photon loop) is still disabled


Check counts from ipython::


    In [2]: genconsistency(1)
    INFO:env.g4dae:g4c : GOPCERENKOV    (612841, 4, 4) 
    INFO:env.g4dae:chc :  OPCERENKOV    (612841, 4, 4) 
    INFO:env.g4dae:stc :    CERENKOV    (7836, 6, 4) ==> N 612841 
    INFO:env.g4dae:g4s : GOPSCINTILLATION    (2817543, 4, 4) 
    INFO:env.g4dae:chs :  OPSCINTILLATION    (2817543, 4, 4) 
    INFO:env.g4dae:sts :    SCINTILLATION    (13898, 6, 4) ==> N 2817543 


Cerenkov
----------

tee up the arrays
~~~~~~~~~~~~~~~~~~~

::

    In [1]: chc, g4c, tst = NPY.mget(1,"opcerenkov","gopcerenkov","test")


wavelength
~~~~~~~~~~~~

Initially had very different wavelength, with chroma flat
due to use of uniform wavelength draw, fixed by moving to 
uniform in reciprocal of wavelenth::

    In [1]: cf('wavelength', tag=1, typs="gopcerenkov opcerenkov test")


Create some new "test" arrays with modified standard wavelengths::

    g4daechroma.sh --wavelengths 80:801:20

    npysend.sh -t1 -icerenkov -otest 

* need to split test into testcerenkov and testscintillation, otherwise write stomping potential ?


G4 distrib has step at 200nm, an artifact of RINDEX edge of quoted range::


    In [105]: chroma_refractive_index(cg)
    [ 0]        LiquidScintillator    (18, 2)     wl   79.99 :  799.90          1.454 :      1.793 
    [ 3]                 GdDopedLS    (18, 2)     wl   79.99 :  799.90          1.454 :      1.793 
    [ 5]                   Acrylic    (18, 2)     wl   79.99 :  799.90          1.462 :      1.793 
    [10]                MineralOil    (18, 2)     wl   79.99 :  799.90          1.434 :      1.759 
    [24]                  IwsWater    (34, 2)     wl  199.97 :  799.90          1.333 :      1.390 
    [27]                  OwsWater    (34, 2)     wl  199.97 :  799.90          1.333 :      1.390 
    [28]                 DeadWater    (34, 2)     wl  199.97 :  799.90          1.333 :      1.390 

::

    In [6]: plot_refractive_index()

    In [7]: plot_refractive_index(standard=True)

    In [130]: for i in np.unique(im):print "%2d : %5d : %s " % ( i, bc[i], cg.unique_materials[i].name[17:-9] )

     0 :  1133 : LiquidScintillator 
     3 :  4220 : GdDopedLS 
     5 :    88 : Acrylic 
    10 :  1046 : MineralOil 
    24 :   791 : IwsWater 
    27 :   530 : OwsWater 
    28 :    28 : DeadWater 



time
~~~~~~

::


    In [9]: cf('time', tag=1, typs="gopcerenkov opcerenkov test")



* with ApplyWaterQE killing enabled

  * very closely matched up to 18ns, beyond that much less g4


* without ApplyWaterQE

  * almost perfect match



xyz pos,dir,pol
~~~~~~~~~~~~~~~~~~

::

    In [9]: cf('3xyz', g4c, chc)


* with ApplyWaterQE killing enabled

  * pos : clear spatial discrepancy, less at extremes of x and y

* without ApplyWaterQE 

  * pos : almost perfect 
  * dir : vgood agreement, except that chroma spikes are more spiky 
  * pol : same as dir with chroma spikes more spiky 


investigate cerenkov wavelength
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`chroma/chroma/cuda/cerenkov.h`::

    202 __device__ void
    203 generate_cerenkov_photon(Photon& p, CerenkovStep& cs, curandState &rng)
    204 {
    205      float cosTheta ;
    206      float sin2Theta ;
    207      float wavelength ;
    208      float sampledRI ;
    209 
    210      // 
    211      //  sampling to get wavelength and cone angle 
    212      //
    213      // pick random wavelength inside the range, 
    214      // lookup refractive index
    215      // calculate cosTheta and sinTheta for the refractive index
    216      // 
    217      do {
    218 
    219         wavelength = sample_value(cs.material, curand_uniform(&rng));
    220 
    221         sampledRI = interp_property(cs.material, wavelength, cs.material->refractive_index);
    222 
    223         cosTheta = cs.BetaInverse / sampledRI;
    224 
    225         sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
    226 
    227       } while ( curand_uniform(&rng)*cs.maxSin2 > sin2Theta);
    228 
    229 
    230       p.wavelength = wavelength ;
    231 


::

    296        G4double Pmin = Rindex->GetMinPhotonEnergy();
    297        G4double Pmax = Rindex->GetMaxPhotonEnergy();
    298        G4double dp = Pmax - Pmin;



    405     for (G4int i = 0; i < NumPhotons; i++) {
    406       // Determine photon energy
    407       G4double rand=0;
    408       G4double sampledEnergy=0, sampledRI=0;
    409       G4double cosTheta=0, sin2Theta=0;
    410 
    411       // sample an energy
    412       do {
    413         rand = G4UniformRand();
    414         sampledEnergy = Pmin + rand * dp;
    415         sampledRI = Rindex->GetProperty(sampledEnergy);
    416         cosTheta = BetaInverse / sampledRI;
    417 
    418         sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
    419         rand = G4UniformRand();
    420 
    421       } while (rand*maxSin2 > sin2Theta);
    422 





::

    In [48]: cls.refractive_index
    Out[48]: 
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

    In [49]: cls.name
    Out[49]: '__dd__Materials__LiquidScintillator0xc2308d0'

    In [50]: ri = cls.refractive_index

    In [51]: plt.scatter(ri[:,0],ri[:,1])
    Out[51]: <matplotlib.collections.PathCollection at 0x125b76a90>

    In [52]: plt.show()












::

    In [53]: _stc = stc(1)

    In [56]: BetaInverse = _stc[:,4,0]   
    Out[56]: array([ 1.,  1.,  1., ...,  1.,  1.,  1.], dtype=float32)

    In [57]: BetaInverse.min()
    Out[57]: 1.0000062

    In [58]: BetaInverse.max()
    Out[58]: 1.4531251

    In [64]: plt.hist(BetaInverse, bins=100,log=True)    # mainly 1.000  with small tail out to 1.45



::

    In [107]: _stc[:,0].view(np.int32)
    Out[107]: 
    array([[   -1,     1,    24,    80],
           [   -2,     1,    24,   108],
           [   -3,     1,    24,    77],
           ..., 
           [-7834,     1,    28,    91],
           [-7835,     1,    28,    83],
           [-7836,     1,    28,    48]], dtype=int32)

    In [108]: _stc[:,0,2].view(np.int32)
    Out[108]: array([24, 24, 24, ..., 28, 28, 28], dtype=int32)

    In [110]: im
    Out[110]: array([24, 24, 24, ..., 28, 28, 28], dtype=int32)

    In [111]: np.unique(im)
    Out[111]: array([ 0,  3,  5, 10, 24, 27, 28], dtype=int32)

    In [129]: bc = np.bincount(im)

    In [130]: for i in np.unique(im):print "%2d : %5d : %s " % ( i, bc[i], cg.unique_materials[i].name[17:-9] )

     0 :  1133 : LiquidScintillator 
     3 :  4220 : GdDopedLS 
     5 :    88 : Acrylic 
    10 :  1046 : MineralOil 
    24 :   791 : IwsWater 
    27 :   530 : OwsWater 
    28 :    28 : DeadWater 





`G4DAEChroma/G4DAECerenkovStep.hh`::

     13     enum {
     14 
     15        _Id,                      //  0
     16        _ParentID,
     17        _Material,
     18        _NumPhotons,
     19 
     20        _x0_x,                    //  1
     21        _x0_y,
     22        _x0_z,
     23        _t0,
     24 
     25        _DeltaPosition_x,         // 2
     26        _DeltaPosition_y,
     27        _DeltaPosition_z,
     28        _step_length,
     29
     30        _code,                    // 3
     31        _charge,
     32        _weight,
     33        _MeanVelocity,
     34 
     35        _BetaInverse,             //  4
     36        _Pmin,
     37        _Pmax,
     38        _maxCos,
     39 
     40        _maxSin2,                 // 5
     41        _MeanNumberOfPhotons1,
     42        _MeanNumberOfPhotons2,
     43        _BialkaliMaterialIndex,



::

    In [73]: maxSin2 = _stc[:,5,0]

    In [76]: plt.hist(maxSin2, bins=100, log=True)   ## mostly flat with few spikes at high end

    In [82]: maxSin2.min()
    Out[82]: 0.00065323891

    In [83]: maxSin2.max()
    Out[83]: 0.53214556



BialkaliMaterialIndex::

    n [69]: _stc[:,5,3].view(np.int32).min()
    Out[69]: 7

    In [70]: _stc[:,5,3].view(np.int32).max()
    Out[70]: 7

    In [71]: cg.unique_materials[7]
    Out[71]: <chroma.geometry.Material at 0x125a9a950>

    In [72]: cg.unique_materials[7].name
    Out[72]: '__dd__Materials__Bialkali0xc2f2428'



cerenkov review
~~~~~~~~~~~~~~~~~


TODO: settle on standard wavelenth range to match G4 better::


    In [4]: cf('3xyzw', tag=1, typs='gopcerenkov opcerenkov', legend=False, log=True)




Scintillation
--------------

tee up the arrays
~~~~~~~~~~~~~~~~~~~

::

    In [8]: g4s, chs = NPY.mget(1, "gopscintillation opscintillation")


wavelength
~~~~~~~~~~~

::

    In [6]: cf('wavelength', typs="gopscintillation opscintillation",tag=1, log=True, range=(100,900) )   ## hmm clear chroma cut at 600nm ???


Scintillation wavelength, chroma distrib is faithfully representing 
a "histogram" stepping shape with "bins" of about 25nm.  
Looks like a problem of mismatched histogram ranges in the chroma
sampling and the input histogram

* not quite, just a case of coarse interpolation

`chroma/chroma/geometry.py`::

     25 # all material/surface properties are interpolated at these
     26 # wavelengths when they are sent to the gpu
     27 standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)
     28 


::

    In [45]: standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)

    In [46]: standard_wavelengths
    Out[46]: 
    array([  60.,   80.,  100.,  120.,  140.,  160.,  180.,  200.,  220.,
            240.,  260.,  280.,  300.,  320.,  340.,  360.,  380.,  400.,
            420.,  440.,  460.,  480.,  500.,  520.,  540.,  560.,  580.,
            600.,  620.,  640.,  660.,  680.,  700.,  720.,  740.,  760.,
            780.,  800.], dtype=float32)

    In [47]: len(standard_wavelengths)
    Out[47]: 38




* what to do about that ?

  * tighten the range to a more relevant one, and reduce bin size to 
    keep roughly the same number of bins 

  * reduce bin size  

  * variable bin size ? bad performance impact presumably 

    * could use a coarse and a fine 


closer look at scintillation wavelength
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* chroma has sharp cutoff at 600nm and less clear drop down at around 350nm


The cdf becomes flat above 600nm::

    In [41]: np.set_printoptions(precision=10)

    In [42]: s_reemission_cdf
    Out[42]: 
    array([[  60.          ,    0.          ],
           [  80.          ,    0.          ],
           [ 100.          ,    0.          ],
           [ 120.          ,    0.          ],
           [ 140.          ,    0.          ],
           [ 160.          ,    0.          ],
           [ 180.          ,    0.          ],
           [ 200.          ,    0.0000000205],
           [ 220.          ,    0.0000161953],
           [ 240.          ,    0.00003237  ],
           [ 260.          ,    0.0000485448],
           [ 280.          ,    0.0000647195],
           [ 300.          ,    0.0000808943],
           [ 320.          ,    0.000097069 ],
           [ 340.          ,    0.0009699235],
           [ 360.          ,    0.003526893 ],
           [ 380.          ,    0.0114005441],
           [ 400.          ,    0.1889369488],
           [ 420.          ,    0.5017676353],
           [ 440.          ,    0.7646831274],
           [ 460.          ,    0.9037991762],
           [ 480.          ,    0.9602615237],
           [ 500.          ,    0.9843546748],
           [ 520.          ,    0.9931191802],
           [ 540.          ,    0.9967452288],
           [ 560.          ,    0.9983744621],
           [ 580.          ,    0.99932307  ],
           [ 600.          ,    0.9999999404],
           [ 620.          ,    1.          ],
           [ 640.          ,    1.          ],
           [ 660.          ,    1.          ],
           [ 680.          ,    1.          ],
           [ 700.          ,    1.          ],
           [ 720.          ,    1.          ],
           [ 740.          ,    1.          ],
           [ 760.          ,    1.          ],
           [ 780.          ,    1.          ],
           [ 800.          ,    1.          ]], dtype=float32)

    In [43]: s_reemission_cdf.shape
    Out[43]: (38, 2)



::

    171 __device__ void
    172 generate_scintillation_photon(Photon& p, ScintillationStep& ss, curandState &rng)
    173 {
    174     float ScintillationTime = ss.ScintillationTime ;
    175     if(ss.scnt == 2)
    176     {
    177         ScintillationTime = ss.slowTimeConstant ;
    178         if(curand_uniform(&rng) < ss.slowerRatio)
    179         {
    180             ScintillationTime = ss.slowerTimeConstant ;
    181         }
    182     }
    183 
    184     p.wavelength = sample_cdf(&rng, ss.material->n,
    185                                     ss.material->wavelength0,
    186                                     ss.material->step,
    187                                     ss.material->reemission_cdf); // reemission_cdf poorly named, intensity_cdf better
    188 

`random.h`::

     25 // Draw a random sample given a cumulative distribution function
     26 // Assumptions: ncdf >= 2, cdf_y[0] is 0.0, and cdf_y[ncdf-1] is 1.0
     27 __device__ float
     28 sample_cdf(curandState *rng, int ncdf, float *cdf_x, float *cdf_y)
     29 {
     30     return interp(curand_uniform(rng),ncdf,cdf_y,cdf_x);
     31 }
     32 
     33 // Sample from a uniformly-sampled CDF
     34 __device__ float
     35 sample_cdf(curandState *rng, int ncdf, float x0, float delta, float *cdf_y)
     36 {
     37     float u = curand_uniform(rng);
     38 
     39     int lower = 0;
     40     int upper = ncdf - 1;     // far left, right bin numbers 
     41 
     42     while(lower < upper-1)    // still not settled on a bin
     43     {
     44         int half = (lower + upper) / 2;
     45 
     46         if (u < cdf_y[half])
                                      // cdf is normalized to 1 at rhs, so this is appropriate
     47             upper = half;
     48         else
     49             lower = half;
     50     }
     51 
     52     float delta_cdf_y = cdf_y[upper] - cdf_y[lower];
     53 
     54     return x0 + delta*lower + delta*(u-cdf_y[lower])/delta_cdf_y;
     55 }

Bins (ie wavelengths) beyond where the CDF reaches 1 will never get sampled. The 
source distribution is SLOWCOMPONENT (same as FASTCOMPONENT)::

     plt_gdls()  



Wavelength comes from sampling::

    In [31]: np.allclose( cg.unique_materials[3].reemission_cdf, cg.unique_materials[0].reemission_cdf )
    Out[31]: True

    In [32]: reemission_cdf = cg.unique_materials[3].reemission_cdf

    In [33]: s_reemission_cdf = standardize(reemission_cdf)



investigate G4 scintillation wavelength
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    catplot(g4s, log=True, range=(100,900))


    catplot(g4s, val='wavelength', cat='pdg', log=True, range=(100,900))   ## edges at 200, 800 nm 

    In [7]: g4s.wavelength.min()
    Out[7]: G4ScintillationPhoton(199.97593688964844, dtype=float32)

    In [8]: g4s.wavelength.max()
    Out[8]: G4ScintillationPhoton(799.8797607421875, dtype=float32)


Where did those edges come from::

    In [11]: gdls.extra.properties['SLOWCOMPONENT']
    Out[11]: 
    array([[  79.99 ,    0.   ],
           [ 120.023,    0.   ],
           [ 199.975,    0.   ],     ## note 130nm jump to zero bin
           [ 330.   ,    0.006],
           [ 331.   ,    0.006],
           [ 332.   ,    0.005],
           [ 333.   ,    0.005],
           ...
           [ 598.001,    0.002],
           [ 599.001,    0.002],
           [ 600.001,    0.002],
           [ 799.898,    0.   ]])    ## note 200nm jump to zero bin  


::

    catplot(g4s, val='time', cat='scnt', log=True, range=(0,50))


::

    [blyth@ntugrid5 geant4.9.2.p01]$ find $PWD -name 'G4PhysicsVector.hh' 
    /home/blyth/local/env/dyb/external/build/LCG/geant4.9.2.p01/include/G4PhysicsVector.hh
    /home/blyth/local/env/dyb/external/build/LCG/geant4.9.2.p01/source/global/management/include/G4PhysicsVector.hh

    [blyth@ntugrid5 geant4.9.2.p01]$ find $PWD -name 'G4PhysicsOrderedFreeVector.hh' 
    /home/blyth/local/env/dyb/external/build/LCG/geant4.9.2.p01/include/G4PhysicsOrderedFreeVector.hh
    /home/blyth/local/env/dyb/external/build/LCG/geant4.9.2.p01/source/global/management/include/G4PhysicsOrderedFreeVector.hh
    [blyth@ntugrid5 geant4.9.2.p01]$ 



Grab the scintillation integrals using G4DAEPropList::

    G4DAEArray::Allocate nitems 275 nfloat 550 
    G4DAEArray::Allocate nitems 275 nfloat 550 
    G4DAEArray::Allocate nitems 28 nfloat 56 
    G4DAEArray::SavePath [/home/blyth/local/env/prop/ls_fast.npy] itemcount 275 itemshape 2 
    G4DAEArray::SavePath [/home/blyth/local/env/prop/ls_slow.npy] itemcount 275 itemshape 2 
    G4DAEArray::SavePath [/home/blyth/local/env/prop/ls_reem.npy] itemcount 28 itemshape 2 
    G4DAEArray::Allocate nitems 275 nfloat 550 
    G4DAEArray::Allocate nitems 275 nfloat 550 
    G4DAEArray::Allocate nitems 28 nfloat 56 
    G4DAEArray::SavePath [/home/blyth/local/env/prop/gdls_fast.npy] itemcount 275 itemshape 2 
    G4DAEArray::SavePath [/home/blyth/local/env/prop/gdls_slow.npy] itemcount 275 itemshape 2 
    G4DAEArray::SavePath [/home/blyth/local/env/prop/gdls_reem.npy] itemcount 28 itemshape 2 
    physicsList->setCut() start.


::

    delta:~ blyth$ export-prop-rget | sh 
    gdls_fast.npy                                                                                                                           100% 2280     2.2KB/s   00:00    
    ls_slow.npy                                                                                                                             100% 2280     2.2KB/s   00:00    
    gdls_slow.npy                                                                                                                           100% 2280     2.2KB/s   00:00    
    gdls_reem.npy                                                                                                                           100%  304     0.3KB/s   00:00    
    ls_reem.npy                                                                                                                             100%  304     0.3KB/s   00:00    
    ls_fast.npy                                                                                                                             100% 2280     2.2KB/s   00:00    


Energy is xscaled to be in reciprocal wavelength (1/nm) and yscale is 1e9::

    ls = pro_("ls_fast")
    plt.plot(ls[:,0], ls[:,1])
    plt.show()

    plt.plot(1./ls[:,0], ls[:,1], "r+")
    plt.show()



Establish connection between scintillation step and the transported scintillation integeral::


   In [3]: g4s = ScintillationStep.get(1)

    In [13]: np.unique(g4s[:,5,1]).item()*1e9
    Out[13]: 410.0278374608024


Cheat by using purloined ScintillationIntegral in gdct- test_ScintillationIntegral succeeds
to reproduce the scintillation wavelengths. 
But this is essentially using the same G4 code so no surprise::

    int test_ScintillationIntegral()
    {
        G4DAEPropList* cdf = G4DAEPropList::Load("gdls_fast");
        cdf->Print();
        G4PhysicsOrderedFreeVector* ScintillationIntegral = G4DAEProp::CreatePOFV(cdf);
        G4double MaxValue = ScintillationIntegral->GetMaxValue() ;

        //size_t size = 1e6 ; 
        size_t size = 2817543 ;  // match the count to current evt "1"

        G4DAEArrayHolder* holder = new G4DAEArrayHolder( size, NULL, "2" );
        for(size_t n=0 ; n<size ; n++ )
        {
            G4double CIIvalue = G4UniformRand()*MaxValue;
            G4double sampledEnergy = ScintillationIntegral->GetEnergy(CIIvalue);

            float* prop = holder->GetNextPointer();
            prop[G4DAEProp::_binEdge]  = float(CIIvalue) ;
            prop[G4DAEProp::_binValue] = float(sampledEnergy) ;
        }

        G4DAEPropList dist(holder);
        dist.Save("1");  // sampledEnergy

        //
        //  cf('wavelength', typs="gopscintillation opscintillation prop",tag=1,  log=True, range=(100,900) )
        //   succeeds to match G4 Scintillation photon distrib 
        //
        return 0 ; 
    }


What about numpy level::

    In [13]: cdf = pro_("gdls_fast")

    In [16]: mx = cdf[:,1].max()

    In [17]: mx
    Out[17]: 410.02786

    In [18]: u = np.random.rand( 2817543 )

    In [19]: u.shape
    Out[19]: (2817543,)

Need to invert x to have wavelength ordinate, but that makes CDF back to front::

    In [32]: plt.plot(1/cdf[:,0],cdf[:,1])
    Out[32]: [<matplotlib.lines.Line2D at 0x11645d5d0>]

So interpolate in 1/wavelength land and invert afterwards,
this avoids the question of how to deal with infinite wavelength::

    In [46]: wi = np.interp( u*cdf[:,1].max(), cdf[:,1], cdf[:,0] )  ## NB x-y flip 

    In [47]: w = 1/wi

    In [51]: plt.hist(w, bins=100, log=True, range=(100,900)) ## looking good


compare cdfs
~~~~~~~~~~~~~~~~

So how does the purloined scintillation integral compare with what have been using.

::

    In [3]: cg = cg_get()

    In [7]: ls = cg.unique_materials[0]

    In [13]: rcdf = ls.reemission_cdf

    In [14]: plt.plot(rcdf[:,0], rcdf[:,1])
    Out[14]: [<matplotlib.lines.Line2D at 0x116369050>]

    In [15]: plt.show()


::

    In [41]: plt.plot( 1/rcdf[:,0], rcdf[:,1], 'b+')
    Out[41]: [<matplotlib.lines.Line2D at 0x10ccc4310>]

    In [45]: plt.plot( cdf[:,0], 1 - cdf[:,1]/cdf[:,1].max(), 'r+')
    Out[45]: [<matplotlib.lines.Line2D at 0x126e99650>]




::

    In [48]: plt.plot( cdf[:,0], 1 - cdf[:,1]/cdf[:,1].max(), 'r+',   1/rcdf[:,0], rcdf[:,1], 'b+'  )

    In [50]: plt.plot( 1/cdf[:,0], 1 - cdf[:,1]/cdf[:,1].max(), 'r+',   rcdf[:,0], rcdf[:,1], 'b+'  )

    In [64]: plt.plot( 1/cdf[:,0], 1 - cdf[:,1]/cdf[:,1].max(), 'r+-',   rcdf[:,0], rcdf[:,1], 'b+-'  )

::

    In [52]: ls = get_ls()

    In [57]: fast = ls.extra.properties['FASTCOMPONENT'].astype(np.float64)

    cy = np.cumsum(fast[:,1], dtype=np.float64)   ## cumulative in wavelength land

    fcdf = np.vstack([fast[:,0],cy/cy[-1]]).T     ## cdf in wavelength 

    In [112]: np.allclose(fcdf, rcdf)
    Out[112]: True


Try duplicating BuildPhysicsTable, fiddly bin averaging::

    In [129]: rfast = fast[::-1]   # reverse order to be in ascending energy 

    In [130]: rfast[0]
    Out[130]: array([ 799.898,    0.   ])

::

    In [146]: x = 1/rfast[:,0]     # work in inverse wavelength 1/nm

    In [147]: y = rfast[:,1]

    (y[1:]+y[:-1])/2      # sum of bins

    np.cumsum( (y[1:]+y[:-1])/2 * np.diff(x) )*1e6

    In [168]: bcdf = np.vstack( [x[1:], cy/cy[-1]] ).T

    In [175]: xcdf = cdf.copy()

    In [176]: xcdf[:,1] = xcdf[:,1]/xcdf[:,1].max()

    In [180]: np.allclose( xcdf[1:], bcdf )
    Out[180]: True


Avoid loosing the bin::

    In [185]: bcdf = np.empty( fast.shape )

    In [194]: bcdf[0] = 1/fast[-1,0], 0

    bcdf[:,0] = x

    np.cumsum(ymid*xdif, out=bcdf[1:,1])

    bcdf[1:,1] = bcdf[1:,1]/bcdf[1:,1].max() 

    In [216]: np.allclose(bcdf, xcdf)
    Out[216]: True


Lay this down in collada_to_chroma:construct_cdf_energywise 



test with energywise cdf 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    g4daechroma.sh --nogeocache

    npysend.sh -t1 -iscintillation -otest 


Getting the energywise CDF onto GPU is complicated 
by chroma wavelength standardization, which does an interpolation
to that standard wavelengths. As interpolation requires
ascending "x" need to flip order.

Some success with handling energywise cdf, but suspect getting
back to front wavelength distrib::

     67         def interp_material_property(wavelengths, prop):
     68             # note that it is essential that the material properties be
     69             # interpolated linearly. this fact is used in the propagation
     70             # code to guarantee that probabilities still sum to one.
     71 
     72             ascending = np.all(np.diff(prop[:,0]) >= 0)
     73             descending = np.all(np.diff(prop[:,0]) <= 0)
     74 
     75             if ascending:
     76                 return np.interp(wavelengths, prop[:,0], prop[:,1]).astype(np.float32)
     77             elif descending:
     78                 # the interpolation needs ascending so reverse here, then reverse back after
     79                 iprop = np.interp(wavelengths, prop[::-1,0], prop[::-1,1]).astype(np.float32)
     80                 return iprop[::-1].copy()
     81             else:
     82                 assert 0, "needs to be all ascending or all descending "
     83                 return None


Access test wavelengths::

    In [1]: t = ttt_(1)

    In [4]: w = t[:,1,3]


Getting some infinites, probably LS material index shift::

    In [26]: w[w==np.inf].shape
    Out[26]: (598018,)

    In [27]: w[w!=np.inf].shape
    Out[27]: (2219525,)

The non infinities look like a wavelength distrib::

    In [7]: ww=w[w!=np.Inf]

    In [9]: plt.hist(ww, bins=100)   ## wavelength flipped distribution, maybe need to "1 - cdf" 


Succeed to get rid of infinities by establishing order of chroma 
materials and surfaces to be based on names with pointer address excluded. 


After adjusting to use 1/wavelength[::-1] domain reemission_cdf 
and using kernel sampling that takes that into account, 

`chroma/cuda/scintillation.h`::

    194     p.wavelength = sample_reciprocal_cdf(&rng, ss.material->n,
    195                                                ss.material->wavelength0,
    196                                                ss.material->step,
    197                                                ss.material->reemission_cdf);



::

    cf('wavelength', tag=1, typs='gopscintillation opscintillation', log=True )




time
~~~~~~~~

Scintillation time, almost perfect close match::

    In [7]: cf('time', g4s , chs )     ## very long tail

    In [30]: cf('time', g4s , chs, range=(0,100))


xyz pos,dir,pol
~~~~~~~~~~~~~~~~~

Position, direction and polarization all almost perfect matches, wavelength needs some attention::

    In [32]: cf('3xyzw', g4s, chs, legend=False)



scintillation review
~~~~~~~~~~~~~~~~~~~~~~

Looking good::

    In [3]: cf('3xyzw', tag=1, typs='gopscintillation opscintillation', legend=False, log=True)



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




Wavelength Ranges from G4 to Chroma
-------------------------------------

::

    In [15]: _gdls = gdls()

    In [18]: _gdls.__class__
    Out[18]: collada.material.Material

    In [21]: slow = _gdls.extra.properties['SLOWCOMPONENT']

    In [22]: plt.scatter(slow[:,0],slow[:,1])
    Out[22]: <matplotlib.collections.PathCollection at 0x115e406d0>

    In [23]: plt.show()


Wide range, but very few entries at extremes and near zero anyhow, all action in middle::


    In [20]: _gdls.extra.properties['SLOWCOMPONENT']
    Out[20]: 
    array([[  79.99 ,    0.   ],
           [ 120.023,    0.   ],
           [ 199.975,    0.   ],
           [ 330.   ,    0.006],
           [ 331.   ,    0.006],
           [ 332.   ,    0.005],
           [ 333.   ,    0.005],
           ...
           [ 598.001,    0.002],
           [ 599.001,    0.002],
           [ 600.001,    0.002],
           [ 799.898,    0.   ]])


    In [24]: slow[:,0].min()
    Out[24]: 79.989835277575907

    In [25]: slow[:,0].max()
    Out[25]: 799.89835277575912


Chopping the extremes::

    In [28]: plt.scatter(slow[10:-10,0],slow[10:-10,1])
    Out[28]: <matplotlib.collections.PathCollection at 0x124b8f110>

    In [29]: plt.show()



The wide range feeds forward into chroma::

    In [33]: cg = chroma_geometry()

    In [37]: cg.unique_materials[0].name
    Out[37]: '__dd__Materials__LiquidScintillator0xc2308d0'

    In [38]: cls = cg.unique_materials[0]

    In [40]: cls.reemission_cdf.shape
    Out[40]: (275, 2)

    In [41]: slow.shape
    Out[41]: (275, 2)

    In [44]: np.allclose( cls.reemission_cdf[:,0], slow[:,0] )
    Out[44]: True



`chroma/chroma/geometry.py`::

     25 # all material/surface properties are interpolated at these
     26 # wavelengths when they are sent to the gpu
     27 standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)
     28 

Hmm thats pretty coarse, this explains the generated scintillation wavelength distrib.  



