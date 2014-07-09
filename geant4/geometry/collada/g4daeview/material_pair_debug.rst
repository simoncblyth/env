Material Pair Debug
====================

Thoughts on how to proceed
---------------------------

There is a wealth of details and curiosities (asymmetries, unexpected positions), 
but do they matter ? 

Some will be artifacts of selection techniques, presentation techniques, truncation.
 
Avoid getting bogged down in these now until have Geant4 equivalents to compare against, 
otherwise risk spending time barking up the wrong trees. This will provide a candle against 
which it should be easier to see which aspects need attention.

So, in summary aim for Geant4 comparison earlier rather than later : once have
interface into main features of the propagation. (currently missing surfaces).
 

Geant4 step level comparison
-----------------------------

Collect G4 steps into VBO like structure ? 

Machinery
-----------

#. More intuitive to select photons that do a particular step
   rather than select steps.  How to implement ?


DAEPhotonsAnalyzer.get_material_pairs
----------------------------------------

Both end material check
~~~~~~~~~~~~~~~~~~~~~~~~~

Pushing up to `--max-slots 30` reduces asymmetry peculiarities a lot. 
Ten slots (which is effectively 8 slots) is simply not enough. Pushing to  
`--max-slots 100` suffers from crashes when in animation modes and slow interface
in other modes.

With selection based on from/to materials. Peculiarities::


    Tyvek,Tyvek                     9755    # not real, just 0,0
    GdDopedLS,Acrylic               6209   confetti and noodle modes show many steps starting on upper acylic of inner AD, almost none from the bottom
           # large numbers along muon path are expected, its almost inevitable that any step staring inside GDLS is going to interesect to acrylic

    udp.py --material GdDopedLS,ANY
           # generalize, still almost nothing downwards


    Acrylic,LiquidScintillator      5809 
    LiquidScintillator,Acrylic      5402 
    MineralOil,Acrylic              5000 
    Acrylic,MineralOil              3575 
    ESR,Air                         2079    
    udp.py --material ESR,Air  --style noodles,confetti   ## almost all at base, pointed down
    pushing up max-slots to 30 equalizes this a lot, so a truncation artifact    

    StainlessSteel,IwsWater         1178 
    Acrylic,Air                     482 
    Acrylic,GdDopedLS               402     # huge asymmetry, whereas not for LS ?
    MineralOil,Pyrex                306     # OK: Mostly just inside the MineralOil, nr Acrylic, and all pointing at PMTs. Top orthographic clearest
    StainlessSteel,MineralOil       258 
    MineralOil,StainlessSteel       240 
    Air,ESR                         237    udp.py --material Air,ESR  --style noodles,confetti   ## also, almost all at base
    Vacuum,Bialkali                 191    # efficiency ? 
    Pyrex,Vacuum                    165    # OK: all inside PMTs, for clear view make geometry invisible with I
    Bialkali,Vacuum                 130 
    UnstStainlessSteel,MineralOil   82 
    Pyrex,MineralOil                59 
    IwsWater,IwsWater               16 
    Vacuum,OpaqueVacuum             14 
    StainlessSteel,Water            9 
    MineralOil,UnstStainlessSteel   8 
    LiquidScintillator,Teflon       8 
    LiquidScintillator,GdDopedLS    7 
    LiquidScintillator,MineralOil   6 
    GdDopedLS,LiquidScintillator    4 
    OwsWater,Pyrex                  3 
    OpaqueVacuum,Vacuum             3 
    IwsWater,StainlessSteel         2 
    Teflon,LiquidScintillator       2 
    Water,StainlessSteel            2 
    StainlessSteel,GdDopedLS        2 
    StainlessSteel,NitrogenGas      2 
    OwsWater,UnstStainlessSteel     1 
    Air,Acrylic                     1 
    IwsWater,Water                  1 
     





::

    g4daeview.sh  --load 1 --wipepropagate --debugkernel --debugphoton 0 --max-slots 30

    54068 Tyvek,Tyvek                        # dummy  

    12936 LiquidScintillator,Acrylic        
    12365 Acrylic,LiquidScintillator        
    12324 MineralOil,Acrylic                

    9128 Acrylic,MineralOil                 
    8646 GdDopedLS,Acrylic                  
    2761 Acrylic,GdDopedLS                  

    2678 ESR,Air                            
    2169 Acrylic,Air                        
    1425 StainlessSteel,IwsWater            
    1139 MineralOil,StainlessSteel          

    759  Air,ESR                            
    697  Bialkali,Vacuum                    
    655  MineralOil,Pyrex                   
    638  Vacuum,Bialkali                    
    636  Pyrex,Vacuum                       
    525  Pyrex,MineralOil                   

    295  StainlessSteel,MineralOil          
    255  Vacuum,Pyrex                       
    240  Vacuum,OpaqueVacuum                
    226  OpaqueVacuum,Vacuum                
    105  UnstStainlessSteel,MineralOil      
    62   MineralOil,UnstStainlessSteel      
    50   UNKNOWN,UNKNOWN                    
    30   LiquidScintillator,GdDopedLS       
    23   Teflon,LiquidScintillator          
    21   LiquidScintillator,MineralOil      
    19   GdDopedLS,LiquidScintillator       
    17   LiquidScintillator,Teflon          
    16   MineralOil,LiquidScintillator      
    10   StainlessSteel,Water               
    5    IwsWater,IwsWater                  
    4    Air,Acrylic                        
    3    StainlessSteel,GdDopedLS           
    3    GdDopedLS,Teflon                   
    3    Water,StainlessSteel               
    3    OwsWater,Pyrex                     
    3    Teflon,GdDopedLS                   
    3    StainlessSteel,NitrogenGas         
    2    IwsWater,StainlessSteel            
    2    IwsWater,Water                     
    1    OwsWater,UnstStainlessSteel        



::

    g4daeview.sh  --load 1 --wipepropagate --debugkernel --debugphoton 0 --max-slots 100

    304179 LiquidScintillator,LiquidScintillator    # dummy zeros

    20825 MineralOil,Acrylic                
    20400 LiquidScintillator,Acrylic        
    19592 Acrylic,LiquidScintillator        

    15015 Acrylic,MineralOil                
    11249 GdDopedLS,Acrylic                 
    5097 Acrylic,GdDopedLS                  

    3423 Acrylic,Air                        
    2886 ESR,Air                            
    2318 MineralOil,StainlessSteel          
    1680 StainlessSteel,IwsWater              # all absorption steps presumably 

    1561 Bialkali,Vacuum                    
    1128 Vacuum,Bialkali                    
    1126 Pyrex,Vacuum                       
    1094 MineralOil,Pyrex                   
    1006 Pyrex,MineralOil                   
    963  Air,ESR                           

    526  Vacuum,Pyrex                       
    466  Vacuum,OpaqueVacuum                
    454  OpaqueVacuum,Vacuum                
    371  StainlessSteel,MineralOil          
    189  LiquidScintillator,GdDopedLS       
    187  Teflon,LiquidScintillator          
    155  LiquidScintillator,MineralOil      
    123  UnstStainlessSteel,MineralOil      
    110  MineralOil,UnstStainlessSteel      
    95   GdDopedLS,LiquidScintillator       
    83   MineralOil,LiquidScintillator      
    65   LiquidScintillator,Teflon          
    51   UNKNOWN,UNKNOWN                    
    15   IwsWater,StainlessSteel            
    14   Water,StainlessSteel               
    12   StainlessSteel,Water               
    6    IwsWater,IwsWater                  
    6    IwsWater,Water                     
    5    Air,Acrylic                        
    3    GdDopedLS,Teflon                   
    3    Teflon,GdDopedLS                   
    3    StainlessSteel,GdDopedLS           
    3    OwsWater,Pyrex                     
    3    StainlessSteel,NitrogenGas         
    2    Nitrogen,Water                     
    1    Water,IwsWater                     
    1    Vacuum,Acrylic                     
    1    OwsWater,UnstStainlessSteel        
    1    Nitrogen,MineralOil                
    1    MineralOil,Nitrogen                
    1    MineralOil,Vacuum                  
    1    Acrylic,Vacuum                     
    1    Vacuum,MineralOil                  







Implement step selection based in *from* and *to* materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need to expand VBO to hold material info.


Broken chain
~~~~~~~~~~~~~~

::

    p_flags[1077]                  p_lht[1077]                                          
             [[ 10.017   10.017 ]  [[  1255      3      5      0] GdDopedLS  Acrylic    
              [ 10.017   19.2213]   [   991      5     11      1] Acrylic    LiquidScin 
              [ 10.017   19.3094]   [   635     11      5      2] LiquidScin Acrylic    
              [ 10.017   21.8971]   [   371      5      6      3] Acrylic    MineralOil 
              [ 10.017   22.0028]   [632049      5      1      4] Acrylic    Air           
              [ 10.017   22.0614]   [632341      1      4      5] Air        ESR        
              [ 10.017   22.0616]   [632354      4      1      6] ESR        Air        
    B_ABSORB  [ 10.017   22.0616]   [    -1      4      1      7] ESR        Air        
    B_ABSORB  [ 10.017   22.0616]   [    -1      4      1      7] ESR        Air        
    B_ABSORB  [ 10.017   22.0616]]  [    -1      4      1      7]] ESR        Air       
    p_post[1077]                                                     p_dirw[1077]                                p_polw[1077]                        p_ccol[1077]       
    [[ -18007.123  -799808.25     -7062.793       10.017 ] -1        [[  -0.2251    0.4768   -0.8497  487.7488]  [[ 0.7983 -0.4097 -0.4414  1.    ]  [[ 1.  1.  1.  1.] 
     [ -18423.6504 -798926.       -8635.          19.2213] 1850.32    [  -0.224     0.4745   -0.8513  487.7488]   [-0.3634  0.7698  0.5247  1.    ]   [ 1.  1.  1.  1.] 
     [ -18427.5977 -798917.625    -8650.          19.3094] 17.6273    [  -0.2251    0.4768   -0.8497  487.7488]   [-0.3628  0.7684  0.5273  1.    ]   [ 1.  1.  1.  1.] 
     [ -18544.6973 -798669.625    -9092.          21.8971] 520.173    [  -0.224     0.4745   -0.8513  487.7488]   [-0.3634  0.7698  0.5247  1.    ]   [ 1.  1.  1.  1.] 
     [ -18549.4336 -798659.5625   -9110.          22.0028] 21.1586    [  -0.2287    0.4843   -0.8445  487.7488]   [-0.3605  0.7637  0.5356  1.    ]   [ 1.  1.  1.  1.] 
     [ -18552.1133 -798653.875    -9119.9004      22.0614] 11.728     [  -0.3425    0.7255   -0.5969  487.7488]   [-0.2548  0.5397  0.8023  1.    ]   [ 1.  1.  1.  1.] 
     [ -18552.1426 -798653.8125   -9119.9502      22.0616] 0.0851179  [  -0.3426    0.7257   -0.5966  487.7488]   [-0.2547  0.5395  0.8026  1.    ]   [ 1.  1.  1.  1.] 
     [ -18552.1426 -798653.8125   -9119.9502      22.0616] 0.0        [  -0.3426    0.7257   -0.5966  487.7488]   [-0.2547  0.5395  0.8026  1.    ]   [ 1.  0.  0.  1.] 
     [ -18552.1426 -798653.8125   -9119.9502      22.0616] 0.0        [  -0.3426    0.7257   -0.5966  487.7488]   [-0.2547  0.5395  0.8026  1.    ]   [ 1.  0.  0.  1.] 
     [ -18334.7812 -799114.25     -8299.5605      17.2575]] 965.551   [  -0.2251    0.4768   -0.8497  487.7488]]  [ 0.7983 -0.4097 -0.4414  1.    ]]  [ 1.  1.  1.  1.]]
    t_post[1077]                                          t_dirw[1077]                              t_polw[1077]                      t_ccol[1077]     


Examine the big hitters
-------------------------

Despite the crash when attempt to animate, can use other modes with `--max-slots=100` 
which eliminates trunctions.::

    g4daeview.sh  --load 1 --wipepropagate --debugkernel --debugphoton 0 --max-slots 100


Interface is slow however. The photon disappearance technique does not prevent the 
processing of the invisibles.

Observations
~~~~~~~~~~~~~~

#. big hitters are mostly chroma level truncated

::
    h = plt.hist(z.slots[z.slots>50], bins=np.linspace(50,100,51)) 
    udp.py --sid 2190 --style spagetti,confetti,noodles


Long lived
------------

::

    daephotonsanalyzer.sh --load 1

    In [2]: z.tl.min(), z.tl.max()
    Out[2]: (0.0, 254.15379)

    In [4]: np.where(z.tl > 200)
    Out[4]: 
    (array([ 181,  415,  809, 1127, 1230, 1244, 1305, 1784, 1798, 1852, 1856,
           2365, 2439, 2547, 2593, 2799, 2926, 2938, 3024, 3077, 3158, 3655,
           3659, 3831]),)


::

    udp.py --sid 181 --style spagetti,confetti,noodles




