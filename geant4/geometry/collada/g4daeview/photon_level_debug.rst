Photon Level Debug
===================

Issues
---------

#. photon visualization disappearance, even with eg `--mode 7` to exclude truncated
#. non-sensical discontinuities in propagation history animation  
#. do many step histories have something different about them,  
 
   * wavelength ? REEMISSION 
   * lots of specular reflections 


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

Use `--debugpropagate` to write files `propagated-<seed>.npz` into the invoking directory
after performing propagations, which happen as event files are loaded  eg::

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
   







