Photon Level Debug
===================

Issues
---------

#. photon disappearance, even with eg `--mode 7` to exclude truncated
#. non-sensical discontinuities in propagation history animation  


Techniques
------------

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


::

    I: photon_id 6 tail_birth 2.510489 tail_death 14.021963  cohort 2.500000 2.600000 1.000000 
    I: photon_id 7 tail_birth 2.548814 tail_death 2.645415  cohort 2.500000 2.600000 1.000000 
    I: photon_id 8 tail_birth 2.552698 tail_death 2.582994  cohort 2.500000 2.600000 1.000000 
    I: photon_id 9 tail_birth 2.586238 tail_death 2.637245  cohort 2.500000 2.600000 1.000000 

    I: photon_id 6 tail_birth 2.510489 tail_death 14.021963  cohort 2.500000 2.600000 1.000000 
    I: photon_id 7 tail_birth 2.548814 tail_death 2.645415  cohort 2.500000 2.600000 1.000000 

::

    udp.py --cohort 2.51,2.52,1.   # down to single photon_id 6 

::

    udp.py --mode 0 --style confetti

    ## despite animation not working, using time reveal --mode 0 and confetti style allows to see the direction, bounce times



Trying to use debugphoton to see the history, suspect starts the same but a RAYLEIGH_SCATTER happened to change history?::

    g4daeview.sh --with-chroma --load 1 --debugkernel --timerange 0,30 --time 0 --debugphoton 6 --pid 6 --fpholine 300  

::

    2014-06-27 15:22:04,949 env.geant4.geometry.collada.g4daeview.daephotonskernelfunc:67  _set_mask : memcpy_htod [-1, -1, 6, -1] 
    FILL_STATE       START    [     6] slot  0 steps  1 lht 621543 tpos    2.510  -16823.59 -801640.62   -7065.90    w  383.88   dir    -0.94     0.20     0.29 pol   -0.121   -0.956    0.266 
    TO_BOUNDARY      PASS     [     6] slot -1 steps  1 lht 621543 tpos    2.924  -16901.80 -801623.94   -7041.46    w  383.88   dir    -0.94     0.20     0.29 pol   -0.121   -0.956    0.266 
    AT_SURFACE       CONTINUE [     6] slot -1 steps  1 lht 621543 tpos    2.924  -16901.80 -801623.94   -7041.46    w  383.88   dir    -0.44    -0.85     0.29 pol   -0.121   -0.956    0.266 REFLECT_SPECULAR 
    FILL_STATE       CONTINUE [     6] slot  1 steps  2 lht    214 tpos    2.924  -16901.80 -801623.94   -7041.46    w  383.88   dir    -0.44    -0.85     0.29 pol   -0.121   -0.956    0.266 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps  2 lht    214 tpos    4.830  -17071.39 -801951.44   -6928.56    w  383.88   dir    -0.44    -0.85     0.29 pol   -0.121   -0.956    0.266 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [     6] slot -1 steps  2 lht    214 tpos    4.830  -17071.39 -801951.44   -6928.56    w  383.88   dir    -0.94     0.20     0.29 pol    0.138    0.968   -0.208 REFLECT_SPECULAR 
    FILL_STATE       PASS     [     6] slot  2 steps  3 lht 621451 tpos    4.830  -17071.39 -801951.44   -6928.56    w  383.88   dir    -0.94     0.20     0.29 pol    0.138    0.968   -0.208 REFLECT_SPECULAR 
    TO_BOUNDARY      CONTINUE [     6] slot -1 steps  3 lht     -1 tpos    6.867  -17457.21 -801870.62   -6807.88    w  383.88   dir     0.41     0.53     0.74 pol   -0.010    0.856   -0.518 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    FILL_STATE       CONTINUE [     6] slot  3 steps  4 lht 621451 tpos    6.867  -17457.21 -801870.62   -6807.88    w  383.88   dir     0.41     0.53     0.74 pol   -0.010    0.856   -0.518 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [     6] slot -1 steps  4 lht 621451 tpos    6.920  -17452.83 -801865.00   -6800.03    w  383.88   dir     0.41     0.53     0.74 pol   -0.010    0.856   -0.518 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    AT_SURFACE       CONTINUE [     6] slot -1 steps  4 lht 621451 tpos    6.920  -17452.83 -801865.00   -6800.03    w  383.88   dir     0.57    -0.35     0.74 pol   -0.010    0.856   -0.518 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    FILL_STATE       CONTINUE [     6] slot  4 steps  5 lht    214 tpos    6.920  -17452.83 -801865.00   -6800.03    w  383.88   dir     0.57    -0.35     0.74 pol   -0.010    0.856   -0.518 RAYLEIGH_SCATTER REFLECT_SPECULAR 
    TO_BOUNDARY      BREAK    [     6] slot -1 steps  5 lht     -1 tpos    8.273  -17295.60 -801960.81   -6597.52    w  383.88   dir     0.57    -0.35     0.74 pol   -0.010    0.856   -0.518 RAYLEIGH_SCATTER REFLECT_SPECULAR BULK_ABSORB 
    2014-06-27 15:22:06,026 env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:115 dump


::

    I: photon_id 6 tail_birth 2.510489 tail_death 8.272550  cohort 2.510000 2.520000 1.000000   // early death



Where to do seed control ?




Repeatability/Seeding Doubts
------------------------------

Repeatability is usually seen, but suspect there are lapses. NEED automated checking to learn more.

::

    g4daeview.sh --with-chroma --load 1 --style movie-extra --timerange 0,30 --time 0 --debugkernel --debugphoton 200 --pid 200 --fpholine 300 


::

    2014-06-27 13:27:49,668 env.geant4.geometry.collada.g4daeview.daephotonskernelfunc:67  _set_mask : memcpy_htod [-1, -1, 200, -1] 
    FILL_STATE       START    [   200] slot  0 steps  1 lht    577 tpos    5.316  -17131.59 -801163.38   -7064.69    w  422.18   dir     0.87    -0.13     0.48 pol    0.466   -0.142   -0.873 
    TO_BOUNDARY      PASS     [   200] slot -1 steps  1 lht    577 tpos    7.107  -16820.82 -801208.62   -6891.35    w  422.18   dir     0.87    -0.13     0.48 pol    0.466   -0.142   -0.873 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps  1 lht    577 tpos    7.107  -16820.82 -801208.62   -6891.35    w  422.18   dir     0.87    -0.13     0.48 pol   -0.099    0.900    0.424 
    FILL_STATE       PASS     [   200] slot  1 steps  2 lht    289 tpos    7.107  -16820.82 -801208.62   -6891.35    w  422.18   dir     0.87    -0.13     0.48 pol   -0.099    0.900    0.424 
    TO_BOUNDARY      PASS     [   200] slot -1 steps  2 lht    289 tpos    7.242  -16797.43 -801212.19   -6878.37    w  422.18   dir     0.87    -0.13     0.48 pol   -0.099    0.900    0.424 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps  2 lht    289 tpos    7.242  -16797.43 -801212.19   -6878.37    w  422.18   dir     0.86    -0.11     0.49 pol   -0.119    0.903    0.413 
    FILL_STATE       PASS     [   200] slot  2 steps  3 lht 621995 tpos    7.242  -16797.43 -801212.19   -6878.37    w  422.18   dir     0.86    -0.11     0.49 pol   -0.119    0.903    0.413 
    TO_BOUNDARY      PASS     [   200] slot -1 steps  3 lht 621995 tpos    9.202  -16452.90 -801256.06   -6682.81    w  422.18   dir     0.86    -0.11     0.49 pol   -0.119    0.903    0.413 
    AT_SURFACE       CONTINUE [   200] slot -1 steps  3 lht 621995 tpos    9.202  -16452.90 -801256.06   -6682.81    w  422.18   dir     0.03     0.87     0.49 pol   -0.119    0.903    0.413 REFLECT_SPECULAR 
    FILL_STATE       CONTINUE [   200] slot  3 steps  4 lht    291 tpos    9.202  -16452.90 -801256.06   -6682.81    w  422.18   dir     0.03     0.87     0.49 pol   -0.119    0.903    0.413 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps  4 lht    291 tpos   11.695  -16439.84 -800814.62   -6434.14    w  422.18   dir     0.03     0.87     0.49 pol   -0.119    0.903    0.413 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps  4 lht    291 tpos   11.695  -16439.84 -800814.62   -6434.14    w  422.18   dir    -0.01     0.88     0.48 pol    0.949   -0.145    0.281 REFLECT_SPECULAR 
    FILL_STATE       PASS     [   200] slot  4 steps  5 lht    579 tpos   11.695  -16439.84 -800814.62   -6434.14    w  422.18   dir    -0.01     0.88     0.48 pol    0.949   -0.145    0.281 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps  5 lht    579 tpos   11.872  -16440.13 -800783.62   -6417.14    w  422.18   dir    -0.01     0.88     0.48 pol    0.949   -0.145    0.281 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps  5 lht    579 tpos   11.872  -16440.13 -800783.62   -6417.14    w  422.18   dir    -0.00     0.88     0.48 pol    0.949   -0.153    0.277 REFLECT_SPECULAR 
    FILL_STATE       PASS     [   200] slot  5 steps  6 lht    589 tpos   11.872  -16440.13 -800783.62   -6417.14    w  422.18   dir    -0.00     0.88     0.48 pol    0.949   -0.153    0.277 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps  6 lht    589 tpos   24.366  -16440.72 -798593.06   -5208.03    w  422.18   dir    -0.00     0.88     0.48 pol    0.949   -0.153    0.277 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps  6 lht    589 tpos   24.366  -16440.72 -798593.06   -5208.03    w  422.18   dir     0.01     0.88     0.48 pol   -0.925   -0.178    0.336 REFLECT_SPECULAR 
    FILL_STATE       PASS     [   200] slot  6 steps  7 lht    301 tpos   24.366  -16440.72 -798593.06   -5208.03    w  422.18   dir     0.01     0.88     0.48 pol   -0.925   -0.178    0.336 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps  7 lht    301 tpos   24.523  -16440.53 -798565.69   -5193.02    w  422.18   dir     0.01     0.88     0.48 pol   -0.925   -0.178    0.336 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps  7 lht    301 tpos   24.523  -16440.53 -798565.69   -5193.02    w  422.18   dir    -0.02     0.87     0.49 pol   -0.925   -0.204    0.322 REFLECT_SPECULAR 
    FILL_STATE       PASS     [   200] slot  7 steps  8 lht    558 tpos   24.523  -16440.53 -798565.69   -5193.02    w  422.18   dir    -0.02     0.87     0.49 pol   -0.925   -0.204    0.322 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps  8 lht    558 tpos   24.724  -16441.39 -798530.00   -5172.91    w  422.18   dir    -0.02     0.87     0.49 pol   -0.925   -0.204    0.322 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps  8 lht    558 tpos   24.724  -16441.39 -798530.00   -5172.91    w  422.18   dir    -0.02     0.85     0.52 pol   -1.000   -0.024    0.000 REFLECT_SPECULAR 
    FILL_STATE       PASS     [   200] slot  8 steps  9 lht    492 tpos   24.724  -16441.39 -798530.00   -5172.91    w  422.18   dir    -0.02     0.85     0.52 pol   -1.000   -0.024    0.000 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps  9 lht    492 tpos   24.883  -16442.05 -798502.94   -5156.41    w  422.18   dir    -0.02     0.85     0.52 pol   -1.000   -0.024    0.000 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps  9 lht    492 tpos   24.883  -16442.05 -798502.94   -5156.41    w  422.18   dir    -0.05     0.85     0.53 pol   -0.915   -0.252    0.315 REFLECT_SPECULAR 
    FILL_STATE       PASS     [   200] slot  9 steps 10 lht 629333 tpos   24.883  -16442.05 -798502.94   -5156.41    w  422.18   dir    -0.05     0.85     0.53 pol   -0.915   -0.252    0.315 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps 10 lht 629333 tpos   26.352  -16457.13 -798250.31   -4998.00    w  422.18   dir    -0.05     0.85     0.53 pol   -0.915   -0.252    0.315 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps 10 lht 629333 tpos   26.352  -16457.13 -798250.31   -4998.00    w  422.18   dir    -0.05     0.83     0.56 pol   -0.998   -0.060    0.000 REFLECT_SPECULAR 
    FILL_STATE       PASS     [   200] slot 10 steps 11 lht 629941 tpos   26.352  -16457.13 -798250.31   -4998.00    w  422.18   dir    -0.05     0.83     0.56 pol   -0.998   -0.060    0.000 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps 11 lht 629941 tpos   26.441  -16458.01 -798235.56   -4988.10    w  422.18   dir    -0.05     0.83     0.56 pol   -0.998   -0.060    0.000 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps 11 lht 629941 tpos   26.441  -16458.01 -798235.56   -4988.10    w  422.18   dir    -0.05     0.83    -0.56 pol   -0.998   -0.060    0.000 REFLECT_SPECULAR 
    FILL_STATE       PASS     [   200] slot 11 steps 12 lht 629333 tpos   26.441  -16458.01 -798235.56   -4988.10    w  422.18   dir    -0.05     0.83    -0.56 pol   -0.998   -0.060    0.000 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps 12 lht 629333 tpos   26.530  -16458.89 -798220.81   -4998.00    w  422.18   dir    -0.05     0.83    -0.56 pol   -0.998   -0.060    0.000 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps 12 lht 629333 tpos   26.530  -16458.89 -798220.81   -4998.00    w  422.18   dir    -0.05     0.85    -0.53 pol    0.998    0.060    0.000 REFLECT_SPECULAR 
    FILL_STATE       PASS     [   200] slot 12 steps 13 lht    231 tpos   26.530  -16458.89 -798220.81   -4998.00    w  422.18   dir    -0.05     0.85    -0.53 pol    0.998    0.060    0.000 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps 13 lht    231 tpos   28.956  -16483.81 -797803.44   -5259.72    w  422.18   dir    -0.05     0.85    -0.53 pol    0.998    0.060    0.000 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps 13 lht    231 tpos   28.956  -16483.81 -797803.44   -5259.72    w  422.18   dir    -0.81    -0.25    -0.53 pol   -0.037   -0.879    0.476 REFLECT_SPECULAR 
    FILL_STATE       PASS     [   200] slot 13 steps 14 lht 628036 tpos   28.956  -16483.81 -797803.44   -5259.72    w  422.18   dir    -0.81    -0.25    -0.53 pol   -0.037   -0.879    0.476 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps 14 lht 628036 tpos   30.603  -16754.71 -797888.25   -5437.40    w  422.18   dir    -0.81    -0.25    -0.53 pol   -0.037   -0.879    0.476 REFLECT_SPECULAR 
    AT_SURFACE       CONTINUE [   200] slot -1 steps 14 lht 628036 tpos   30.603  -16754.71 -797888.25   -5437.40    w  422.18   dir    -0.05     0.85    -0.53 pol   -0.037   -0.879    0.476 REFLECT_SPECULAR 
    FILL_STATE       CONTINUE [   200] slot 14 steps 15 lht    231 tpos   30.603  -16754.71 -797888.25   -5437.40    w  422.18   dir    -0.05     0.85    -0.53 pol   -0.037   -0.879    0.476 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps 15 lht    231 tpos   32.247  -16771.07 -797605.25   -5614.64    w  422.18   dir    -0.05     0.85    -0.53 pol   -0.037   -0.879    0.476 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps 15 lht    231 tpos   32.247  -16771.07 -797605.25   -5614.64    w  422.18   dir    -0.81    -0.25    -0.53 pol   -0.036   -0.878    0.477 REFLECT_SPECULAR 
    FILL_STATE       PASS     [   200] slot 15 steps 16 lht 628359 tpos   32.247  -16771.07 -797605.25   -5614.64    w  422.18   dir    -0.81    -0.25    -0.53 pol   -0.036   -0.878    0.477 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps 16 lht 628359 tpos   33.891  -17041.45 -797690.44   -5791.91    w  422.18   dir    -0.81    -0.25    -0.53 pol   -0.036   -0.878    0.477 REFLECT_SPECULAR 
    AT_SURFACE       CONTINUE [   200] slot -1 steps 16 lht 628359 tpos   33.891  -17041.45 -797690.44   -5791.91    w  422.18   dir    -0.05     0.85    -0.53 pol   -0.036   -0.878    0.477 REFLECT_SPECULAR 
    FILL_STATE       CONTINUE [   200] slot 16 steps 17 lht    232 tpos   33.891  -17041.45 -797690.44   -5791.91    w  422.18   dir    -0.05     0.85    -0.53 pol   -0.036   -0.878    0.477 REFLECT_SPECULAR 
    TO_BOUNDARY      PASS     [   200] slot -1 steps 17 lht    232 tpos   35.314  -17055.63 -797445.56   -5945.29    w  422.18   dir    -0.05     0.85    -0.53 pol   -0.036   -0.878    0.477 REFLECT_SPECULAR 
    AT_BOUNDARY      CONTINUE [   200] slot -1 steps 17 lht    232 tpos   35.314  -17055.63 -797445.56   -5945.29    w  422.18   dir    -0.33     0.53    -0.78 pol   -0.503   -0.798   -0.331 REFLECT_SPECULAR 
    FILL_STATE       PASS     [   200] slot 17 steps 18 lht    112 tpos   35.314  -17055.63 -797445.56   -5945.29    w  422.18   dir    -0.33     0.53    -0.78 pol   -0.503   -0.798   -0.331 REFLECT_SPECULAR 
    TO_BOUNDARY      BREAK    [   200] slot -1 steps 18 lht     -1 tpos   35.314  -17055.63 -797445.56   -5945.30    w  422.18   dir    -0.33     0.53    -0.78 pol   -0.503   -0.798   -0.331 REFLECT_SPECULAR BULK_ABSORB 
    2014-06-27 13:27:50,779 env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:115 dump



