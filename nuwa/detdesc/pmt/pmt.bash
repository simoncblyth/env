# === func-gen- : nuwa/detdesc/pmt/pmt fgp nuwa/detdesc/pmt/pmt.bash fgn pmt fgh nuwa/detdesc/pmt
pmt-src(){      echo nuwa/detdesc/pmt/pmt.bash ; }
pmt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pmt-src)} ; }
pmt-vi(){       vi $(pmt-source) ; }
pmt-env(){      elocal- ; }
pmt-usage(){ cat << EOU

Analytic PMT Geometry Description
======================================


Sources
--------

analytic.py
     top level steering for pmt-analytic

dd.py 
     detdesc XML parsing 



Usage
------

To visualize analytic PMT in a box, and test ggeo- optixrap- loading::

    ggv-pmt () 
    { 
        type $FUNCNAME;
        ggv --tracer --test --eye 0.5,0.5,0.0
    }


TODO debug why below is failing::

    ggv --restrictmesh 1 --analyticmesh 1  
          # misses a GPmt associated to the MM

    ggv --test --eye 0.5,0.5,0.0 --animtimemax 10


High Level CSG Persisting for G4 geometry
-------------------------------------------

* how to represent a single node (of the 5 separate nodes) ?

  * primitives and operations

* tree of nodes  


PMT Modelling
-----------------------

Detdesc::

    lvPmtHemi                 (Pyrex)  union of 3 sphere-intersection and tubs
        pvPmtHemiVacuum

    lvPmtHemiVacuum          (Vacuum) union of 3 sphere-intersection and tubs 
        pvPmtHemiCathode
        pvPmtHemiBottom
        pvPmtHemiDynode

        [Vacuum radii names match the Pyrex with "vac" suffix]    

            PmtHemiBellyROC : 102.000000 
         PmtHemiBellyROCvac : 99.000000 

             PmtHemiFaceROC : 131.000000 
          PmtHemiFaceROCvac : 128.000000 

      PmtHemiGlassThickness : 3.000000 
    PmtHemiCathodeThickness : 0.050000 



    lvPmtHemiCathode        (Bialkali) union of 2 partial spherical shells

          outerRadius="PmtHemiFaceROCvac"
          innerRadius="PmtHemiFaceROCvac-PmtHemiCathodeThickness"

          outerRadius="PmtHemiBellyROCvac"
          innerRadius="PmtHemiBellyROCvac-PmtHemiCathodeThickness"      

      **OUTER SURFACE OF CATHODE COINCIDES WITH VACUUM/PYREX INTERFACE* 
        

    lvPmtHemiBottom        (OpaqueVacuum) partial spherical shell

          outerRadius="PmtHemiBellyROCvac"
          innerRadius="PmtHemiBellyROCvac-1*mm"

      **OUTER SURFACE OF BOTTOM COINCIDES WITH VACUUM/PYREX INTERFACE* 
           



    lvPmtHemiDynode        (OpaqueVacuum) tubs

          outerRadius="PmtHemiDynodeRadius" 


            PmtHemiDynodeRadius : 27.500000 



 PmtHemiBellyCathodeAngleDelta : 26.735889 
  PmtHemiBellyCathodeAngleStart : 55.718631 
           PmtHemiBellyIntAngle : 82.454520 
                PmtHemiBellyOff : 13.000000 
         PmtHemiCathodeDiameter : 190.000000 
        PmtHemiFaceCathodeAngle : 40.504998 
                 PmtHemiFaceOff : 56.000000 

              PmtHemiFaceTopOff : 55.046512 
         PmtHemiGlassBaseLength : 169.000000 
         PmtHemiGlassBaseRadius : 42.250000 


Partitioned Boundary Translation
---------------------------------

Cathode Z region, 3 boundary spherical parts::

    MineralOil///Pyrex
    Pyrex/lvPmtHemiCathodeSensorSurface//Bialkali
    Bialkali///Vacuum 

    * different from literal translation:

      MineralOil///Pyrex
      Pyrex///Vacuum                                     <-- coincident
      Vacuum/lvPmtHemiCathodeSensorSurface//Bialkali     <-- coincident
      Bialkali///Vacuum 

Bottom Z region, 3 boundary spherical parts::

    MineralOil///Pyrex
    Pyrex///OpaqueVacuum  
    OpaqueVacuum///Vacuum 

    * different from literal translation (with zero size slice Vacuum)

      MineralOil///Pyrex
      Pyrex///Vacuum                <--- coincident
      Vacuum///OpaqueVacuum         <--- coincident
      OpaqueVacuum///Vacuum 

    * Bottom OpaqueVacuum is 1mm thick, but Cathode is 0.05mm stuck to inside of Pyrex
      so 0.95mm of protuding OpaqueVacuum : what will happen to photons hitting that 
      protuberance ...  
      BUT the Cathode in "std" model absorbs/detects all photons that hit it, so 
      probably do not need to worry much about loose edges inside ?


Dynode Z region, 3 boundary tubs parts::

    MineralOil///Pyrex
    Pyrex///Vacuum
    Vacuum///OpaqueVacuum
 
    * dynode tubs overlaps the bottom spherical shell



What about joining up the Z regions ? 

* Does the BBox approach auto-close open ends ? need to shoot lots of photons and see..

* MineralOil///Pyrex is no problem, as Z splits chosen for contiguity 
 


Implementation
----------------

* Surface model identities must diverge from Triangulated due to translation differences
  so need to label the parts with these boundaries  

Original direct translation::

    Part Sphere        Pyrex    pmt-hemi-bot-glass_part_zleft    [0, 0, 69.0] r: 102.0 sz:  0.0 bb:BBox      [-101.17 -101.17  -23.84]      [ 101.17  101.17   56.  ] xyz [  0.     0.    16.08]
    Part Sphere        Pyrex  pmt-hemi-top-glass_part_zmiddle    [0, 0, 43.0] r: 102.0 sz:  0.0 bb:BBox      [-101.17 -101.17   56.  ]      [ 101.17  101.17  100.07] xyz [  0.     0.    78.03]
    Part Sphere        Pyrex  pmt-hemi-face-glass_part_zright       [0, 0, 0] r: 131.0 sz:  0.0 bb:BBox      [ -84.54  -84.54  100.07]      [  84.54   84.54  131.  ] xyz [   0.      0.    115.53]
    Part   Tubs        Pyrex               pmt-hemi-base_part   [0, 0, -84.5] r: 42.25 sz:169.0 bb:BBox      [ -42.25  -42.25 -169.  ]         [ 42.25  42.25 -23.84] xyz [  0.     0.   -96.42]

    Part Sphere       Vacuum    pmt-hemi-face-vac_part_zright       [0, 0, 0] r: 128.0 sz:  0.0 bb:BBox         [-82.29 -82.29  98.05]      [  82.29   82.29  128.  ] xyz [   0.      0.    113.02]
    Part Sphere       Vacuum    pmt-hemi-top-vac_part_zmiddle    [0, 0, 43.0] r:  99.0 sz:  0.0 bb:BBox         [-98.14 -98.14  56.  ]         [ 98.14  98.14  98.05] xyz [  0.     0.    77.02]
    Part Sphere       Vacuum      pmt-hemi-bot-vac_part_zleft    [0, 0, 69.0] r:  99.0 sz:  0.0 bb:BBox         [-98.14 -98.14 -21.89]         [ 98.14  98.14  56.  ] xyz [  0.     0.    17.06]
    Part   Tubs       Vacuum           pmt-hemi-base-vac_part   [0, 0, -81.5] r: 39.25 sz:166.0 bb:BBox      [ -39.25  -39.25 -164.5 ]         [ 39.25  39.25 -21.89] xyz [  0.     0.   -93.19]

    Part Sphere     Bialkali       pmt-hemi-cathode-face_part       [0, 0, 0] r: 128.0 sz:  0.0 bb:BBox         [-82.29 -82.29  98.05]      [  82.29   82.29  128.  ] xyz [   0.      0.    113.02]
    Part Sphere     Bialkali       pmt-hemi-cathode-face_part       [0, 0, 0] r:127.95 sz:  0.0 bb:BBox         [-82.25 -82.25  98.01]      [  82.25   82.25  127.95] xyz [   0.      0.    112.98]
    Part Sphere     Bialkali      pmt-hemi-cathode-belly_part    [0, 0, 43.0] r:  99.0 sz:  0.0 bb:BBox         [-98.14 -98.14  56.  ]         [ 98.14  98.14  98.05] xyz [  0.     0.    77.02]
    Part Sphere     Bialkali      pmt-hemi-cathode-belly_part    [0, 0, 43.0] r: 98.95 sz:  0.0 bb:BBox         [-98.09 -98.09  55.99]         [ 98.09  98.09  98.01] xyz [  0.   0.  77.]

    Part Sphere OpaqueVacuum                pmt-hemi-bot_part    [0, 0, 69.0] r:  99.0 sz:  0.0 bb:BBox         [-98.14 -98.14 -30.  ]         [ 98.14  98.14  56.  ] xyz [  0.   0.  13.]
    Part Sphere OpaqueVacuum                pmt-hemi-bot_part    [0, 0, 69.0] r:  98.0 sz:  0.0 bb:BBox         [-97.15 -97.15 -29.  ]         [ 97.15  97.15  56.13] xyz [  0.     0.    13.57]
    Part   Tubs OpaqueVacuum             pmt-hemi-dynode_part   [0, 0, -81.5] r:  27.5 sz:166.0 bb:BBox         [ -27.5  -27.5 -164.5]            [ 27.5  27.5   1.5] xyz [  0.    0.  -81.5]


With coincident surface removal and boundary name rejig and persisting as bndspec list GPmt.txt::

    Part Sphere        Pyrex    pmt-hemi-bot-glass_part_zleft    [0, 0, 69.0] r: 102.0 sz:  0.0 BB      [-101.17 -101.17  -23.84]      [ 101.17  101.17   56.  ] z  16.08 MineralOil///Pyrex
    Part Sphere        Pyrex  pmt-hemi-top-glass_part_zmiddle    [0, 0, 43.0] r: 102.0 sz:  0.0 BB      [-101.17 -101.17   56.  ]      [ 101.17  101.17  100.07] z  78.03 MineralOil///Pyrex
    Part Sphere        Pyrex  pmt-hemi-face-glass_part_zright       [0, 0, 0] r: 131.0 sz:  0.0 BB      [ -84.54  -84.54  100.07]      [  84.54   84.54  131.  ] z 115.53 MineralOil///Pyrex
    Part   Tubs        Pyrex               pmt-hemi-base_part   [0, 0, -84.5] r: 42.25 sz:169.0 BB      [ -42.25  -42.25 -169.  ]         [ 42.25  42.25 -23.84] z -96.42 MineralOil///Pyrex
    Part Sphere       Vacuum      pmt-hemi-bot-vac_part_zleft    [0, 0, 69.0] r:  99.0 sz:  0.0 BB         [-98.14 -98.14 -21.89]         [ 98.14  98.14  56.  ] z  17.06 Pyrex///OpaqueVacuum
    Part Sphere       Vacuum    pmt-hemi-top-vac_part_zmiddle    [0, 0, 43.0] r:  99.0 sz:  0.0 BB         [-98.14 -98.14  56.  ]         [ 98.14  98.14  98.05] z  77.02 Pyrex/lvPmtHemiCathodeSensorSurface//Bialkali
    Part Sphere       Vacuum    pmt-hemi-face-vac_part_zright       [0, 0, 0] r: 128.0 sz:  0.0 BB         [-82.29 -82.29  98.05]      [  82.29   82.29  128.  ] z 113.02 Pyrex/lvPmtHemiCathodeSensorSurface//Bialkali
    Part   Tubs       Vacuum           pmt-hemi-base-vac_part   [0, 0, -81.5] r: 39.25 sz:166.0 BB      [ -39.25  -39.25 -164.5 ]         [ 39.25  39.25 -21.89] z -93.19 Pyrex///Vacuum
    Part Sphere     Bialkali       pmt-hemi-cathode-face_part       [0, 0, 0] r:127.95 sz:  0.0 BB         [-82.25 -82.25  98.01]      [  82.25   82.25  127.95] z 112.98 Bialkali///Vacuum
    Part Sphere     Bialkali      pmt-hemi-cathode-belly_part    [0, 0, 43.0] r: 98.95 sz:  0.0 BB         [-98.09 -98.09  55.99]         [ 98.09  98.09  98.01] z  77.00 Bialkali///Vacuum
    Part Sphere OpaqueVacuum                pmt-hemi-bot_part    [0, 0, 69.0] r:  98.0 sz:  0.0 BB         [-97.15 -97.15 -29.  ]         [ 97.15  97.15  56.13] z  13.57 OpaqueVacuum///Vacuum
    Part   Tubs OpaqueVacuum             pmt-hemi-dynode_part   [0, 0, -81.5] r:  27.5 sz:166.0 BB         [ -27.5  -27.5 -164.5]            [ 27.5  27.5   1.5] z -81.50 Vacuum///OpaqueVacuum
    [tree.py +214                 save ] saving to $IDPATH/GPmt/0/GPmt.npy shape (12, 4, 4) 
    [tree.py +217                 save ] saving boundaries to /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPmt/0/GPmt.txt 





FUNCTIONS
-----------


*pmt-dd*
     test detdesc parsing 

*pmt-parts*
     convert detdesc hemi-pmt.xml into parts buffer $IDPATH/GPmt/0/GPmt.npy 
     using a direct translation approach 

*pmt-analytic*
     convert detdesc hemi-pmt.xml into parts buffer $IDPATH/GPmt/0/GPmt.npy 
     using more nuanced translation better suited to surface geometry lingo 

EOU
}
pmt-dir(){ echo $(local-base)/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT ; }
pmt-edir(){ echo $(env-home)/nuwa/detdesc/pmt ; }
pmt-export(){  
    export PMT_DIR=$(pmt-dir) 
}

pmt-cd(){  cd $(pmt-dir); }
pmt-ecd(){ cd $(pmt-edir) ; }

pmt-xml(){ vi $(pmt-dir)/hemi-pmt.xml ; }


pmt-i(){
   pmt-ecd
   i
}

pmt-run(){ 
   pmt-export
   python $(pmt-edir)/${1:-pmt}.py  
}

pmt-dd(){    pmt-run dd ;}

pmt-parts(){ 
   pmt-export
   python $(pmt-edir)/tree.py $*  
}

pmt-analytic(){ 
   pmt-export
   python $(pmt-edir)/analytic.py $*  
}

pmt-csg(){ 
   pmt-export
   python $(pmt-edir)/csg.py $*  
}




