# === func-gen- : nuwa/detdesc/pmt/pmt fgp nuwa/detdesc/pmt/pmt.bash fgn pmt fgh nuwa/detdesc/pmt
pmt-src(){      echo nuwa/detdesc/pmt/pmt.bash ; }
pmt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pmt-src)} ; }
pmt-vi(){       vi $(pmt-source) ; }
pmt-env(){      elocal- ; }
pmt-usage(){ cat << EOU



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



