# === func-gen- : geant4/g4op/g4op fgp geant4/g4op/g4op.bash fgn g4op fgh geant4/g4op
g4op-src(){      echo geant4/g4op/g4op.bash ; }
g4op-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4op-src)} ; }
g4op-vi(){       vi $(g4op-source) ; }
g4op-env(){      elocal- ; }
g4op-news(){ open http://hypernews.slac.stanford.edu/HyperNews/geant4/get/opticalphotons.html ; }
g4op-eoe(){ open ~/opticks_refs/fresnel-eoe.pdf ; }

g4op-usage(){ cat << EOU


G4 Optical Notes
==================

Objectives
----------

#. find place in G4 to capture photon steps
   analogous to simple chroma stepping loop  

   * multiple processes are relevant to optical photons
     prefer to impinge at a single point to capture steps 

::

    delta:include blyth$ grep class\ G4Op *.hh 
    G4OpAbsorption.hh:class G4OpAbsorption : public G4VDiscreteProcess 
    G4OpBoundaryProcess.hh:class G4OpBoundaryProcess : public G4VDiscreteProcess
    G4OpMieHG.hh:class G4OpMieHG : public G4VDiscreteProcess
    G4OpRayleigh.hh:class G4OpRayleigh : public G4VDiscreteProcess 
    G4OpWLS.hh:class G4OpWLS : public G4VDiscreteProcess 
    delta:include blyth$ 


Geant4 Optical Photon Hypernews
-------------------------------------

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/opticalphotons.html?


Search : optical photon simulation
-------------------------------------

* :google:`optical photon simulation`


Peculiarities of G4 Optical Photon Simulation
-----------------------------------------------

* https://arxiv.org/pdf/1612.05162.pdf
* ~/opticks_refs/Peculiarities_G4OpticalPhoton_1612.05162.pdf

p1: Additionally, all optical properties have to be specified on the same energy
range. Otherwise errors will occur, e.g. if a scintillation photon is created
with an energy for which no refractive index is specified in the material.

p3: Geant4 itself does not limit the emission spectrum of the Cherenkov process to
reasonable energy ranges but creates Cherenkov photons distributed over the
full energy range for which n is specified (and the Cherenkov criterion is
fulfilled). Thus, simulated Cherenkov radiation can occur in non-physical
energy regions, e.g. in the keV, MeV or TeV range, instead of in the optical
range. Therefore, the refractive index should be restricted to the necessary
optical energy range or the physical Cherenkov energy range, respectively. As
explained before, all other optical properties should be restricted to the same
energy range, since defining individual optical properties on different energy
ranges will most probably cause problems.

* http://publications.rwth-aachen.de/record/667646
* ~/opticks_refs/g4_optical_peculiarities_thesis_667646.pdf

Dietz-Laursonn, Erik* ; Hebbeker, Thomas (Thesis advisor)* ; Pooth, Oliver (Thesis advisor)

Detailed studies of light transport in optical components of particle detectors







G4 Classes
-----------

::

   g4-;g4-cls G4OpBoundaryProcess


Fresnel References
-------------------

Very detailed slides, well presented, lots of derivations, 31 pages. 

* http://www.patarnott.com/atms749/pdf/FresnelEquations.pdf

* ~/opticks_refs/patarnott_FresnelEquations.pdf

* http://www.patarnott.com/atms749/index.html
* https://scienceworld.wolfram.com/physics/BrewstersAngle.html
* https://scienceworld.wolfram.com/physics/FresnelEquations.html


G4 Optical Photon Reference 
-------------------------------

* http://geant4.slac.stanford.edu/UsersWorkshop/PDF/Peter/OpticalPhoton.pdf

Polarization Sources
---------------------

* http://fp.optics.arizona.edu/chipman/Publications/Publications.html


Others
-------

Photon propagation code

* http://icecube.wisc.edu/~dima/work/WISC/ppc/


Chroma propagation pseudo-code
---------------------------------

* easier to collect steps in "chroma" structure in a manner 
  that corresponds to "g4" than vv 

* G4SteppingManager::InvokePSDIP sees all changes from all processes 

::

    529          fParticleChange
    530             = fCurrentProcess->PostStepDoIt( *fTrack, *fStep);


Summarized /usr/local/env/chroma_env/src/chroma/chroma/cuda/propagate_vbo.cu::


    Photon p ;
    State s ; 
    Geometry g ;

    while(steps < max_steps)
    {
        steps++ ;

        fill_state(p, s, g)                   // mesh_intersect to set p.last_hit_triangle and fill material props in state
        if(p.last_hit_triangle == -1) break 

        ///// collect initial (generated) and  CONTINUE steps here 
        ///// do here as need to fill_state after changing p.direction 
        ///// to get last_hit_triangle intersection and state update

        cmd = propagate_to_boundary(p, s)    // absorb | scatter/reemit  | sail

        if(cmd == BREAK) break ;            // BULK_ABSORB[G4OpAbsorption]   ->
        if(cmd == CONTINUE) continue  ;     // BULK_REEMIT[DetSim handled by adding single 2ndary to the change] + RAYLEIGH_SCATTER[G4OpRayleigh]
        if(cmd == PASS) noop ;              // survivors pass to boundary 
        

        if(s.surface_index != -1)           //  [G4OpBoundaryProcess]
        {
            cmd = propagate_at_surface(p, s, g)
            if(cmd == BREAK) break ;        // SURFACE_ABSORB SURFACE_DETECT   ->
            if(cmd == CONTINUE) continue  ; // REFLECT_DIFFUSE REFLECT_SPECULAR    .direction .polarization
        }
        else  // 
        {
            propagate_at_boundary(p, s)     // REFLECT_SPECULAR REFRACT   .direction .polarization
            ""continue""
        }

    } 
     
    ///// collect BREAK steps (absorb,detect,truncate) here ... 


propagate_to_boundary
~~~~~~~~~~~~~~~~~~~~~~~~
       
#. absorption: advance .time .position to absorption point, then either:
       
   * BULK_REEMIT(CONTINUE) change .direction .polarization .wavelength
   * BULK_ABSORB(BREAK)    .last_hit_triangle -1   [G4OpAbsorption]
       
#. scattering: advance .time .position to scatter point 

   * RAYLEIGH_SCATTER(CONTINUE) change .direction .polarization  [G4OpRayleigh]
         
#. survive: advance .time .position to boundary  (not absorbed/reemitted/scattered)
      
   * PASS


drawing parallels with g4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* where is the G4 decision to absorb(reemit)/scatter/survive-to-boundary ?  
   
  * G4SteppingManager::DefinePhysicalStepLength picks PostStepDoIt process with smallest random scaled down distance
  * G4SteppingManager::InvokePostStepDoItProcs invokes the PostStepDoIt which makes the change 
    
* reemission handled rather differently 
    
  * just another step for Chroma
  * 2nd-ary for G4 


g4 reemission 2ndary, accessible from G4Step ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    aParticleChange.SetNumberOfSecondaries(NumTracks);
    aParticleChange.AddSecondary(aSecondaryTrack);

    G4SteppingManager::Invoke*DoItProcs passes secondaries on to fSecondary 

Maybe::

     const std::vector<const G4Track*>* G4Step::GetSecondaryInCurrentStep() const


source/track/include/G4VParticleChange.hh::

    191     G4int GetNumberOfSecondaries() const;
    192     //  Returns the number of secondaries current stored in
    193     //  G4TrackFastVector.
    194 
    195     G4Track* GetSecondary(G4int anIndex) const;
    196     //  Returns the pointer to the generated secondary particle
    197     //  which is specified by an Index.




vector treatment of fresnel eqns with polarization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.classe.cornell.edu/~ib38/tmp/reading/syn_rad/wigner_dist/polarized_light/DKE498_ch8.pdf


* http://old.rqc.ru/quantech/pubs/2013/fresnel-eoe.pdf
* ~/opticks_refs/fresnel-eoe.pdf

* http://en.wikipedia.org/wiki/Mueller_calculus
* http://en.wikipedia.org/wiki/Stokes_parameters


* https://scholar.harvard.edu/files/schwartz/files/lecture14-polarization.pdf


:google:`refraction polarization vector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://users.hep.manchester.ac.uk/u/xiaguo/waveoptics/Polarisation_supplement.pdf

Brewster angle (or polarizing angle): when angle between reflected and
refracted (transmitted) light is 90deg, the reflected light is 100% linearly polarized
perpendicular to the plane of incidence [Sir David Brewster, 1781-1868


For light originating in air, Brewster angle given by tan -1 (refractive index of
reflecting medium); ~56deg for air-glass interface; ~53deg for air-water interface

::

    In [4]: np.arctan([1.5,1.333])/np.pi*180
    Out[4]: array([56.31 , 53.123])


looking for efficiency tricks in implementing fresnel eqns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :google:`ray trace polarization fresnel`

* http://www.patarnott.com/atms749/pdf/RayTracingAbsorbingMedia.pdf

* http://web.cs.wpi.edu/~emmanuel/courses/cs563/S10/final_projects/steve_olivieri/

  * rendering diamonds


* :google:`cuda fresnel raytrace`

  * https://github.com/wuhao1117/CUDA-Path-Tracer

* :google:`fresnel schlick`

  * https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/


* http://www.theseus-fe.com/images/downloads/publications/optics-glass-transmission.pdf

* http://kylehalladay.com/blog/tutorial/2014/02/18/Fresnel-Shaders-From-The-Ground-Up.html

* http://renderman.pixar.com/view/cook-torrance-shader


First, the Fresnel term, F, is the simplest to implement in SL, because there is a built-in function for calculating it. 
From the RSL Docs:

    void fresnel( vector I, N; float eta, Kr, Kt [; output vector R, T] )

Return the reflection coefficient Kr and refraction (or transmission) coefficient Kt 
given:

* an incident direction I, 
* the surface normal N, 
* and the relative index of refraction eta. 
  Eta is the ratio of the index of refraction in the volume containing the incident vector to that of the volume being entered. 

These coefficients are computed using the Fresnel formula. 
Optionally, this procedure also returns the reflected (R) and transmitted (T) vectors. 
The transmitted vector is computed using Snell's law. 
The angle between I and N is supposed to be larger than 90 degrees. 
If the angle is less than 90 degrees, fresnel will return full reflection (Kr = 1, Kt = 0).


* http://renderman.pixar.com/resources/RPS_17/search.html?q=fresnel
* http://renderman.pixar.com/resources/RPS_17/rslFunctions.html#fresnel


* http://http.developer.nvidia.com/GPUGems/gpugems_ch02.html

  * rendering sea bed caustics  


optix examples
~~~~~~~~~~~~~~~~

All using Schlick approximation::

    simon:OptiX_370b2_sdk blyth$ find . -name '*.cu' -exec grep -Hil fresnel {} \;
    ./cook/clearcoat.cu
    ./glass/glass.cu
    ./isgReflections/glossy_isg.cu
    ./julia/julia.cu
    ./mcmc_sampler/mcmc_sampler.cu
    ./ocean/ocean_render.cu
    ./rayDifferentials/glass_mip.cu
    ./transparency/transparent.cu
    ./tutorial/tutorial10.cu
    ./tutorial/tutorial11.cu
    ./tutorial/tutorial6.cu
    ./tutorial/tutorial7.cu
    ./tutorial/tutorial8.cu
    ./tutorial/tutorial9.cu
    ./whirligig/glass.cu
    ./whitted/glass.cu


feynmann lecture on light
~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.youtube.com/watch?v=UUAG3yDInQQ&feature=youtu.be

* https://www.youtube.com/watch?v=UUAG3yDInQQ

Richard Feynman QED Lecture 2, Reflection and Transmission : 1/7

I like the description about 7:45 of Feynman_youtube lecture_on_light where
Feynman talks about something that 'follows the particle along changing its
disposition to do things' - ie, it reflects or refracts depending on the phase.

* https://physics.stackexchange.com/questions/76095/calculate-the-polarization-vector-on-reflection-or-refraction-from-a-dielectric





fresnel eqn
~~~~~~~~~~~~

* tangential components of electric E and magnetic H fields are 
  continuous across the boundary 

* transverse EM wave can be decomposed into 

  * P-polarized: E field vector inside plane of incidence
  * S-polarized: E field vector orthogonal to plane of incidence 
  * plane of incidence is the defined by the wave direction vector
    and normal to the interface [not defined at normal incidence]


source/processes/optical/src/G4OpBoundaryProcess.cc (yuck what a tabbing mess)::

     881 void G4OpBoundaryProcess::DielectricDielectric()
     882 {
     ...
     886     leap:
     ...
     891     do {
     ...
     909        G4double PdotN = OldMomentum * theFacetNormal;
     910        G4double EdotN = OldPolarization * theFacetNormal;
     911 
     912        cost1 = - PdotN;

     ///    
     ///    cf propagate_at_boundary_geant4_style
     ///       
     ///         const float c1 = -dot(p.direction, s.surface_normal ); // c1 arranged to be +ve  
     /// 
     ///         will be +1 at normal incidence 
     ///
     ///

     ...
     914           sint1 = std::sqrt(1.-cost1*cost1);
     915           sint2 = sint1*Rindex1/Rindex2;     // *** Snells Law ***
     ...
     922        if (sint2 >= 1.0) {         // TIR
     926           if (Swap) Swap = !Swap;
     928           theStatus = TotalInternalReflection;
     929 
     930           if ( theModel == unified && theFinish != polished )
     931                              ChooseReflection();
     932 
     933           if ( theStatus == LambertianReflection ) {
     934                  DoReflection();
     935           }
     936           else if ( theStatus == BackScattering ) {
     937                 NewMomentum = -OldMomentum;
     938                 NewPolarization = -OldPolarization;
     939           }
     940           else {   // specular TIR
     942                  PdotN = OldMomentum * theFacetNormal;
     943                  NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
     944                  EdotN = OldPolarization * theFacetNormal;
     945                  NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
     947           }
     948        }
     949        else if (sint2 < 1.0) {     // Calculate amplitude for transmission (Q = P x N)
     952 
     953           if (cost1 > 0.0) {
     954              cost2 =  std::sqrt(1.-sint2*sint2);
     955           }
     956           else {
     957              cost2 = -std::sqrt(1.-sint2*sint2);
     958           }
     959 
     960           G4ThreeVector A_trans, A_paral, E1pp, E1pl;
     961           G4double E1_perp, E1_parl;
     962 
     963           if (sint1 > 0.0) {
     964              A_trans = OldMomentum.cross(theFacetNormal); 
     965              A_trans = A_trans.unit();                    // vector perpendicular to plane of incidence 
     966              E1_perp = OldPolarization * A_trans;         // fraction of pol transverse to plane of incidence
                                                                   // polz is normalized?

     ///   http://en.wikipedia.org/wiki/Linear_polarization
     ///   hmm regarding polarization vector to correspond to the E field direction
     ///

     967              E1pp    = E1_perp * A_trans;                 // component of OldPolarization perpendicular to plane of incidence
     968              E1pl    = OldPolarization - E1pp;            // component of OldPolarization within the plane of incidence   
     969              E1_parl = E1pl.mag();

     ///
     ///               E1_perp and E2_parl are fractions of s and p ? based on OldPolarization
     ///

     970               }
     971           else {

     ////
     ////     hmm no epsilon, G4 expecting sint1 to be precisely zero(double) for normal incidence
     ////
     ////

     972              A_trans  = OldPolarization;
     973              // Here we Follow Jackson's conventions and we set the
     974              // parallel component = 1 in case of a ray perpendicular
     975              // to the surface
     976              E1_perp  = 0.0;
     977              E1_parl  = 1.0;
     978           }
     979 
     980               G4double s1 = Rindex1*cost1;
     981               G4double E2_perp = 2.*s1*E1_perp/(Rindex1*cost1+Rindex2*cost2);  // (13) S-polarized  t_s * E1_perp
     982               G4double E2_parl = 2.*s1*E1_parl/(Rindex2*cost1+Rindex1*cost2);  // (8)  P-polarized  t_p * E1_parl
     ///
     ///               fresnel eqns : using the boundary conditions on field to relate transmitted to incident amplitudes
     ///               separately for s and p polarizations  
     ///
     983               G4double E2_total = E2_perp*E2_perp + E2_parl*E2_parl;         // square up s and p amplitudes to get overall intensity
     984               G4double s2 = Rindex2*cost2*E2_total;   //  is this the planar angle term    (24)
     985 
     986               G4double TransCoeff;
     987 
     988               if (theTransmittance > 0) TransCoeff = theTransmittance;
     989               else if (cost1 != 0.0) TransCoeff = s2/s1;     //  transmission probability  "Transmittance = 1 - Reflectance"
     990               else TransCoeff = 0.0;

     ///   fresnel-eoe.pdf
     ///       ...the intensity is calculated per unit of the wavefront area, and the wavefronts of the incident 
     ///       and transmitted wave are tilted with respect to the interface at different angles theta_i and theta_t, respectively. 
     ///       Therefore, the intensity transmissivity is given by (24)
     ///
     ///
     ///                         n2 cost2 |Et|^2        n2 cost2
     ///                   T = ------------------- =   ---------- |t|^2
     ///                         n1 cost1 |Ei|^2        n1 cost1 
     ///
     ...
     992           G4double E2_abs, C_parl, C_perp;
     993 
     994           if ( !G4BooleanRand(TransCoeff) ) {   // not transmission, so reflection
     998                  if (Swap) Swap = !Swap;
    1000                  theStatus = FresnelReflection;
    1002                  if ( theModel == unified && theFinish != polished )
    1003                                 ChooseReflection();
    1004 
    1005                  if ( theStatus == LambertianReflection ) {
    1006                      DoReflection();
    1007                  }
    1008                  else if ( theStatus == BackScattering ) {
    1009                      NewMomentum = -OldMomentum;
    1010                      NewPolarization = -OldPolarization;
    1011                  }
    1012                  else {  // specular reflection 
    1014                      PdotN = OldMomentum * theFacetNormal;
    1015                      NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
    1016 
    1017                      if (sint1 > 0.0) {   // incident ray oblique
    1018 
    1019                          E2_parl   = Rindex2*E2_parl/Rindex1 - E1_parl;
    ///
    ////
    ////        (P-polarization: E field within plane of incidence) 
    ////         magnetic field continuity at boundary (sign from triad convention wrt field directions)
    ////
    ////               Hi - Hr = Ht        eoe[3] 
    ////         n1 (Ei - Er ) = n2 Et             relating to E brings in material characteristics  eoe[6] 
    ////
    ////                  Er = (n2 Et/n1) - Ei 
    ////
    ////
    1020                          E2_perp   = E2_perp - E1_perp;
    ////
    ////                                    t_s * E1_perp  - E1_perp
    ////
    ////               for perp (S polarization) with E field perpendicular to plane of incidence
    ////                            Ei + Er = Et ==>   Er = Et - Ei 
    ////                         
    ////                amplitude transmission and reflection coefficients
    ////                          t = Et/Ei
    ////                          r = Er/Ei               (T + R = 1 for intensities, not amplitudes)
    ////
    1021                          E2_total  = E2_perp*E2_perp + E2_parl*E2_parl;
    ////    
    1022                          A_paral   = NewMomentum.cross(A_trans);
    1023                          A_paral   = A_paral.unit();
    ////
    1024                          E2_abs    = std::sqrt(E2_total);
    1025                          C_parl    = E2_parl/E2_abs;
    1026                          C_perp    = E2_perp/E2_abs;
    1027 
    1028                          NewPolarization = C_parl*A_paral + C_perp*A_trans;
    ////
    ////                A_trans : normal to incident plane   (E field direction for S-polarization)
    ////                A_paral : 
    ////           
    1029 
    1030                      }
    1032                      else {               // incident ray perpendicular
    ////                           normal incidence
    1034                          if (Rindex2 > Rindex1) {
    1035                               NewPolarization = - OldPolarization;
    1036                          }
    1037                          else {
    1038                               NewPolarization =   OldPolarization;
    1039                          }
    1041                      }
    1042              }
    1043           }
    1044           else { // photon gets transmitted
    1045 
    1046              // Simulate transmission/refraction
    1047 
    1048              Inside = !Inside;
    1049              Through = true;
    1050              theStatus = FresnelRefraction;
    1051 
    1052              if (sint1 > 0.0) {      // incident ray oblique
    1053 
    1054                    G4double alpha = cost1 - cost2*(Rindex2/Rindex1);
    ////
    ////                    Bram de Greve (22) : can avoid the normalization using:
    ////
    ////                     t = n1/n2  i^  +  ( n1/n2 cost1 - cost2 ) n^    
    ////
    ////
    1055                    NewMomentum = OldMomentum + alpha*theFacetNormal;
    1056                    NewMomentum = NewMomentum.unit();
    1057                    PdotN = -cost2;
    1058                    A_paral = NewMomentum.cross(A_trans);
    1059                    A_paral = A_paral.unit();
    1060                    E2_abs  = std::sqrt(E2_total);
    1061                    C_parl  = E2_parl/E2_abs;
    1062                    C_perp  = E2_perp/E2_abs;
    1063 
    1064                    NewPolarization = C_parl*A_paral + C_perp*A_trans;
    1065 
    1066              }
    1067              else {                  // incident ray perpendicular
    1069                    NewMomentum = OldMomentum;
    1070                    NewPolarization = OldPolarization;
    1072              }
    1073           }
    1074        }
    1075 
    1076        OldMomentum = NewMomentum.unit();
    1077        OldPolarization = NewPolarization.unit();
    1078 
    1079        if (theStatus == FresnelRefraction) {
    1080           Done = (NewMomentum * theGlobalNormal <= 0.0);   // expecting opposite hemi to theGlobalNormal for refraction
    1081        }
    1082        else {
    1083           Done = (NewMomentum * theGlobalNormal >= 0.0);   // same hemi for reflection
    1084        }
    1085 
    1086     } while (!Done);
    1087 
    1088     if (Inside && !Swap) {
    1089           if( theFinish == polishedbackpainted ||
    1090               theFinish == groundbackpainted ) {
    1091 
    1092           if( !G4BooleanRand(theReflectivity) ) {
    1093                  DoAbsorption();
    1094           }
    1095           else {
    1096               if (theStatus != FresnelRefraction ) {
    1097                   theGlobalNormal = -theGlobalNormal;
    1098               }
    1099               else {
    1100                   Swap = !Swap;
    1101                   G4SwapPtr(Material1,Material2);
    1102                   G4SwapObj(&Rindex1,&Rindex2);
    1103               }
    1104               if ( theFinish == groundbackpainted )
    1105                     theStatus = LambertianReflection;
    1106 
    1107               DoReflection();
    1108 
    1109               theGlobalNormal = -theGlobalNormal;
    1110               OldMomentum = NewMomentum;
    1111 
    1112               goto leap;
    1113           }
    1114       }   // handling backpainted with the leap
    1115     }
    1116 }



G4OpBoundaryProcess look again : in light of TMM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    1041 void G4OpBoundaryProcess::DielectricDielectric()
    1042 {

    1086            PdotN = OldMomentum * theFacetNormal;
    1087            EdotN = OldPolarization * theFacetNormal;

    ////            components of mom and pol in normal direction 


    1088 
    1089            cost1 = - PdotN;
    1090            if (std::abs(cost1) < 1.0-kCarTolerance){
    1091               sint1 = std::sqrt(1.-cost1*cost1);
    1092               sint2 = sint1*Rindex1/Rindex2;     // *** Snell's Law ***
    1093            }
    1094            else {      
    ////
    ////            Handle Normal incidence where cost1 is close to -1 or 1 
    ////
    1095               sint1 = 0.0;
    1096               sint2 = 0.0;
    1097            }

    1099            if (sint2 >= 1.0) {
    1100 
    1101               // Simulate total internal reflection
    1104 
    1105               theStatus = TotalInternalReflection;
    1106 
    1113               if ( theStatus == LambertianReflection ) {
    1114                  DoReflection();
    1115               }
    1116               else if ( theStatus == BackScattering ) {
    1117                  NewMomentum = -OldMomentum;
    1118                  NewPolarization = -OldPolarization;
    1119               }
    1120               else {
    1121 
    1122                  PdotN = OldMomentum * theFacetNormal;
    1123                  NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;

           
    //         \   / 
    //          \ /
    //       ----+------
    //             
                

    1124                  EdotN = OldPolarization * theFacetNormal;
    1125                  NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;

     //        JUSTIFICATION ?


    1126 
    1127               }
    1128            }


    1129            else if (sint2 < 1.0) {
    1130 
    1131               // Calculate amplitude for transmission (Q = P x N)
    1132 
    1133               if (cost1 > 0.0) {
    1134                  cost2 =  std::sqrt(1.-sint2*sint2);
    1135               }
    1136               else {
    1137                  cost2 = -std::sqrt(1.-sint2*sint2);
    1138               }

    ////            
    ////      HUH: cost1 = -PdotN 
    ////      handing -ve cost1 suggests the normal is not oriented against the incident direction ?
    ////                             


    1139 
    1140               if (sint1 > 0.0) {
    1141                  A_trans = OldMomentum.cross(theFacetNormal);
    1142                  A_trans = A_trans.unit();

    ////       A_trans : Transverse unit vector : perpendicular to the plane of incidence 

    1143                  E1_perp = OldPolarization * A_trans;

    ////       E1_perp scalar : fraction of polarization perpendicular to plane of incidence (S-polarized fraction) 

    1144                  E1pp    = E1_perp * A_trans;

    ////       E1pp vector : perpendicular component of OldPolarization (in A_trans direction, the S-polarized component)

    1145                  E1pl    = OldPolarization - E1pp;

    ////        E1pl vector : parallel component of OldPolarization (within plane of incidence, the P-polarized component) 

    1146                  E1_parl = E1pl.mag();

    ////        E1_parl scalar : fraction of polarization within plane of incidence (P-polarized fraction)      


    1147               }
    1148               else {
    1149                  A_trans  = OldPolarization;
    1150                  // Here we Follow Jackson's conventions and we set the
    1151                  // parallel component = 1 in case of a ray perpendicular
    1152                  // to the surface
    1153                  E1_perp  = 0.0;
    1154                  E1_parl  = 1.0;
    1155               }

    ////         Above handles of normal incidence where the distinction between S and P is meaningless


    //// eoe eqn references below are to 
    ////   http://old.rqc.ru/quantech/pubs/2013/fresnel-eoe.pdf   Encyclopedia of Optical Engineering 
    ////   ~/opticks_refs/fresnel-eoe.pdf 
    ////
    ////  Google for "Encyclopedia of Optical Engineering Lvovsky pdf"
    ////    
    //// Fresnel Equations
    //// Alexander I. Lvovsky
    //// Department of Physics and Astronomy, University of Calgary, Calgary, Alberta, Canada
    //// Published online: 27 Feb 2013
    ////    

    1156 
    1157               s1 = Rindex1*cost1;
    1158               E2_perp = 2.*s1*E1_perp/(Rindex1*cost1+Rindex2*cost2);  

    ////    eoe[13] t_s   : Relate incident and transmitted S pol amplitudes

    1159               E2_parl = 2.*s1*E1_parl/(Rindex2*cost1+Rindex1*cost2);
   
    ////    eoe[8]  t_p   : Relate incident and transmitted P pol amplitudes

    1160               E2_total = E2_perp*E2_perp + E2_parl*E2_parl;
    1161               s2 = Rindex2*cost2*E2_total;
    1162 
    1163               if (theTransmittance > 0) TransCoeff = theTransmittance;
    1164               else if (cost1 != 0.0) TransCoeff = s2/s1;
    1165               else TransCoeff = 0.0;

    ////     eoe[24] T :  intensity transmissivity  


    1166 
    1167               if ( !G4BooleanRand(TransCoeff) ) {
    1168 
    1169                  // Simulate reflection
    1170 
    1171                  if (Swap) Swap = !Swap;
    1172 
    1173                  theStatus = FresnelReflection;
    ....
    1190                     PdotN = OldMomentum * theFacetNormal;
    1191                     NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
    1192 
    1193                     if (sint1 > 0.0) {   // incident ray oblique
    1194 
    1195                        E2_parl   = Rindex2*E2_parl/Rindex1 - E1_parl;
    1196                        E2_perp   = E2_perp - E1_perp;
    1197                        E2_total  = E2_perp*E2_perp + E2_parl*E2_parl;

    //// 
    ////         E1_parl 
    ////             incident P-pol fraction   "Ei" 
    ////
    ////   1195: E2_parl = Rindex2*E2_parl/Rindex1 - E1_parl; 
    ////             here P-pol "Et" is changed to be P-pol  "Er"
    ////             This from eoe[3] magnetic boundary condition, converted to E eoe[6] 
    ////          
    ////   1196: E2_perp (S-pol reflection) = E2_perp (S-pol transmission) - E1_perp (S-pol incident) 
    ////
    ////          eoe[10]   Ei + Er = Et  (S-pol)        
    ////         


    1198                        A_paral   = NewMomentum.cross(A_trans);
    1199                        A_paral   = A_paral.unit();

    ////
    ////          A_trans vector 
    ////              transverse unit vector, perpendicular to plane of incidence, from OldMomentum.cross(facetNormal)
    ////
    ////          A_paral vector
    ////              orthogonal to both the new mom and A_trans 
    ////              so its in the plane of incidence and its transverse to new mom
    ////              making it the new P-polarization vector direction 
    ////                    

    1200                        E2_abs    = std::sqrt(E2_total);
    1201                        C_parl    = E2_parl/E2_abs;
    1202                        C_perp    = E2_perp/E2_abs;
    1203 
    1204                        NewPolarization = C_parl*A_paral + C_perp*A_trans;
 
    ////
    ////        C_parl 
    ////            P-pol fraction in reflected wave
    ////        C_perp 
    ////            S-pol fraction in reflected wave
    ////        NewPolarization
    ////            addition of the S and P fraction vectors
    ////
    ////

    1205 
    1206                     }
    1207 
    1208                     else {               // incident ray perpendicular
    1209 
    1210                        if (Rindex2 > Rindex1) {
    1211                           NewPolarization = - OldPolarization;
    1212                        }
    1213                        else {
    1214                           NewPolarization =   OldPolarization;
    1215                        }
    ....
    1220               else { // photon gets transmitted
    1221 
    1222                 // Simulate transmission/refraction
    1223 
    1224                 Inside = !Inside;
    1225                 Through = true;
    1226                 theStatus = FresnelRefraction;
    1227 
    1228                 if (sint1 > 0.0) {      // incident ray oblique
    1229 
    1230                    alpha = cost1 - cost2*(Rindex2/Rindex1);
    1231                    NewMomentum = OldMomentum + alpha*theFacetNormal;
    1232                    NewMomentum = NewMomentum.unit();

    ////
    ////  cf qsim.h:propagate_at_boundary
    ////     t = eta i  + (eta c1 - c2 ) n      eta = n1/n2 
    ////     t/eta = i + (c1 - c2/eta ) n 
    ////
    ////    Because Geant4 normalizes NewMomentum it gets away with 
    ////    playing fast and loose with factors of 1/eta 
    ////

    1234                    A_paral = NewMomentum.cross(A_trans);
    1235                    A_paral = A_paral.unit();

    ////    A_trans
    ////         OldMomentum.cross(normal) : perpendicular to plane of incidence : old S-pol direction 
    ////
    ////    A_paral
    ////          Transmitted P-pol direction 
    ////
    ////

    1236                    E2_abs     = std::sqrt(E2_total);
    1237                    C_parl     = E2_parl/E2_abs;
    1238                    C_perp     = E2_perp/E2_abs;
    1239 
    1240                    NewPolarization = C_parl*A_paral + C_perp*A_trans;

    ////
    ////     Use S and P fractions to construct the NewPolarization  
    ////
    ////


    1241 
    1242                 }
    1243                 else {                  // incident ray perpendicular
    1244 
    1245                    NewMomentum = OldMomentum;
    1246                    NewPolarization = OldPolarization;
    1247 
    1248                 }
    1249               }
    1250            }




Rigorous vector wave propagation for arbitrary flat media
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matrix methods and fresnel 

* https://arxiv.org/pdf/1811.09777.pdf


G4 tracing reflection ?
~~~~~~~~~~~~~~~~~~~~~~~~~

oxrap-/cu/generate.cu::

    ///
    ///   surface handling is signalled by the index value in optical buffer
    ///   
    ///

    406         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    407         {
    408             command = propagate_at_surface(p, s, rng);
    409             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    410             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT   <-- DR/SR diffuse/specular reflection
    411         }
    412         else
    413         {
    414             //propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    415             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    416             // tacit CONTINUE
    417         }



G4 surface priority order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check the surface logic/priority order::

     314         G4LogicalSurface* Surface = NULL;
     315 
     316         Surface = G4LogicalBorderSurface::GetSurface(thePrePV, thePostPV);
     317 
     318         if (Surface == NULL)
                 {
     319               G4bool enteredDaughter= (thePostPV->GetMotherLogical() ==
     320                                        thePrePV ->GetLogicalVolume());
     ///
     ///

     321               if(enteredDaughter)  
     ///                     stepping inwards
                       {
     322                    Surface =
     323                               G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
     324                    if(Surface == NULL)
     325                          Surface =
     326                                   G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
     327               }
     328               else 
                       {
     329                   Surface =
     330                               G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
     331                   if(Surface == NULL)
     332                         Surface =
     333                                  G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
     334               }
     335         }



G4 Surface docs
~~~~~~~~~~~~~~~~~

* http://geant4-userdoc.web.cern.ch/geant4-userdoc/UsersGuides/ForApplicationDeveloper/BackupVersions/V9.4/html/ch05s02.html#sect.PhysProc.Photo

Surface objects of the second type are stored in a related table and can be
retrieved by either specifying the two ordered pairs of physical volumes
touching at the surface, or by the logical volume entirely surrounded by this
surface. The former is called a border surface while the latter is referred to
as the skin surface. **This second type of surface is useful in situations where
a volume is coded with a reflector and is placed into many different mother
volumes**. A limitation is that the skin surface can only have one and the same
optical property for all of the enclosed volume's sides. The border surface is
an ordered pair of physical volumes, so in principle, the user can choose
different optical properties for photons arriving from the reverse side of the
same interface. For the optical boundary process to use a border surface, the
two volumes must have been positioned with G4PVPlacement. The ordered
combination can exist at many places in the simulation. When the surface
concept is not needed, and a perfectly smooth surface exists beteen two
dielectic materials, the only relevant property is the index of refraction, a
quantity stored with the material, and no restriction exists on how the volumes
were positioned.


Priority order, when stepping inwards (ie motherLV of postPV is prePV_lv) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     ///

     321               if(enteredDaughter)  
     ///                     stepping inwards
                       {
     322                    Surface =
     323                               G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
     324                    if(Surface == NULL)
     325                          Surface =
     326                                   G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
     327               }




1. border surface prePV->postPV  <-- this makes sense
2. skin surface of postPV_LV  (inner surface where are headed)  
3. skin surface of prePV_LV   (outer surface where are coming from)

::

           parent prePV              /
                                    /
                   ~~~~~~~~~~~~~~~~/~~~~~(3)~~~~~~  osur
            ______(1)_____________/_____________________________
                                 /
                   ~~~~~~~~~~~~~/~~~~~(2)~~~~~~~~~~~isur 
           child postPV        /
                             \/_



When stepping outwards:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     328               else 
                       {
     329                   Surface =
     330                               G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
     331                   if(Surface == NULL)
     332                         Surface =
     333                                  G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
     334               }
     335         }


1. border surface prePV->postPV  (opposite pair of volumes to inwards, makes sense)
2. skin surface of prePV_LV     (inner surface where are coming from)   
3. skin surface of postPV_LV    (outer surface where are headed)

::
     
                                     _ 
                                      /\
           parent postPV             /
                                    /
                   ~~~~~~~~~~~~~~~~/~~~~~(3)~~~~~~ osur
            ______(1)_____________/_____________________________
                                 /
                   ~~~~~~~~~~~~~/~~~~~(2)~~~~~~~~~ isur
           child prePV         /
                              /


Without border surface the inner skin has priority over the outer skin, for 
both inwards and outwards going photons.
Only if the inner skin has no surface does the outer skin get a chance.

Given that logical volumes proliferate, its much safer to use a 
border surface to target just the desired volume pairs.

   
Opticks/G4 difference
~~~~~~~~~~~~~~~~~~~~~~~~~

Opticks boundary orientation (ie what inner/outer mean) 
is based on the geometric normal to the surface. The sign of 
boundary index comes from the dot product of the photon direction 
and the outwards pointing normal to the surface.


But G4 pre/post is just from the stepping direction of the photon, 
so maybe causes a double flip(?) brings Opticks and G4.

TODO: experiment with double skin surface interface, 
      to try to exercise this : to see if there really is a difference


All solids have rigidly attached normals pointing outwards


* at a boundary Opticks will use either isur or osur depending on 
  photon direction relative to surface normal, 

  * it doesnt try using one and then the other if the first was not set 
  * it could do this fairly easily to duplicate G4, but it doesnt now
    (the G4 logic needs some clarification) 




Can Opticks boundary model fully translate the Geant4 model ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Opticks boundary: omat/osur/isur/imat 

* outwards (child to parent) going photons use ISUR
* inwards (from parent to child) going photons use OSUR

* ISUR and OSUR can point to the same or different surface, or be unset 

::

     22 enum {  
     23     OMAT,
     24     OSUR,
     25     ISUR,
     26     IMAT 
     27 };
     28 
     29 __device__ void fill_state( State& s, int boundary, uint4 identity, float wavelength )
     30 {       
     31     // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
     32     // >0 outward going photon
     33     // <0 inward going photon
     34     //  
     35     // NB the line is above the details of the payload (ie how many float4 per matsur) 
     36     //    it is just 
     37     //                boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 
     38     //  
     39         
     40     int line = boundary > 0 ? (boundary - 1)*BOUNDARY_NUM_MATSUR : (-boundary - 1)*BOUNDARY_NUM_MATSUR  ;
     41 
     42     // pick relevant lines depening on boundary sign, ie photon direction relative to normal
     43     // 
     44     int m1_line = boundary > 0 ? line + IMAT : line + OMAT ;
     45     int m2_line = boundary > 0 ? line + OMAT : line + IMAT ;   
     46     int su_line = boundary > 0 ? line + ISUR : line + OSUR ;   

     //    outward going photon uses ISUR
     //    inward going photon uses OSUR

     47         
     48     //  consider photons arriving at PMT cathode surface
     49     //  geometry normals are expected to be out of the PMT 
     50     //
     51     //  boundary sign will be -ve : so line+3 outer-surface is the relevant one
     52 
     53     s.material1 = boundary_lookup( wavelength, m1_line, 0);
     54     s.m1group2  = boundary_lookup( wavelength, m1_line, 1);
     55 
     56     s.material2 = boundary_lookup( wavelength, m2_line, 0);
     57     s.surface   = boundary_lookup( wavelength, su_line, 0);
     58 
     59     s.optical = optical_buffer[su_line] ;   // index/type/finish/value
     60 
     61     s.index.x = optical_buffer[m1_line].x ; // m1 index
     62     s.index.y = optical_buffer[m2_line].x ; // m2 index 
     63     s.index.z = optical_buffer[su_line].x ; // su index
     64     s.index.w = identity.w   ;

      

To use the surface needs s.optical.x > 0 indicating a non-"NULL" surface::

    555         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    556         {
    557             command = propagate_at_surface(p, s, rng);
    558             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    559             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    560         }
    561         else
    562         {
    563             //propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    564             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    565             // tacit CONTINUE
    566         }










::

    epsilon:ggeo blyth$ g4-cc SetMotherLogical
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/divisions/src/G4PVDivision.cc:  SetMotherLogical(pMotherLogical);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/divisions/src/G4PVDivision.cc:  SetMotherLogical(pMotherLogical);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/divisions/src/G4PVDivision.cc:  SetMotherLogical(pMotherLogical);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/divisions/src/G4ReplicatedSlice.cc:  SetMotherLogical(pMotherLogical);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/volumes/src/G4PVPlacement.cc:    SetMotherLogical(motherLogical);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/volumes/src/G4PVPlacement.cc:    SetMotherLogical(motherLogical);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/volumes/src/G4PVPlacement.cc:  SetMotherLogical(pMotherLogical);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/volumes/src/G4PVPlacement.cc:  SetMotherLogical(pMotherLogical);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/volumes/src/G4PVReplica.cc:  SetMotherLogical(motherLogical);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/geometry/volumes/src/G4PVReplica.cc:  SetMotherLogical(pMotherLogical);
    epsilon:ggeo blyth$ 



::

     43 G4PVPlacement::G4PVPlacement( G4RotationMatrix *pRot,
     44                         const G4ThreeVector &tlate,
     45                         const G4String& pName,
     46                               G4LogicalVolume *pLogical,
     47                               G4VPhysicalVolume *pMother,
     48                               G4bool pMany,
     49                               G4int pCopyNo,
     50                               G4bool pSurfChk )
     51   : G4VPhysicalVolume(pRot,tlate,pName,pLogical,pMother),
     52     fmany(pMany), fallocatedRotM(false), fcopyNo(pCopyNo)
     53 {
     54   if (pMother)
     55   {
     56     G4LogicalVolume* motherLogical = pMother->GetLogicalVolume();
     57     if (pLogical == motherLogical)
     58     {
     59       G4Exception("G4PVPlacement::G4PVPlacement()", "GeomVol0002",
     60                   FatalException, "Cannot place a volume inside itself!");
     61     }
     62     SetMotherLogical(motherLogical);
     63     motherLogical->AddDaughter(this);
     64     if (pSurfChk) { CheckOverlaps(); }
     65   }
     66 }







G4OpBoundaryProcess::PostStepDoIt::

     313 
     314         G4LogicalSurface* Surface = NULL;
     315 
     316         Surface = G4LogicalBorderSurface::GetSurface(thePrePV, thePostPV);
     ...
     318         if (Surface == NULL){
     319             G4bool enteredDaughter= (thePostPV->GetMotherLogical() ==
     320                                    thePrePV ->GetLogicalVolume());
     321             if(enteredDaughter){
     322                  Surface =
     323                       G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
     ...
     ...            bla bla, trying other volume
     ...      
     336 
     337     if (Surface) OpticalSurface =
     338            dynamic_cast <G4OpticalSurface*> (Surface->GetSurfaceProperty());
     339 
     340     if (OpticalSurface) {
     341 
     342            type      = OpticalSurface->GetType();
     343        theModel  = OpticalSurface->GetModel();
     344        theFinish = OpticalSurface->GetFinish();
     345 
     346        aMaterialPropertiesTable = OpticalSurface->
     347                     GetMaterialPropertiesTable();
     ...
     ...
     366               PropertyPointer =
     367                       aMaterialPropertiesTable->GetProperty("REFLECTIVITY");
     ...
     376               if (PropertyPointer) {
     377 
     378                  theReflectivity =
     379                           PropertyPointer->Value(thePhotonMomentum);
     ...
     387               PropertyPointer =
     388               aMaterialPropertiesTable->GetProperty("EFFICIENCY");
     389               if (PropertyPointer) {
     390                       theEfficiency =
     391                       PropertyPointer->Value(thePhotonMomentum);
     392               }
     ...
     406           if ( theModel == unified ) {
     ...
     ...                  SPECULARLOBECONSTANT ->prob_sl 
     ...                  SPECULARSPIKECONSTANT -> prob_ss
     ...                  BACKSCATTERCONSTANT -> prob_bs
     ...
     468     if (type == dielectric_metal) {
     469 
     470       DielectricMetal();
     471 
     472     }



     328 inline
     329 void G4OpBoundaryProcess::DoReflection()
     330 {
     331         if ( theStatus == LambertianReflection ) {
     332 
     333           NewMomentum = G4LambertianRand(theGlobalNormal);
     334           theFacetNormal = (NewMomentum - OldMomentum).unit();
     335 
     336         }
     337         else if ( theFinish == ground ) {
     338 
     339       theStatus = LobeReflection;
     340           if ( PropertyPointer1 && PropertyPointer2 ){
     ///
     ///               means calculated REFLECTIVITY 
     ///
     341           } else {
     342              theFacetNormal =
     343                  GetFacetNormal(OldMomentum,theGlobalNormal);
     ///
     ///               smeared based on 
     ///                         OpticalSurface->GetSigmaAlpha()   theModel = unified 
     ///                         OpticalSurface->GetPolish()       otherwise (theModel = glisur? ) 
     ///                     
     ///
     344           }
     ///
     ///               just like specular, but for an ensemble its diffuse from smearing of normal
     ///               looks like can identify
     ///                       LobeReflection -> SURFACE_DREFLECT  "DR"
     ///
     345           G4double PdotN = OldMomentum * theFacetNormal;
     346           NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
     347 
     348         }
     349         else {
     350 
     351           theStatus = SpikeReflection;
     ///
     ///    looks like can just use "theStatus = SpikeReflection" to signal specular reflection, SURFACE_SREFLECT "SR"
     ///
     352           theFacetNormal = theGlobalNormal;
     353           G4double PdotN = OldMomentum * theFacetNormal;
     354           NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
     355 
     356         }
     357         G4double EdotN = OldPolarization * theFacetNormal;
     358         NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
     359 }






efficient refraction direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://http.developer.nvidia.com/GPUGems/gpugems_ch02.html

* http://ptgmedia.pearsoncmg.com/images/9780321399526/samplepages/0321399528.pdf

  * Foley, Computer Graphics principles and practice : sample chapter

* http://steve.hollasch.net/cgindex/render/refraction.txt

* :google:`heckbert refraction`

  * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.67.4048



optixu_math_namespace.h::

    2036 /**
    2037 *  Calculates refraction direction
    2038 *  r   : refraction vector
    2039 *  i   : incident vector
    2040 *  n   : surface normal
    2041 *  ior : index of refraction ( n2 / n1 )
    2042 *  returns false in case of total internal reflection, in that case r is
    2043 *          initialized to (0,0,0).
    2044 */
    2045 OPTIXU_INLINE RT_HOSTDEVICE bool refract(float3& r, const float3& i, const float3& n, const float ior)
    2046 {
    2047   float3 nn = n;
    2048   float negNdotV = dot(i,nn);
    2049   float eta;
    2050 
    2051   if (negNdotV > 0.0f)
    2052   {
    2053     eta = ior;
    2054     nn = -n;
    2055     negNdotV = -negNdotV;
    2056   }
    2057   else
    2058   {
    2059     eta = 1.f / ior;
    2060   }
    2061 
    2062   const float k = 1.f - eta*eta * (1.f - negNdotV * negNdotV);
    2063 
    2064   if (k < 0.0f) {
    2065     // Initialize this value, so that r always leaves this function initialized.
    2066     r = make_float3(0.f);
    2067     return false;
    2068   } else {
    2069     r = normalize(eta*i - (eta*negNdotV + sqrtf(k)) * nn);
    2070     return true;
    2071   }
    2072 }





chroma at_boundary
~~~~~~~~~~~~~~~~~~~~~



photon.h::


    411 __noinline__ __device__ void
    412 propagate_at_boundary(Photon &p, State &s, curandState &rng)
    413 {
    414     float incident_angle = get_theta(s.surface_normal,-p.direction);
    415     float refracted_angle = asinf(sinf(incident_angle)*s.refractive_index1/s.refractive_index2);
    416 
    417     float3 incident_plane_normal = cross(p.direction, s.surface_normal);
    418     float incident_plane_normal_length = norm(incident_plane_normal);
    419 
    420     // Photons at normal incidence do not have a unique plane of incidence,
    421     // so we have to pick the plane normal to be the polarization vector
    422     // to get the correct logic below
    423     if (incident_plane_normal_length < 1e-6f)
    424         incident_plane_normal = p.polarization;
    425     else
    426         incident_plane_normal /= incident_plane_normal_length;
    427 
    428     float normal_coefficient = dot(p.polarization, incident_plane_normal);
    429     float normal_probability = normal_coefficient*normal_coefficient;
    ///
    ///     normal_coefficient
    ///             fraction of E field amplitude perpendicular to plane of incidence (S polarization fraction)
    ///     normal_probability
    ///              coeff^2
    ///     
    430 
    431     float reflection_coefficient;
    432     if (curand_uniform(&rng) < normal_probability)
    433     {
    434         // photon polarization normal to plane of incidence
    435         reflection_coefficient = -sinf(incident_angle-refracted_angle)/sinf(incident_angle+refracted_angle);
    ///
    ///             fresnel-eoe.pdf (16)
    ///
    ///                      sin(th_i - th_t)
    ///             r_s = -  -----------------         S polarized : amplitude reflection coefficient
    ///                      sin(th_i + th_t)
    ///
    ///
    ///             r_s =  n1 cost1 - n2 cost2
    ///                   ----------------------
    ///                    n1 cost1 + n2 cost2 
    ///
    ///
    ///             isnan(refracted_angle) to handle TIR
    ///
    ///           S polarized : E field perpendicular to plane of incidence
    ///
    436 
    437         if ((curand_uniform(&rng) < reflection_coefficient*reflection_coefficient) || isnan(refracted_angle))
    438         {
    439             p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);
    440             p.history |= REFLECT_SPECULAR;
    441         }
    442         else
    443         {
    444             // hmm maybe add REFRACT_? flag for this branch  
    445             p.direction = rotate(s.surface_normal, PI-refracted_angle, incident_plane_normal);
    446         }
    447         p.polarization = incident_plane_normal;
    448     }
    449     else
    450     {
    451         // photon polarization parallel to plane of incidence
    452         reflection_coefficient = tanf(incident_angle-refracted_angle)/tanf(incident_angle+refracted_angle);
    ///
    ///
    ///             fresnel-eoe.pdf (14)     (sign convention difference, but amplitudes so doesnt matter)
    ///
    ///                         tan(th_i - th_t )
    ///             r_p  =  -  --------------------      P polarized : amplitude reflection coefficient
    ///                         tan(th_i + th_t )
    ///
    ///           P polarized : E field within plane of incidence
    ///
    ///           although the snells law substituted forms are prettier, expect they will
    ///           be more expensive than the unsubstituted 
    ///
    ///             r_p  = n1 cost2 - n2 cost1  
    ///                   ----------------------
    ///                    n1 cost2 + n2 cost1 
    ///
    453 
    454         if ((curand_uniform(&rng) < reflection_coefficient*reflection_coefficient) || isnan(refracted_angle))
    455         {
    456             p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);
    457             p.history |= REFLECT_SPECULAR;
    458         }
    459         else
    460         {
    461             // hmm maybe add REFRACT_? flag for this branch  
    462             p.direction = rotate(s.surface_normal, PI-refracted_angle, incident_plane_normal);
    463         }
    464 
    465         p.polarization = cross(incident_plane_normal, p.direction);
    466         p.polarization /= norm(p.polarization);
    467     }
    468 
    469 } // propagate_at_boundary



photon.h::

    168 __device__ float
    169 get_theta(const float3 &a, const float3 &b)
    170 {
    171     return acosf(fmaxf(-1.0f,fminf(1.0f,dot(a,b))));
    172 }

rotate.h::

     19 /* rotate points counterclockwise, when looking towards +infinity,
     20    through an angle `phi` about the axis `n`. */
     21 __device__ float3
     22 rotate(const float3 &a, float phi, const float3 &n)
     23 {   
     24     float cos_phi = cosf(phi);
     25     float sin_phi = sinf(phi);
     26 
     27     return a*cos_phi + n*dot(a,n)*(1.0f-cos_phi) + cross(a,n)*sin_phi;
     28 }


Chroma approach much simpler as

#. random branches on S/P polarization, whereas Geant4 handles both together 
#. random branches on reflect/transmit

Are they equivalent ?




rayleigh scatter
~~~~~~~~~~~~~~~~

* http://bugzilla-geant4.kek.jp/show_bug.cgi?id=207
* http://antares.in2p3.fr/users/bailey/thesis/html/node114.html

::

    //
    // one way of applying two rotations to the axis vector
    //
    240 __device__ float3
    241 pick_new_direction(float3 axis, float theta, float phi)
    242 {
    243     // Taken from SNOMAN rayscatter.for
    244     float cos_theta, sin_theta;
    245     sincosf(theta, &sin_theta, &cos_theta);
    246     float cos_phi, sin_phi;
    247     sincosf(phi, &sin_phi, &cos_phi);
    248 
    249     float sin_axis_theta = sqrt(1.0f - axis.z*axis.z);
    250     float cos_axis_phi, sin_axis_phi;
    251 
    252     if (isnan(sin_axis_theta) || sin_axis_theta < 0.00001f) {
    253     cos_axis_phi = 1.0f;
    254     sin_axis_phi = 0.0f;
    255     }
    256     else {
    257     cos_axis_phi = axis.x / sin_axis_theta;
    258     sin_axis_phi = axis.y / sin_axis_theta;
    259     }
    260 
    261     float dirx = cos_theta*axis.x +
    262     sin_theta*(axis.z*cos_phi*cos_axis_phi - sin_phi*sin_axis_phi);
    263     float diry = cos_theta*axis.y +
    264     sin_theta*(cos_phi*axis.z*sin_axis_phi - sin_phi*cos_axis_phi);
    265     float dirz = cos_theta*axis.z - sin_theta*cos_phi*sin_axis_theta;
    266 
    267     return make_float3(dirx, diry, dirz);
    268 }



               dirx      cos_theta           0                     axis.x
  
               diry                  cos_theta                     axis.y
 
               dirz                              cos_theta         axis.z


    270 __device__ void
    271 rayleigh_scatter(Photon &p, curandState &rng)
    272 {
    273     float cos_theta = 2.0f*cosf((acosf(1.0f - 2.0f*curand_uniform(&rng))-2*PI)/3.0f);

                             acosf(u[-1:1])          0->pi
                             acosf(u[-1:1]) - 2pi    -2pi->-pi     ??

                * http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDA__MATH__SINGLE_g63d1c22538561dc228fc230d10d85dca.html
                * the 1/3.0 makes me think this is following the old (wrong) Geant4 distrib ? 
                  (see Xin Qian attachment)
                               

    274     if (cos_theta > 1.0f)
    275     cos_theta = 1.0f;
    276     else if (cos_theta < -1.0f)
    277     cos_theta = -1.0f;
    278 
    279     float theta = acosf(cos_theta);
    280     float phi = uniform(&rng, 0.0f, 2.0f * PI);
    281 
    282     p.direction = pick_new_direction(p.polarization, theta, phi);
    283 
    284     if (1.0f - fabsf(cos_theta) < 1e-6f) {
    285     p.polarization = pick_new_direction(p.polarization, PI/2.0f, phi);
    286     }
    287     else {
    288     // linear combination of old polarization and new direction
    289     p.polarization = p.polarization - cos_theta * p.direction;
    290     }
    291 
    292     p.direction /= norm(p.direction);
    293     p.polarization /= norm(p.polarization);
    294 } // scatter


Xin Qian : Rayleigh Scattering In GEANT4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://bugzilla-geant4.kek.jp/show_bug.cgi?id=207
* http://bugzilla-geant4.kek.jp/attachment.cgi?id=77
* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/opticalphotons/455.html

Correct way to simulate the angular distribution is:

#. Randomly generate the outgoing photon momentum vector.
#. Calculate the outgoing photon polarization vector,

   * Perpendicular to the momentum vector
   * Same plane as the momentum vector and initial polarization vector

#. Weight or Generate distribution according to (cos theta)^2

   * where theta is the angle between the two photon polarization vectors



source/processes/optical/src/G4OpRayleigh.cc::

    ///
    ///  code follows Xin Qian patch
    ///
    124 G4VParticleChange*
    125 G4OpRayleigh::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    126 {
    127         aParticleChange.Initialize(aTrack);
    128 
    129         const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    ...
    139         G4double cosTheta;
    140         G4ThreeVector OldMomentumDirection, NewMomentumDirection;
    141         G4ThreeVector OldPolarization, NewPolarization;
    142 
    143         do {
    144            // Try to simulate the scattered photon momentum direction
    145            // w.r.t. the initial photon momentum direction
    146 
    147            G4double CosTheta = G4UniformRand();
    148            G4double SinTheta = std::sqrt(1.-CosTheta*CosTheta);
    149            // consider for the angle 90-180 degrees
    150            if (G4UniformRand() < 0.5) CosTheta = -CosTheta;
    151 
    152            // simulate the phi angle
    153            G4double rand = twopi*G4UniformRand();
    154            G4double SinPhi = std::sin(rand);
    155            G4double CosPhi = std::cos(rand);
    156 
    157            // start constructing the new momentum direction
    158        G4double unit_x = SinTheta * CosPhi;
    159        G4double unit_y = SinTheta * SinPhi;
    160        G4double unit_z = CosTheta;
    161        NewMomentumDirection.set (unit_x,unit_y,unit_z);
    162
    163            // Rotate the new momentum direction into global reference system
    164            OldMomentumDirection = aParticle->GetMomentumDirection();
    165            OldMomentumDirection = OldMomentumDirection.unit();
    166            NewMomentumDirection.rotateUz(OldMomentumDirection);
    167            NewMomentumDirection = NewMomentumDirection.unit();
    168 
    169            // calculate the new polarization direction
    170            // The new polarization needs to be in the same plane as the new
    171            // momentum direction and the old polarization direction
    172            OldPolarization = aParticle->GetPolarization();
    173            G4double constant = -1./NewMomentumDirection.dot(OldPolarization);
    174 
    175            NewPolarization = NewMomentumDirection + constant*OldPolarization;
    176            NewPolarization = NewPolarization.unit();
    177 
    178            // There is a corner case, where the Newmomentum direction
    179            // is the same as oldpolariztion direction:
    180            // random generate the azimuthal angle w.r.t. Newmomentum direction
    181            if (NewPolarization.mag() == 0.) {
    182               rand = G4UniformRand()*twopi;
    183               NewPolarization.set(std::cos(rand),std::sin(rand),0.);
    184               NewPolarization.rotateUz(NewMomentumDirection);
    185            } else {
    186               // There are two directions which are perpendicular
    187               // to the new momentum direction
    188               if (G4UniformRand() < 0.5) NewPolarization = -NewPolarization;
    189            }
    190      
    191        // simulate according to the distribution cos^2(theta)
    192            cosTheta = NewPolarization.dot(OldPolarization);
    193         } while (std::pow(cosTheta,2) < G4UniformRand());
    194 


::

    296 __device__
    297 int propagate_to_boundary(Photon &p, State &s, curandState &rng,
    298                           bool use_weights=false, int scatter_first=0)
    299 {
    300     float absorption_distance = -s.absorption_length*logf(curand_uniform(&rng));
    301     float scattering_distance = -s.scattering_length*logf(curand_uniform(&rng));
    ...
    346     if (absorption_distance <= scattering_distance) {
    347         if (absorption_distance <= s.distance_to_boundary)
    348         {
    349             p.time += absorption_distance/(SPEED_OF_LIGHT/s.refractive_index1);
    350             p.position += absorption_distance*p.direction;
    351 
    352             float uniform_sample_reemit = curand_uniform(&rng);
    353             if (uniform_sample_reemit < s.reemission_prob)
    354             {
    ...                 //  .wavelength .direction .polarization
    364                 p.history |= BULK_REEMIT;
    365                 return CONTINUE;
    366             } // photon is reemitted isotropically
    367             else
    368             {
    369                 p.last_hit_triangle = -1;
    370                 p.history |= BULK_ABSORB;
    371                 return BREAK;
    372             } // photon is absorbed in material1
    373         }
    374     }
    378     else
    379     {
    380         if (scattering_distance <= s.distance_to_boundary) {
    385             p.time += scattering_distance/(SPEED_OF_LIGHT/s.refractive_index1);
    386             p.position += scattering_distance*p.direction;
    388             rayleigh_scatter(p, rng);
    390             p.history |= RAYLEIGH_SCATTER;
    392             p.last_hit_triangle = -1;
    393 
    394             return CONTINUE;
    395         } // photon is scattered in material1
    396     } // if scattering_distance < absorption_distance
    402     //  Survive to boundary(PASS)  .position .time advanced to boundary 
    404     p.position += s.distance_to_boundary*p.direction;
    405     p.time += s.distance_to_boundary/(SPEED_OF_LIGHT/s.refractive_index1);
    407     return PASS;





propagate_at_surface (default surface model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. SURFACE_ABSORB(BREAK)
#. SURFACE_DETECT(BREAK)
#. REFLECT_DIFFUSE(CONTINUE) .direction .polarization
#. REFLECT_SPECULAR(CONTINUE) .direction
#. NO other option, so never PASS? 

propagate_at_boundary
~~~~~~~~~~~~~~~~~~~~~~~

Depending on materials refractive indices and incidence angle

#. REFLECT_SPECULAR("CONTINUE")  .direction .polarization
#. "REFRACT"("CONTINUE")         .direction .polarization
#. NO other option


geant4 process characteristic distances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

source/processes/management/src/G4VProcess.cc::

     95 void G4VProcess::ResetNumberOfInteractionLengthLeft()
     96 {
     97   theNumberOfInteractionLengthLeft =  -std::log( G4UniformRand() );
     98   theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft;
     99 }

source/processes/management/include/G4VProcess.hh::

    543 inline
    544 void G4VProcess::SubtractNumberOfInteractionLengthLeft(
    545                                   G4double previousStepSize )
    546 {
    547   if (currentInteractionLength>0.0) {
    548     theNumberOfInteractionLengthLeft -= previousStepSize/currentInteractionLength;
    549     if(theNumberOfInteractionLengthLeft<0.) {
    550        theNumberOfInteractionLengthLeft=CLHEP::perMillion;
    551     }
    ...
    569 }

source/processes/management/include/G4VProcess.hh::

    498 inline G4double G4VProcess::PostStepGPIL( const G4Track& track,
    499                                    G4double   previousStepSize,
    500                                    G4ForceCondition* condition )
    501 {
    502   G4double value
    503    =PostStepGetPhysicalInteractionLength(track, previousStepSize, condition);
    504   return thePILfactor*value;
    505 }

source/processes/management/src/G4VDiscreteProcess.cc::

    071 G4double G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(
     72                              const G4Track& track,
     73                  G4double   previousStepSize,
     74                  G4ForceCondition* condition
     75                 )
     76 {
     77   if ( (previousStepSize < 0.0) || (theNumberOfInteractionLengthLeft<=0.0)) {
     78     // beggining of tracking (or just after DoIt of this process)
     79     ResetNumberOfInteractionLengthLeft();
     80   } else if ( previousStepSize > 0.0) {
     81     // subtract NumberOfInteractionLengthLeft 
     82     SubtractNumberOfInteractionLengthLeft(previousStepSize);
     83   } else {
     84     // zero step
     85     //  DO NOTHING
     86   }
     87 
     88   // condition is set to "Not Forced"
     89   *condition = NotForced;
     90 
     91   // get mean free path
     92   currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
     93 
     94   G4double value;
     95   if (currentInteractionLength <DBL_MAX) {
     96     value = theNumberOfInteractionLengthLeft * currentInteractionLength;
     97   } else {
     98     value = DBL_MAX;
     99   }
    ...
    109   return value;
    110 }


::

    1121 G4double G4OpBoundaryProcess::GetMeanFreePath(const G4Track& ,
    1122                                               G4double ,
    1123                                               G4ForceCondition* condition)
    1124 {
    ////     forced(but not exclusively) but with "infinite" length, so will come last 
    ////     but discrete so this means that the other processes didnt trigger ?
    ////
    1125     *condition = Forced;
    1127     return DBL_MAX;
    1128 }

    117 G4double G4OpAbsorption::GetMeanFreePath(const G4Track& aTrack,
    118                          G4double ,
    119                          G4ForceCondition* )
    120 {
    ...
    148         return AttenuationLength;  // "ABSLENGTH" energy lookup
    149 }

    265 G4double G4OpRayleigh::GetMeanFreePath(const G4Track& aTrack,
    266                                      G4double ,
    267                                      G4ForceCondition* )
    268 {
    ...
    305         return AttenuationLength;  // "RAYLEIGH" energy lookup
    306 }




Optical Surface Defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~

source/materials/include/G4OpticalSurface.hh::

    134     G4OpticalSurface(const G4String& name,
    135              G4OpticalSurfaceModel model = glisur,
    136              G4OpticalSurfaceFinish finish = polished,
    137              G4SurfaceType type = dielectric_dielectric,
    138              G4double value = 1.0);
    139         // Constructor of an optical surface object.

Fresnel Eqn for polarized light
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://en.wikipedia.org/wiki/Fresnel_equations

  * separate reflectivity eqns for s and p polarized light 
  * would be good for find a vector treatment 
  * http://old.rqc.ru/quantech/pubs/2013/fresnel-eoe.pdf



G4OpBoundaryProcess
~~~~~~~~~~~~~~~~~~~~~

source/global/HEPRandom/include/G4RandomTools.hh
source/global/HEPRandom/include/G4RandomDirection.hh

  * inline G4ThreeVector G4LambertianRand(const G4ThreeVector& normal)




source/tracking/src/G4SteppingManager.cc::

    116 G4StepStatus G4SteppingManager::Stepping()
    ...
    173 //---------------------------------
    174 // AlongStep and PostStep Processes
    175 //---------------------------------
    176 
    177 
    178    else{
    179      // Find minimum Step length demanded by active disc./cont. processes
    180      DefinePhysicalStepLength();
    181 
    182      // Store the Step length (geometrical length) to G4Step and G4Track
    183      fStep->SetStepLength( PhysicalStep );
    184      fTrack->SetStepLength( PhysicalStep );
    185      G4double GeomStepLength = PhysicalStep;
    186 
    187      // Store StepStatus to PostStepPoint
    188      fStep->GetPostStepPoint()->SetStepStatus( fStepStatus );
    189 
    190      // Invoke AlongStepDoIt 
    191      InvokeAlongStepDoItProcs();
    192 
    193      // Update track by taking into account all changes by AlongStepDoIt
    194      fStep->UpdateTrack();
    195 
    196      // Update safety after invocation of all AlongStepDoIts
    197      endpointSafOrigin= fPostStepPoint->GetPosition();
    198 //     endpointSafety=  std::max( proposedSafety - GeomStepLength, 0.);
    199      endpointSafety=  std::max( proposedSafety - GeomStepLength, kCarTolerance);
    200 
    201      fStep->GetPostStepPoint()->SetSafety( endpointSafety );
    ...
    208      // Invoke PostStepDoIt
    209      InvokePostStepDoItProcs();
    210 
    ...
    221 // Update 'TrackLength' and remeber the Step length of the current Step
    222    fTrack->AddTrackLength(fStep->GetStepLength());
    223    fPreviousStepSize = fStep->GetStepLength();
    224    fStep->SetTrack(fTrack);
    ...
    230 // Send G4Step information to Hit/Dig if the volume is sensitive
    231    fCurrentVolume = fStep->GetPreStepPoint()->GetPhysicalVolume();
    232    StepControlFlag =  fStep->GetControlFlag();
    233    if( fCurrentVolume != 0 && StepControlFlag != AvoidHitInvocation) {
    234       fSensitive = fStep->GetPreStepPoint()->
    235                                    GetSensitiveDetector();
    236       if( fSensitive != 0 ) {
    237         fSensitive->Hit(fStep);
    238       }
    239    }
    240 
    241 // User intervention process.
    242    if( fUserSteppingAction != 0 ) {
    243       fUserSteppingAction->UserSteppingAction(fStep);
    244    }
    245    G4UserSteppingAction* regionalAction
    246     = fStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetRegion()
    247       ->GetRegionalSteppingAction();
    248    if( regionalAction ) regionalAction->UserSteppingAction(fStep);
    249 
    250 // Stepping process finish. Return the value of the StepStatus.
    251    return fStepStatus;
    252 
    253 }





source/tracking/src/G4SteppingManager2.cc::

    056 void G4SteppingManager::GetProcessNumber()
    ...
    064   G4ProcessManager* pm= fTrack->GetDefinition()->GetProcessManager();
    ...
    095    MAXofPostStepLoops = pm->GetPostStepProcessVector()->entries();
    096    fPostStepDoItVector = pm->GetPostStepProcessVector(typeDoIt);
    097    fPostStepGetPhysIntVector = pm->GetPostStepProcessVector(typeGPIL);
    ...


source/tracking/src/G4SteppingManager2.cc::

    128  void G4SteppingManager::DefinePhysicalStepLength()
    ...
    133    PhysicalStep  = DBL_MAX;          // Initialize by a huge number    
    134    physIntLength = DBL_MAX;          // Initialize by a huge number  
    ...

    162 // GPIL for PostStep
    163    fPostStepDoItProcTriggered = MAXofPostStepLoops;
    ...
    165    for(size_t np=0; np < MAXofPostStepLoops; np++){
    166      fCurrentProcess = (*fPostStepGetPhysIntVector)(np);
    ...
    172      physIntLength = fCurrentProcess->
    173                      PostStepGPIL( *fTrack,
    174                                                  fPreviousStepSize,
    175                                                       &fCondition );
    ...
    181      switch (fCondition) {
    182      case ExclusivelyForced:
    183          (*fSelectedPostStepDoItVector)[np] = ExclusivelyForced;
    184          fStepStatus = fExclusivelyForcedProc;
    185          fStep->GetPostStepPoint()
    186          ->SetProcessDefinedStep(fCurrentProcess);
    187          break;
    188      case Conditionally:
    189        //        (*fSelectedPostStepDoItVector)[np] = Conditionally;
    190          G4Exception("G4SteppingManager::DefinePhysicalStepLength()", "Tracking1001", FatalException, "This feature no more supported");
    191 
    192          break;
    193      case Forced:
    194          (*fSelectedPostStepDoItVector)[np] = Forced;
    195          break;
    196      case StronglyForced:
    197          (*fSelectedPostStepDoItVector)[np] = StronglyForced;
    198          break;
    199      default:
    200          (*fSelectedPostStepDoItVector)[np] = InActivated;
    201          break;
    202      }
    203 
    ...   
    206      if (fCondition==ExclusivelyForced) {
    207      for(size_t nrest=np+1; nrest < MAXofPostStepLoops; nrest++){
    208          (*fSelectedPostStepDoItVector)[nrest] = InActivated;
    209      }
    210      return;  // Take note the 'return' at here !!! 
    211      }
    212      else{
    ///
    ///     picking process that yields the smallest PostStepGPIL
    ///
    213      if(physIntLength < PhysicalStep ){
    214          PhysicalStep = physIntLength;
    215          fStepStatus = fPostStepDoItProc;
    216          fPostStepDoItProcTriggered = G4int(np);
    217          fStep->GetPostStepPoint()
    218          ->SetProcessDefinedStep(fCurrentProcess);
    219      }
    223    }   // process loop
    ...
    ...     set the winning process to NotForced
    ...
    225    if (fPostStepDoItProcTriggered<MAXofPostStepLoops) {
    226        if ((*fSelectedPostStepDoItVector)[fPostStepDoItProcTriggered] ==
    227        InActivated) {
    228        (*fSelectedPostStepDoItVector)[fPostStepDoItProcTriggered] =
    229            NotForced;
    230        }
    231    }


source/tracking/src/G4SteppingManager2.cc::

    482 ////////////////////////////////////////////////////////
    483 void G4SteppingManager::InvokePostStepDoItProcs()
    484 ////////////////////////////////////////////////////////
    485 {
    486 
    487 // Invoke the specified discrete processes
    488    for(size_t np=0; np < MAXofPostStepLoops; np++){
    489    //
    490    // Note: DoItVector has inverse order against GetPhysIntVector
    491    //       and SelectedPostStepDoItVector.
    492    //
    493      G4int Cond = (*fSelectedPostStepDoItVector)[MAXofPostStepLoops-np-1];
    494      if(Cond != InActivated){
    495        if( 
                     ((Cond == NotForced)         && (fStepStatus == fPostStepDoItProc)) 
               // NotForced is default setting for discrete 
                     ||
    496              ((Cond == Forced)            && (fStepStatus != fExclusivelyForcedProc)) 
                     ||
    498              ((Cond == ExclusivelyForced) && (fStepStatus == fExclusivelyForcedProc)) 
                     ||
    499              ((Cond == StronglyForced) )
    500          ) {
    501 
    502      InvokePSDIP(np);
    503          if ((np==0) && (fTrack->GetNextVolume() == 0)){
    504            fStepStatus = fWorldBoundary;
    505            fStep->GetPostStepPoint()->SetStepStatus( fStepStatus );
    506          }
    507        }
    508      } //if(*fSelectedPostStepDoItVector(np)........
    509 
    510      // Exit from PostStepLoop if the track has been killed,
    511      // but extra treatment for processes with Strongly Forced flag
    512      if(fTrack->GetTrackStatus() == fStopAndKill) {
    513        for(size_t np1=np+1; np1 < MAXofPostStepLoops; np1++){
    514           G4int Cond2 = (*fSelectedPostStepDoItVector)[MAXofPostStepLoops-np1-1];
    515           if (Cond2 == StronglyForced) {
    516              InvokePSDIP(np1);
    517           }
    518        }
    519        break;
    520      }
    521    } //for(size_t np=0; np < MAXofPostStepLoops; np++){
    522 }
    ...
    526 void G4SteppingManager::InvokePSDIP(size_t np)
    527 {
    528          fCurrentProcess = (*fPostStepDoItVector)[np];
    529          fParticleChange
    530             = fCurrentProcess->PostStepDoIt( *fTrack, *fStep);
    ///
    ///       LOOKS TO BE A GOOD PLACE TO PARACHUTE IN 
    ///       ACTUALLY: CAN GET TO SAME INFO FROM STANDARD UserSteppingAction
    /// 
    ///        * fParticleChange has most everything needed
    ///        * init : allocate NPY array for the photon steps
    ///
    ///          * Nx10x4x4 step array (for 10 steps, and recording 4 quads per step)
    ///
    ///        * how to arrive at indices for photons and steps
    ///
    ///          * photon index (vector<G4Track*> restricted to OP can provide an index)
    ///            
    ///          * step index ? 
    /// 
    ///        * reemission handled in DetSim by adding 2nd-ary : very different to Chroma approach of changing the photon 
    ///             
    ///          * maybe map<G4Track*,vector<G4Step*>> to give step index : but need special reemission handling
    ///
    ///          * flag identifying the type of change ? process name to code
    ///
    ///          * will there ever be multiple changes for one photon/step ? 
    ///            presumably not for discrete process, first process
    ///            that returns the change wins...
    ///
    ///          * is is necessary to match up the CONTINUE/BREAK/PASS chroma logic with G4 equiv ?
    ///
    ///
    531 
    532          // Update PostStepPoint of Step according to ParticleChange
    533      fParticleChange->UpdateStepForPostStep(fStep);
    534 #ifdef G4VERBOSE
    535                  // !!!!! Verbose
    536            if(verboseLevel>0) fVerbose->PostStepDoItOneByOne();
    537 #endif
    538          // Update G4Track according to ParticleChange after each PostStepDoIt
    539          fStep->UpdateTrack();
    540 
    541          // Update safety after each invocation of PostStepDoIts
    542          fStep->GetPostStepPoint()->SetSafety( CalculateSafety() );
    543 
    544          // Now Store the secondaries from ParticleChange to SecondaryList
    545          G4Track* tempSecondaryTrack;
    546          G4int    num2ndaries;
    547 
    548          num2ndaries = fParticleChange->GetNumberOfSecondaries();
    549 
    550          for(G4int DSecLoop=0 ; DSecLoop< num2ndaries; DSecLoop++){
    551             tempSecondaryTrack = fParticleChange->GetSecondary(DSecLoop);



* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSim/src/DsG4OpBoundaryProcess.cc

G4ParticleChange
-------------------

* looks like PostStep collects everything needed ?


source/track/src/G4ParticleChange.cc::

    348 G4Step* G4ParticleChange::UpdateStepForPostStep(G4Step* pStep)
    349 {
    350   // A physics process always calculates the final state of the particle
    351 
    352   // Take note that the return type of GetMomentumChange is a
    353   // pointer to G4ParticleMometum. Also it is a normalized 
    354   // momentum vector.
    355 
    356   G4StepPoint* pPostStepPoint = pStep->GetPostStepPoint();
    357   G4Track* pTrack = pStep->GetTrack();
    358 
    359   // Set Mass/Charge
    360   pPostStepPoint->SetMass(theMassChange);
    361   pPostStepPoint->SetCharge(theChargeChange);
    362   pPostStepPoint->SetMagneticMoment(theMagneticMomentChange);
    363 
    364   // update kinetic energy and momentum direction
    365   pPostStepPoint->SetMomentumDirection(theMomentumDirectionChange);
    366   pPostStepPoint->SetKineticEnergy( theEnergyChange );
    367 
    368   // calculate velocity
    369   pTrack->SetKineticEnergy( theEnergyChange );
    370   if (!isVelocityChanged) {
    371     if(theEnergyChange > 0.0) {
    372       theVelocityChange = pTrack->CalculateVelocity();
    373     } else if(theMassChange > 0.0) {
    374       theVelocityChange = 0.0;
    375     }
    376   }
    377   pPostStepPoint->SetVelocity(theVelocityChange);
    378 
    379    // update polarization
    380   pPostStepPoint->SetPolarization( thePolarizationChange );
    381      
    382   // update position and time
    383   pPostStepPoint->SetPosition( thePositionChange  );
    384   pPostStepPoint->AddGlobalTime(theTimeChange - theLocalTime0);
    385   pPostStepPoint->SetLocalTime( theTimeChange );
    386   pPostStepPoint->SetProperTime( theProperTimeChange  );
    387 

source/track/include/G4ParticleChange.hh::

    134     const G4ThreeVector* GetMomentumDirection() const;
    135     void ProposeMomentumDirection(G4double Px, G4double Py, G4double Pz);
    136     void ProposeMomentumDirection(const G4ThreeVector& Pfinal);
    137     // Get/Propose the MomentumDirection vector: it is the final momentum direction.
    138 
    139     const G4ThreeVector* GetPolarization() const;
    140     void  ProposePolarization(G4double Px, G4double Py, G4double Pz);
    141     void  ProposePolarization(const G4ThreeVector& finalPoralization);
    142     // Get/Propose the final Polarization vector.
    143 
    144     G4double GetEnergy() const;
    145     void ProposeEnergy(G4double finalEnergy);
    146     // Get/Propose the final kinetic energy of the current particle.
    147 
    148     G4double GetVelocity() const;
    149     void ProposeVelocity(G4double finalVelocity);
    150     // Get/Propose the final velocity of the current particle.
    151 
    152     G4double GetProperTime() const;
    153     void ProposeProperTime(G4double finalProperTime);
    154     //  Get/Propose th final ProperTime 
    155 
    156     const G4ThreeVector* GetPosition() const;
    157     void ProposePosition(G4double x, G4double y, G4double z);
    158     void ProposePosition(const G4ThreeVector& finalPosition);
    159     //  Get/Propose the final position of the current particle.
    160 
    161     void     ProposeGlobalTime(G4double t);
    162     void     ProposeLocalTime(G4double t);


source/track/include/G4ParticleChange.icc::     

     61 inline
     62  const G4ThreeVector* G4ParticleChange::GetMomentumDirection() const
     63 {
     64   return &theMomentumDirectionChange;
     65 }
     66 
     67 inline
     68  void G4ParticleChange::ProposeMomentumDirection(
     69                         G4double Px,
     70                         G4double Py,
     71                         G4double Pz )
     72 {
     73   theMomentumDirectionChange.setX(Px);
     74   theMomentumDirectionChange.setY(Py);
     75   theMomentumDirectionChange.setZ(Pz);
     76 }



G4OpRayleigh
--------------

::

    124 G4VParticleChange*
    125 G4OpRayleigh::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    126 {
    127         aParticleChange.Initialize(aTrack);
    ...
    195         aParticleChange.ProposePolarization(NewPolarization);
    196         aParticleChange.ProposeMomentumDirection(NewMomentumDirection);
    197 




G4VProcess
-----------

* source/processes/management/include/G4VProcess.hh

pParticleChange points to aParticleChange::

    282   protected:
    283       G4VParticleChange* pParticleChange;
    284       //  The pointer to G4VParticleChange object 
    285       //  which is modified and returned by address by the DoIt() method.
    286       //  This pointer should be set in each physics process
    287       //  after construction of derived class object.  
    288 
    289       G4ParticleChange aParticleChange;
    290       //  This object is kept for compatibility with old scheme
    291       //  This will be removed in future


source/processes/management/src/G4VProcess.cc::

     52 G4VProcess::G4VProcess(const G4String& aName, G4ProcessType   aType )
     53                   : aProcessManager(0),
     54                 pParticleChange(0),
     55                     theNumberOfInteractionLengthLeft(-1.0),
     56                     currentInteractionLength(-1.0),
     57             theInitialNumberOfInteractionLength(-1.0),
     58                     theProcessName(aName),
     59             theProcessType(aType),
     60             theProcessSubType(-1),
     61                     thePILfactor(1.0),
     62                     enableAtRestDoIt(true),
     63                     enableAlongStepDoIt(true),
     64                     enablePostStepDoIt(true),
     65                     verboseLevel(0),
     66                     masterProcessShadow(0)
     67 
     68 {
     69   pParticleChange = &aParticleChange;
     70 }


G4VDiscreteProcess 
-------------------

* source/processes/management/include/G4VDiscreteProcess.hh
* no operation in  AtRestDoIt and  AlongStepDoIt,  all in PostStepDoIt

::

    112 G4VParticleChange* G4VDiscreteProcess::PostStepDoIt(
    113                             const G4Track& ,
    114                             const G4Step&
    115                             )
    116 {
    117 //  clear NumberOfInteractionLengthLeft
    118     ClearNumberOfInteractionLengthLeft();
    119 
    120     return pParticleChange;
    121 }


G4OpAbsorption
-----------------

::

    100 G4VParticleChange*
    101 G4OpAbsorption::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    102 {
    103         aParticleChange.Initialize(aTrack);
    104 
    105         aParticleChange.ProposeTrackStatus(fStopAndKill);
    106 
    107         if (verboseLevel>0) {
    108        G4cout << "\n** Photon absorbed! **" << G4endl;
    109         }
    110         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    111 }



G4OpBoundaryProcess
--------------------

GetMeanFreePath
~~~~~~~~~~~~~~~~

::

    163     G4double GetMeanFreePath(const G4Track& ,
    164                  G4double ,
    165                  G4ForceCondition* condition);
    166         // Returns infinity; i. e. the process does not limit the step,
    167         // but sets the 'Forced' condition for the DoIt to be invoked at
    168         // every step. However, only at a boundary will any action be
    169         // taken.



ChooseReflection sets *theStatus*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

      SpikeReflection|LobeReflection|BackScattering|LambertianReflection
                     |              |              |     
                    _ss          _sl+_ss        _sl+_ss+_bs




G4OpticalSurface
------------------

::

    simon:geant4.10.00.p01 blyth$ find . -name '*.cc' -exec grep -l OpticalSurface {} \;
    ./examples/advanced/air_shower/src/UltraDetectorConstruction.cc
    ./examples/advanced/underground_physics/src/DMXDetectorConstruction.cc
    ./examples/extended/optical/LXe/src/LXeDetectorConstruction.cc
    ./examples/extended/optical/LXe/src/LXeMainVolume.cc
    ./examples/extended/optical/OpNovice/src/OpNoviceDetectorConstruction.cc
    ./examples/extended/optical/wls/src/WLSDetectorConstruction.cc
    ./source/materials/src/G4OpticalSurface.cc
    ./source/persistency/gdml/src/G4GDMLReadSolids.cc
    ./source/persistency/gdml/src/G4GDMLWriteSolids.cc
    ./source/persistency/gdml/src/G4GDMLWriteStructure.cc
    ./source/physics_lists/constructors/electromagnetic/src/G4OpticalPhysicsMessenger.cc
    ./source/processes/optical/src/G4OpBoundaryProcess.cc
    simon:geant4.10.00.p01 blyth$ 


::

     G4OpBoundaryProcess::PostStepDoIt

     g4op-boundary 
     (lldb) b "G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)"
     (lldb) expr verboseLevel = 10 

     (lldb) p pStep->GetPostStepPoint()->GetStepStatus()
     (G4StepStatus) $2 = fGeomBoundary

     (lldb) br l
     Current breakpoints:
     5: name = 'G4VParticleChange* G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)', locations = 0 (pending)

     6: name = 'G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)', locations = 1, resolved = 1, hit count = 5

     7: file = '/usr/local/env/g4/geant4.10.02/source/processes/optical/src/G4OpBoundaryProcess.cc', line = 491, locations = 1, resolved = 1, hit count = 0

     (lldb) br del 5
     1 breakpoints deleted; 0 breakpoint locations disabled.
     (lldb) br del 6
     1 breakpoints deleted; 0 breakpoint locations disabled.
     (lldb) br l 
     Current breakpoints:
     7: file = '/usr/local/env/g4/geant4.10.02/source/processes/optical/src/G4OpBoundaryProcess.cc', line = 491, locations = 1, resolved = 1, hit count = 0

     (lldb) 


     [2016-Mar-04 13:05:59.051653]:info:  eV      3.263 nm     380.000 v        1.483

     (lldb) p thePhotonMomentum*1e6
     (double) $6 = 3.2627417774210459

     (lldb) p Rindex1
     (G4double) $7 = 1.4826403856277466



     162 G4VParticleChange*
     163 G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     164 {
     165         theStatus = Undefined;
     166 
     167         aParticleChange.Initialize(aTrack);
     168         aParticleChange.ProposeVelocity(aTrack.GetVelocity());
     169 
     170         // Get hyperStep from  G4ParallelWorldProcess
     171         //  NOTE: PostSetpDoIt of this process should be
     172         //        invoked after G4ParallelWorldProcess!
     173 
     174         const G4Step* pStep = &aStep;
     175 
     176         const G4Step* hStep = G4ParallelWorldProcess::GetHyperStep();
     177 
     178         if (hStep) pStep = hStep;
     179 
     180         G4bool isOnBoundary =
     181                 (pStep->GetPostStepPoint()->GetStepStatus() == fGeomBoundary);
     182 
     183         if (isOnBoundary) {
     184            Material1 = pStep->GetPreStepPoint()->GetMaterial();
     185            Material2 = pStep->GetPostStepPoint()->GetMaterial();
     186         } else {
     187            theStatus = NotAtBoundary;
     188            if ( verboseLevel > 0) BoundaryProcessVerbose();
     189            return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     190         }

     ///
     /// lack of material RINDEX causes StopAndKill
     ///

     271     G4MaterialPropertiesTable* aMaterialPropertiesTable;
     272     G4MaterialPropertyVector* Rindex;
     273 
     274     aMaterialPropertiesTable = Material1->GetMaterialPropertiesTable();
     275     if (aMaterialPropertiesTable) {
     276         Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
     277     }
     278     else {
     279         theStatus = NoRINDEX;
     280         if ( verboseLevel > 0) BoundaryProcessVerbose();
     281         aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
     282         aParticleChange.ProposeTrackStatus(fStopAndKill);
     283         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     284     }
     285 
     286     if (Rindex) {
     287         Rindex1 = Rindex->Value(thePhotonMomentum);
     288     }
     289     else {
     290         theStatus = NoRINDEX;
     291         if ( verboseLevel > 0) BoundaryProcessVerbose();
     292         aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
     293         aParticleChange.ProposeTrackStatus(fStopAndKill);
     294         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     295     }


     ///  defaults without an OpticalSurface

     297         theReflectivity =  1.;
     298         theEfficiency   =  0.;
     299         theTransmittance = 0.;
     300 
     301         theModel = glisur;
     302         theFinish = polished;
     303 
     304         G4SurfaceType type = dielectric_dielectric;


     ///  if fail to find border surface for the volume pair look for skin surface for either volume

     306         Rindex = NULL;
     307         OpticalSurface = NULL;
     308 
     309         G4LogicalSurface* Surface = NULL;
     310 
     311         Surface = G4LogicalBorderSurface::GetSurface(thePrePV, thePostPV);
     312 
     313         if (Surface == NULL){
     314              G4bool enteredDaughter= (thePostPV->GetMotherLogical() ==
     315                                       thePrePV ->GetLogicalVolume());
     316              if(enteredDaughter){
     317                  Surface =
     318                         G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
     319                  if(Surface == NULL)
     320                       Surface =
     321                             G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
     322              }
     323              else {
     324                  Surface =
     325                         G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
     326                  if(Surface == NULL)
     327                       Surface =
     328                                G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
     329              }
     330         }
     331 
     332         if (Surface) OpticalSurface =
     333                         dynamic_cast <G4OpticalSurface*> (Surface->GetSurfaceProperty());


     335     if (OpticalSurface) {
     336 
     337            type      = OpticalSurface->GetType();
     338            theModel  = OpticalSurface->GetModel();
     339            theFinish = OpticalSurface->GetFinish();
     340 
     341            aMaterialPropertiesTable = OpticalSurface->
     342                                                       GetMaterialPropertiesTable();
     343 

     ///


     344            if (aMaterialPropertiesTable) 
                    {
     345 
     346               if (theFinish == polishedbackpainted ||
     347                   theFinish == groundbackpainted ) 
                       {
     ///
     ///                       for these backpainted finishes if opticalsurface MPT has no RINDEX, StopAndKill
     ///
     348                         Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
     349                         if (Rindex) {
     350                               Rindex2 = Rindex->Value(thePhotonMomentum);
     351                         }
     352                         else {
     353                               theStatus = NoRINDEX;
     354                               if ( verboseLevel > 0) BoundaryProcessVerbose();
     355                               aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
     356                               aParticleChange.ProposeTrackStatus(fStopAndKill);
     357                               return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     358                         }
     359               }
     ///
     ///
     ///               set 
     ///                      theReflectivity   (calulate if no REFLECTIVITY and have REALRINDEX IMAGINARYRINDEX) 
     ///                      theEfficiency 
     ///                      theTransmittance
     ///
     ///                if theModel unified  set the below prob from props, otherwise default to 0.
     ///
     ///                        prob_sl  SPECULARLOBECONSTANT
     ///                        prob_ss  SPECULARSPIKECONSTANT
     ///                        prob_bs  BACKSCATTERCONSTANT
     ///  
     ///
     360 
     361               PropertyPointer =
     362                                  aMaterialPropertiesTable->GetProperty("REFLECTIVITY");
     363               PropertyPointer1 =
     364                                  aMaterialPropertiesTable->GetProperty("REALRINDEX");
     365               PropertyPointer2 =
     366                                  aMaterialPropertiesTable->GetProperty("IMAGINARYRINDEX");
     367 
     368               iTE = 1;
     369               iTM = 1;
     370 
     371               if (PropertyPointer) {
     372 
     373                  theReflectivity =
     374                                      PropertyPointer->Value(thePhotonMomentum);
     375 
     376               } else if (PropertyPointer1 && PropertyPointer2) {
     377 
     378                   CalculateReflectivity();
     379 
     380               }
     381 
     382               PropertyPointer =
     383                                    aMaterialPropertiesTable->GetProperty("EFFICIENCY");
     384               if (PropertyPointer) {
     385                       theEfficiency =
     386                                      PropertyPointer->Value(thePhotonMomentum);
     387               }
     388 
     389               PropertyPointer =
     390                                  aMaterialPropertiesTable->GetProperty("TRANSMITTANCE");
     391               if (PropertyPointer) {
     392                       theTransmittance =
     393                                        PropertyPointer->Value(thePhotonMomentum);
     394               }
     395 
     396               if ( theModel == unified ) 
                       {
     397                    PropertyPointer =
     398                                      aMaterialPropertiesTable->GetProperty("SPECULARLOBECONSTANT");
     399                    if (PropertyPointer) {
     400                          prob_sl =
     401                                    PropertyPointer->Value(thePhotonMomentum);
     402                    } else {
     403                          prob_sl = 0.0;
     404                    }
     405 
     406                    PropertyPointer =
     407                                      aMaterialPropertiesTable->GetProperty("SPECULARSPIKECONSTANT");
     408                    if (PropertyPointer) {
     409                          prob_ss =
     410                                    PropertyPointer->Value(thePhotonMomentum);
     411                    } else {
     412                          prob_ss = 0.0;
     413                    }
     414 
     415                    PropertyPointer =
     416                                      aMaterialPropertiesTable->GetProperty("BACKSCATTERCONSTANT");
     417                    if (PropertyPointer) {
     418                          prob_bs =
     419                                    PropertyPointer->Value(thePhotonMomentum);
     420                    } else {
     421                          prob_bs = 0.0;
     422                    }
     423              }          // unified
     424        }     // with MPT
     425            else if (theFinish == polishedbackpainted ||
     426                     theFinish == groundbackpainted ) {
     ///
     ///              StopAndKill backpainted without MPT
     ///
     427                       aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
     428                       aParticleChange.ProposeTrackStatus(fStopAndKill);
     429                       return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     430            }
     431         }   // with OpticalSurface



     433         if (type == dielectric_dielectric ) {
     434             if (theFinish == polished || theFinish == ground ) 
                     {
     435 
     436                  if (Material1 == Material2){
     437                       theStatus = SameMaterial;
     438                       if ( verboseLevel > 0) BoundaryProcessVerbose();
     439                       return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     440                  }
     441                  aMaterialPropertiesTable =
     442                                               Material2->GetMaterialPropertiesTable();
     443                  if (aMaterialPropertiesTable)
     444                                                Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
     445                  if (Rindex) {
     446                                               Rindex2 = Rindex->Value(thePhotonMomentum);
     447                  }
     448                  else {
     449                              theStatus = NoRINDEX;
     450                              if ( verboseLevel > 0) BoundaryProcessVerbose();
     451                              aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
     452                              aParticleChange.ProposeTrackStatus(fStopAndKill);
     453                              return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     454                  }
     455             }
     456         }


     ///
     ///   the chances of such "fortran style?"  coding doing the correct thing seem minimal
     ///

     457 
     458     if (type == dielectric_metal) 
             {
     459 
     460           DielectricMetal();
     461 
     462           // Uncomment the following lines if you wish to have 
     463           //         Transmission instead of Absorption
     464           // if (theStatus == Absorption) {
     465           //    return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     466           // }
     467 
     468     }
     469     else if (type == dielectric_LUT) 
             {
     470 
     471           DielectricLUT();
     472 
     473     }
     474     else if (type == dielectric_dichroic) {
     475 
     476           DielectricDichroic();
     477 
     478     }
     479     else if (type == dielectric_dielectric) {
     480 
     481           if ( theFinish == polishedbackpainted ||
     482                theFinish == groundbackpainted ) {
     483                DielectricDielectric();
     484           }
     485           else {
     486                if ( !G4BooleanRand(theReflectivity) ) {
     487                      DoAbsorption();
     488                }
     489                else {
     490                       if ( theFinish == polishedfrontpainted ) {
     491                            DoReflection();
     492                       }
     493                       else if ( theFinish == groundfrontpainted ) {
     494                            theStatus = LambertianReflection;
     495                            DoReflection();
     496                       }
     497                       else {
     498                            DielectricDielectric();
     499                       }
     500                }
     501           }
     502     }
     503     else {
     504 
     505       G4cerr << " Error: G4BoundaryProcess: illegal boundary type " << G4endl;
     506       return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     507 
     508     }


     510         NewMomentum = NewMomentum.unit();
     511         NewPolarization = NewPolarization.unit();
     ...
     519         aParticleChange.ProposeMomentumDirection(NewMomentum);
     520         aParticleChange.ProposePolarization(NewPolarization);
     521 
     522         if ( theStatus == FresnelRefraction ) {
     523            G4MaterialPropertyVector* groupvel =
     524                    Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
     525            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
     526            aParticleChange.ProposeVelocity(finalVelocity);
     527         }
     528 
     529         if ( theStatus == Detection ) InvokeSD(pStep);
     530 
     531         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     532 }



::

    261 inline
    262 G4bool G4OpBoundaryProcess::G4BooleanRand(const G4double prob) const
    263 {
    264   /* Returns a random boolean variable with the specified probability */
    265 
    266   return (G4UniformRand() < prob);
    267 }
    ...
    282 inline
    283 void G4OpBoundaryProcess::ChooseReflection()
    284 {
    285                  G4double rand = G4UniformRand();
    286                  if ( rand >= 0.0 && rand < prob_ss ) {
    287                     theStatus = SpikeReflection;
    288                     theFacetNormal = theGlobalNormal;
    289                  }
    290                  else if ( rand >= prob_ss &&
    291                            rand <= prob_ss+prob_sl) {
    292                     theStatus = LobeReflection;
    293                  }
    294                  else if ( rand > prob_ss+prob_sl &&
    295                            rand < prob_ss+prob_sl+prob_bs ) {
    296                     theStatus = BackScattering;
    297                  }
    298                  else {
    299                     theStatus = LambertianReflection;
    300                  }
    301 }

    ///
    ///    default with prob_ss = prob_sl = prob_bs = 0. will be LambertianReflection
    ///



     55 inline G4ThreeVector G4RandomDirection()
     56 {
     57   G4double cosTheta  = 2.*G4UniformRand()-1.;
     58   G4double sinTheta2 = 1. - cosTheta*cosTheta;
     59   if( sinTheta2 < 0.)  sinTheta2 = 0.;
     60   G4double sinTheta  = std::sqrt(sinTheta2);
     61   G4double phi       = CLHEP::twopi*G4UniformRand();
     62   return G4ThreeVector(sinTheta*std::cos(phi),
     63                        sinTheta*std::sin(phi), cosTheta).unit();
     64 }



     52 // ---------------------------------------------------------------------------
     53 // Returns a random lambertian unit vector
     54 //
     55 inline G4ThreeVector G4LambertianRand(const G4ThreeVector& normal)
     56 {
     57   G4ThreeVector vect;
     58   G4double ndotv;
     59 
     60   do
     61   {
     62     vect = G4RandomDirection();
     63     ndotv = normal * vect;
     64 
     65     if (ndotv < 0.0)
     66     {
     67       vect = -vect;
     68       ndotv = -ndotv;
     69     }
     70 
     71   } while (!(G4UniformRand() < ndotv));
     72 
     73   return vect;
     74 }
     75 



    325 inline
    326 void G4OpBoundaryProcess::DoReflection()
    327 {
    328         if ( theStatus == LambertianReflection ) {
    329 
    330           NewMomentum = G4LambertianRand(theGlobalNormal);
    331           theFacetNormal = (NewMomentum - OldMomentum).unit();
    332 
    333         }
    334         else if ( theFinish == ground ) {
    335 
    336       theStatus = LobeReflection;
    337           if ( PropertyPointer1 && PropertyPointer2 ){
    338           } else {
    339              theFacetNormal =
    340                  GetFacetNormal(OldMomentum,theGlobalNormal);
    341           }
    342           G4double PdotN = OldMomentum * theFacetNormal;
    343           NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
    344 
    345         }
    346         else {
    347 
    348           theStatus = SpikeReflection;
    349           theFacetNormal = theGlobalNormal;
    350           G4double PdotN = OldMomentum * theFacetNormal;
    351           NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
    352 
    353         }
    354         G4double EdotN = OldPolarization * theFacetNormal;
    355         NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
    356 }




::

     689 void G4OpBoundaryProcess::DielectricMetal()
     690 {
     691     G4int n = 0;
     693     do {
     695            n++;
     697            if( !G4BooleanRand(theReflectivity) && n == 1 ) {
     702                DoAbsorption();
     704                break;
     706            }
     707            else {
     ...
     719              if ( theModel == glisur || theFinish == polished ) {
     721                 DoReflection();
     723              } else {
     724 
     725                 if ( n == 1 ) ChooseReflection();
     726 
     727                 if ( theStatus == LambertianReflection ) {
     728                    DoReflection();
     729                 }
     730                 else if ( theStatus == BackScattering ) {
     731                    NewMomentum = -OldMomentum;
     732                    NewPolarization = -OldPolarization;
     733                 }
     734                 else {
     735 
     736                    if(theStatus==LobeReflection){
     737                      if ( PropertyPointer1 && PropertyPointer2 ){
     738                      } else {
     739                         theFacetNormal =
     740                             GetFacetNormal(OldMomentum,theGlobalNormal);
     741                      }
     742                    }
     ...
     744                    G4double PdotN = OldMomentum * theFacetNormal;
     745                    NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
     746                    G4double EdotN = OldPolarization * theFacetNormal;
     747 
     748                    G4ThreeVector A_trans, A_paral;
     749 
     ///
     ///      horrendous coding style, depending on smth lying around in the scope
     ///      eg from some random prior photon bounce
     ////
     ///
     750                    if (sint1 > 0.0 ) {
     751                       A_trans = OldMomentum.cross(theFacetNormal);
     752                       A_trans = A_trans.unit();
     753                    } else {
     754                       A_trans  = OldPolarization;
     755                    }
     756                    A_paral   = NewMomentum.cross(A_trans);
     757                    A_paral   = A_paral.unit();
     758 
     759                    if(iTE>0&&iTM>0) {
     760                      NewPolarization =
     761                            -OldPolarization + (2.*EdotN)*theFacetNormal;
     762                    } else if (iTE>0) {
     763                      NewPolarization = -A_trans;
     764                    } else if (iTM>0) {
     765                      NewPolarization = -A_paral;
     766                    }
     767 
     768                 }
     769 
     770              }
     771 
     772              OldMomentum = NewMomentum;
     773              OldPolarization = NewPolarization;
     774 
     775        }
     776 
     777     } while (NewMomentum * theGlobalNormal < 0.0);
     778 }






g4dae::

    248 void G4DAEWriteStructure::
    249 OpticalSurfaceWrite(xercesc::DOMElement* targetElement,
    250                     const G4OpticalSurface* const surf)
    251 {
    252    xercesc::DOMElement* optElement = NewElement("opticalsurface");
    253    G4OpticalSurfaceModel smodel = surf->GetModel();
    254    G4double sval = (smodel==glisur) ? surf->GetPolish() : surf->GetSigmaAlpha();
    255 
    256    optElement->setAttributeNode(NewNCNameAttribute("name", surf->GetName()));
    257    optElement->setAttributeNode(NewAttribute("model", smodel));
    258    optElement->setAttributeNode(NewAttribute("finish", surf->GetFinish()));
    259    optElement->setAttributeNode(NewAttribute("type", surf->GetType()));
    260    optElement->setAttributeNode(NewAttribute("value", sval));
    261 
    262    G4MaterialPropertiesTable* ptable = surf->GetMaterialPropertiesTable();
    263    PropertyWrite( optElement, ptable );
    264 
    265    targetElement->appendChild(optElement);
    266 }


::

    152930       <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" type="0" value="1">
    152931         <matrix coldim="2" name="REFLECTIVITY0xc04f6a8">1.5e-06 0 6.5e-06 0</matrix>
    152932         <property name="REFLECTIVITY" ref="REFLECTIVITY0xc04f6a8"/>
    152933         <matrix coldim="2" name="RINDEX0xc33da70">1.5e-06 0 6.5e-06 0</matrix>
    152934         <property name="RINDEX" ref="RINDEX0xc33da70"/>
    152935       </opticalsurface>
    152936       <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface" type="0" value="1">
    152937         <matrix coldim="2" name="BACKSCATTERCONSTANT0xc28d340">1.55e-06 0 6.2e-06 0 1.033e-05 0 1.55e-05 0</matrix>
    152938         <property name="BACKSCATTERCONSTANT" ref="BACKSCATTERCONSTANT0xc28d340"/>
    152939         <matrix coldim="2" name="REFLECTIVITY0xc563328">1.55e-06 0.0393 1.771e-06 0.0393 2.066e-06 0.0394 2.48e-06 0.03975 2.755e-06 0.04045 3.01e-06 0.04135 3.542e-06 0.0432 4.133e-06 0.04655 4.       959e-06 0.0538 6.2e-06 0.067 1.033e-05 0.114 1.55e-05 0.173</matrix>
    152940         <property name="REFLECTIVITY" ref="REFLECTIVITY0xc563328"/>
    152941         <matrix coldim="2" name="SPECULARLOBECONSTANT0xbfa85d0">1.55e-06 0 6.2e-06 0 1.033e-05 0 1.55e-05 0</matrix>
    152942         <property name="SPECULARLOBECONSTANT" ref="SPECULARLOBECONSTANT0xbfa85d0"/>
    152943         <matrix coldim="2" name="SPECULARSPIKECONSTANT0xc03fc20">1.55e-06 0 6.2e-06 0 1.033e-05 0 1.55e-05 0</matrix>
    152944         <property name="SPECULARSPIKECONSTANT" ref="SPECULARSPIKECONSTANT0xc03fc20"/>
    152945       </opticalsurface>

::

    simon:DayaBay_VGDX_20140414-1300 blyth$ grep \<optical  g4_00.dae


          <opticalsurface finish="0" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop"          type="0" value="0">   **
          <opticalsurface finish="0" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot"          type="0" value="0">   **

          <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1"     type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2"     type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface"             type="0" value="1">

          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearIWSCurtainSurface"   type="0" value="0.2">   **
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSLinerSurface"      type="0" value="0.2"> **
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface"     type="0" value="0.2"> ** 

          ## above are border surfaces tied to pairs of volumes below just tied to single volumes

          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface"              type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface"        type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface"   type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtBaseRingSurface"  type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtRib1Surface"      type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtRib2Surface"      type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtRib3Surface"        type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__LegInIWSTubSurface"      type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__TablePanelSurface"       type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__SupportRib1Surface"      type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__SupportRib5Surface"      type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__SlopeRib1Surface"        type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__SlopeRib5Surface"        type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__ADVertiCableTraySurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__ShortParCableTraySurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearInnInPiperSurface"   type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearInnOutPiperSurface"  type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__LegInOWSTubSurface"       type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__UnistrutRib6Surface"     type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__UnistrutRib7Surface"     type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib3Surface"      type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib5Surface"      type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib4Surface"      type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib1Surface"      type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib2Surface"      type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib8Surface"      type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib9Surface"       type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__TopShortCableTraySurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__TopCornerCableTraySurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__VertiCableTraySurface"     type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOutInPiperSurface"    type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOutOutPiperSurface"   type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__LegInDeadTubSurface"       type="0" value="1">

    simon:DayaBay_VGDX_20140414-1300 blyth$ 


All optical surfaces are ground except the two polished mirrors, all dielectric_metal 

g4ten::

     61 enum G4OpticalSurfaceFinish
     62 {
     63    polished,                    // smooth perfectly polished surface
     64    polishedfrontpainted,        // smooth top-layer (front) paint
     65    polishedbackpainted,         // same is 'polished' but with a back-paint
     66 
     67    ground,                      // rough surface
     68    groundfrontpainted,          // rough top-layer (front) paint
     69    groundbackpainted,           // same as 'ground' but with a back-paint


/data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/materials/include/G4OpticalSurface.hh::

     61 enum G4OpticalSurfaceFinish
     62 {
     63    polished,                    // smooth perfectly polished surface
     64    polishedfrontpainted,        // smooth top-layer (front) paint
     65    polishedbackpainted,         // same is 'polished' but with a back-paint
     66    ground,                      // rough surface
     67    groundfrontpainted,          // rough top-layer (front) paint
     68    groundbackpainted            // same as 'ground' but with a back-paint
     69 };
     70 
     71 enum G4OpticalSurfaceModel
     72 {
     73    glisur,                      // original GEANT3 model
     74    unified                      // UNIFIED model
     75 };



::

     65 enum G4SurfaceType
     66 {
     67    dielectric_metal,            // dielectric-metal interface
     68    dielectric_dielectric,       // dielectric-dielectric interface
     69    dielectric_LUT,              // dielectric-Look-Up-Table interface
     70    dielectric_dichroic,         // dichroic filter interface
     71    firsov,                      // for Firsov Process
     72    x_ray                        // for x-ray mirror process
     73 };


     66 enum G4SurfaceType
     67 {
     68    dielectric_metal,            // dielectric-metal interface
     69    dielectric_dielectric,       // dielectric-dielectric interface
     70    firsov,                      // for Firsov Process
     71    x_ray                        // for x-ray mirror process
     72 };



All optical surfaces are using unified::

    099 enum G4OpticalSurfaceModel
    100 {
    101    glisur,                      // original GEANT3 model
    102    unified,                     // UNIFIED model
    103    LUT,                         // Look-Up-Table model
    104    dichroic                     // dichroic filter
    105 };



assimp-fork::

    1387 void ColladaLoader::BuildMaterialsExtras( ColladaParser& pParser, const Collada::Material& material , aiMaterial* mat )
    1388 {
    1389     if(material.mExtra)
    1390     {
    1391         Collada::ExtraProperties::ExtraPropertiesMap& epm = material.mExtra->mProperties ;
    1392 
    1393         for(Collada::ExtraProperties::ExtraPropertiesMap::iterator it=epm.begin() ; it != epm.end() ; it++ )
    1394         {
    1395             const char* key = it->first.c_str() ;
    1396             const char* val = it->second.c_str() ;
    1397 
    1398             const char* prefix = "g4dae_" ;
    1399             if(strncmp(key, prefix, strlen(prefix)) == 0)
    1400             {
    1401                 aiString sval(val);
    1402                 mat->AddProperty( &sval, key);
    1403                 //printf("ColladaLoader::BuildMaterialsExtras AddProperty [%s] [%s] \n", val, key );
    1404             }
    1405             else
    1406             {
    1407                 if(pParser.mDataLibrary.find(it->second) != pParser.mDataLibrary.end())
    1408                 {
    1409                     //printf("ColladaLoader::BuildMaterialsExtras AddProperty<float> [%s] [%s] \n", val, key );
    1410                     Collada::Data& data = pParser.mDataLibrary[it->second];
    1411                     mat->AddProperty<float>( data.mValues.data(), data.mValues.size(), key);
    1412                 }
    1413                 else
    1414                 {
    1415                     printf("ColladaLoader::BuildMaterialsExtras BAD DATA REF  key %s val %s \n", key, val );
    1416                 }
    1417             }
    1418 
    1419         }
    1420     }
    1421 }
    1422 #endif


Suspect are missing the model/type/... attributes in the import::

    2551 void ColladaParser::FakeExtraSkinSurface(Collada::SkinSurface& pSkinSurface,  Collada::Material& pMaterial)
    2552 {
    2553     // hijack Assimp material infrastructure to hold skin surface properties
    2554     if(!pMaterial.mExtra )
    2555         pMaterial.mExtra = new Collada::ExtraProperties();
    2556 
    2557     if(pSkinSurface.mOpticalSurface)
    2558     {
    2559         std::map<std::string,std::string>& ssm = pSkinSurface.mOpticalSurface->mExtra->mProperties ;
    2560         pMaterial.mExtra->mProperties.insert( ssm.begin(), ssm.end() );
    2561         pMaterial.mExtra->mProperties[g4dae_skinsurface_volume] = pSkinSurface.mVolume ;
    2562     }
    2563 }
    2564 


dielectric_metal reflection
----------------------------

* :google:`dielectric metal reflection polarization`

* :google:`reflection metal complex refractive index polarization`

* http://physics.stackexchange.com/questions/10911/polarization-and-mirrors


Plasma Physics 
~~~~~~~~~~~~~~~~

Polarization measurements used to understand plasmas. So they need to
have a detailed understanding of ordinary polarization from reflection too.
 
Tokamak MSE (Motional Stark Effect) : extensive polarized reflection calcs

* https://www.psfc.mit.edu/~sscott/MSEmemos/mse_memo_83c.pdf

* https://en.wikipedia.org/wiki/Stark_effect E-field effect on spectral lines

* http://www.novaphotonics.com/MSE%20Diagnostic/MSE_Diagnostic.html

    The motional Stark effect diagnostic (MSE) was invented by Dr. Fred M.
    Levinton in 1989 and has subsequently become the worldwide touchstone for the
    measurement of internal magnetic fields in high temperature plasma experiments.

    The concept of MSE is based on the measurement of the polarization state of
    light emitted from an atomic hydrogen beam as it passes through a magnetized
    plasma. As the beam moves through the magnetic field, B, at high velocity, v,
    it experiences in its reference frame an electric field, E = v x B. The beam
    atoms are excited via collisions with background plasma, and emit visible light
    which is split into nine spectral lines by the electric field in a phenomemon
    known as the Stark effect. The polarization of the light is aligned with
    respect to the magnetic field, so a measurement of the polarization orientation
    can be used to determine the magnetic field direction in the plasma.




Time Updating
--------------

::

    264 G4Step* G4ParticleChange::UpdateStepForAlongStep(G4Step* pStep)
    265 {
    266   // A physics process always calculates the final state of the
    267   // particle relative to the initial state at the beginning
    268   // of the Step, i.e., based on information of G4Track (or
    269   // equivalently the PreStepPoint). 
    270   // So, the differences (delta) between these two states have to be
    271   // calculated and be accumulated in PostStepPoint. 
    272 
    273   // Take note that the return type of GetMomentumDirectionChange is a
    274   // pointer to G4ParticleMometum. Also it is a normalized 
    275   // momentum vector.
    276 


    348 G4Step* G4ParticleChange::UpdateStepForPostStep(G4Step* pStep)
    349 {
    350   // A physics process always calculates the final state of the particle
    351 
    352   // Take note that the return type of GetMomentumChange is a
    353   // pointer to G4ParticleMometum. Also it is a normalized 
    354   // momentum vector.
    ...
    382   // update position and time
    383   pPostStepPoint->SetPosition( thePositionChange  );
    384   pPostStepPoint->AddGlobalTime(theTimeChange - theLocalTime0);
    385   pPostStepPoint->SetLocalTime( theTimeChange );
    386   pPostStepPoint->SetProperTime( theProperTimeChange  );
    387 

    simon:cfg4 blyth$ ggv-;ggv-box-test --cfg4 --dbg 
    (lldb) b "G4ParticleChange::UpdateStepForPostStep(G4Step*)" 

::

    (lldb) b "G4StepPoint::SetGlobalTime(double)" 
    Breakpoint 1: 3 locations.
    (lldb) c
    Process 54677 resuming
    Process 54677 stopped
    * thread #1: tid = 0x2c7642, 0x00000001027104d1 libG4tracking.dylib`G4StepPoint::SetGlobalTime(this=0x0000000108658e50, aValue=0.10000000149011612) + 17 at G4StepPoint.icc:60, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x00000001027104d1 libG4tracking.dylib`G4StepPoint::SetGlobalTime(this=0x0000000108658e50, aValue=0.10000000149011612) + 17 at G4StepPoint.icc:60
       57   
       58   inline 
       59    void G4StepPoint::SetGlobalTime(const G4double aValue)
    -> 60    { fGlobalTime = aValue; }
       61   
       62   inline 
       63    void G4StepPoint::AddGlobalTime(const G4double aValue)
    (lldb) p aValue
    (G4double) $0 = 0.10000000149011612
    (lldb) bt
    * thread #1: tid = 0x2c7642, 0x00000001027104d1 libG4tracking.dylib`G4StepPoint::SetGlobalTime(this=0x0000000108658e50, aValue=0.10000000149011612) + 17 at G4StepPoint.icc:60, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x00000001027104d1 libG4tracking.dylib`G4StepPoint::SetGlobalTime(this=0x0000000108658e50, aValue=0.10000000149011612) + 17 at G4StepPoint.icc:60
        frame #1: 0x000000010270fa39 libG4tracking.dylib`G4Step::InitializeStep(this=0x0000000108658df0, aValue=0x0000000114f96b80) + 137 at G4Step.icc:200
        frame #2: 0x000000010270eb5e libG4tracking.dylib`G4SteppingManager::SetInitialStep(this=0x0000000108658c60, valueTrack=0x0000000114f96b80) + 1774 at G4SteppingManager.cc:351
        frame #3: 0x00000001027255fa libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000108658c20, apValueG4Track=0x0000000114f96b80) + 538 at G4TrackingManager.cc:89
        frame #4: 0x0000000102602e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000108658b90, anEvent=0x00000001148377b0) + 3188 at G4EventManager.cc:185
        frame #5: 0x0000000102603b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x0000000108658b90, anEvent=0x00000001148377b0) + 47 at G4EventManager.cc:336
        frame #6: 0x0000000102530c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010834dd50, i_event=3) + 69 at G4RunManager.cc:399
        frame #7: 0x0000000102530ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010834dd50, n_event=50, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #8: 0x000000010252f8e4 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010834dd50, n_event=50, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #9: 0x0000000100f140dd libcfg4.dylib`CfG4::propagate(this=0x0000000108212ec0) + 685 at CfG4.cc:163
        frame #10: 0x000000010000cd45 cfg4Test`main(argc=20, argv=0x00007fff5fbfdd58) + 101 at cfg4Test.cc:9
        frame #11: 0x00007fff8568c5fd libdyld.dylib`start + 1
        frame #12: 0x00007fff8568c5fd libdyld.dylib`start + 1
    (lldb) 

::

    (lldb) c
    Process 62039 resuming
    Process 62039 stopped
    * thread #1: tid = 0x2ca879, 0x000000010571c001 libG4track.dylib`G4StepPoint::SetLocalTime(this=0x0000000108372970, aValue=1.0551967349070439) + 17 at G4StepPoint.icc:48, queue = 'com.apple.main-thread', stop reason = breakpoint 1.3
        frame #0: 0x000000010571c001 libG4track.dylib`G4StepPoint::SetLocalTime(this=0x0000000108372970, aValue=1.0551967349070439) + 17 at G4StepPoint.icc:48
       45   
       46   inline 
       47    void G4StepPoint::SetLocalTime(const G4double aValue)
    -> 48    { fLocalTime = aValue; }
       49   
       50   inline 
       51    void G4StepPoint::AddLocalTime(const G4double aValue)
    (lldb) p aValue
    (G4double) $1 = 1.0551967349070439
    (lldb) bt
    * thread #1: tid = 0x2ca879, 0x000000010571c001 libG4track.dylib`G4StepPoint::SetLocalTime(this=0x0000000108372970, aValue=1.0551967349070439) + 17 at G4StepPoint.icc:48, queue = 'com.apple.main-thread', stop reason = breakpoint 1.3
      * frame #0: 0x000000010571c001 libG4track.dylib`G4StepPoint::SetLocalTime(this=0x0000000108372970, aValue=1.0551967349070439) + 17 at G4StepPoint.icc:48
        frame #1: 0x000000010571861a libG4track.dylib`G4ParticleChange::UpdateStepForPostStep(this=0x00000001083ca298, pStep=0x0000000108372840) + 410 at G4ParticleChange.cc:385
        frame #2: 0x00000001027132ba libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x00000001083726b0, np=4) + 186 at G4SteppingManager2.cc:533
        frame #3: 0x00000001027130d8 libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x00000001083726b0) + 232 at G4SteppingManager2.cc:502
        frame #4: 0x000000010270e28e libG4tracking.dylib`G4SteppingManager::Stepping(this=0x00000001083726b0) + 798 at G4SteppingManager.cc:209
        frame #5: 0x000000010272592d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000108372670, apValueG4Track=0x00000001098cba70) + 1357 at G4TrackingManager.cc:126
        frame #6: 0x0000000102602e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x00000001083725e0, anEvent=0x0000000109151200) + 3188 at G4EventManager.cc:185
        frame #7: 0x0000000102603b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x00000001083725e0, anEvent=0x0000000109151200) + 47 at G4EventManager.cc:336
        frame #8: 0x0000000102530c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x00000001082676f0, i_event=0) + 69 at G4RunManager.cc:399
        frame #9: 0x0000000102530ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x00000001082676f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #10: 0x000000010252f8e4 libG4run.dylib`G4RunManager::BeamOn(this=0x00000001082676f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #11: 0x0000000100f140dd libcfg4.dylib`CfG4::propagate(this=0x0000000108212ec0) + 685 at CfG4.cc:163
        frame #12: 0x000000010000cd45 cfg4Test`main(argc=21, argv=0x00007fff5fbfdd38) + 101 at cfg4Test.cc:9
        frame #13: 0x00007fff8568c5fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 1
    frame #1: 0x000000010571861a libG4track.dylib`G4ParticleChange::UpdateStepForPostStep(this=0x00000001083ca298, pStep=0x0000000108372840) + 410 at G4ParticleChange.cc:385
       382    // update position and time
       383    pPostStepPoint->SetPosition( thePositionChange  );
       384    pPostStepPoint->AddGlobalTime(theTimeChange - theLocalTime0);
    -> 385    pPostStepPoint->SetLocalTime( theTimeChange );           
       386    pPostStepPoint->SetProperTime( theProperTimeChange  );
       387  
       388    if (isParentWeightProposed ){
    (lldb) p theTimeChange
    (G4double) $2 = 1.0551967349070439
    (lldb) p theLocalTime0
    (G4double) $3 = 1.0551967349070439
    (lldb) p thePositionChange
    (G4ThreeVector) $4 = (dx = -86.375713348388671, dy = 1.8436908721923828, dz = 100)
    (lldb) p theProperTimeChange
    (G4double) $5 = 0


::

    077 class G4ParticleChange: public G4VParticleChange

    161     void     ProposeGlobalTime(G4double t);
    162     void     ProposeLocalTime(G4double t);
    163     //  Get/Propose the final global/local Time
    164     // NOTE: DO NOT INVOKE both methods in a step
    165     //       Each method affects both local and global time 

::

    (lldb) b "G4ParticleChange::ProposeLocalTime(double)" 

       159  inline 
       160    void G4ParticleChange::ProposeLocalTime(G4double t)
       161  {
    -> 162    theTimeChange = t;
       163  }
       164     
       165  inline
    (lldb) p t 
    (G4double) $6 = 2.0278695996039202

::

    (lldb) bt
    * thread #1: tid = 0x2ca879, 0x0000000102a2cde1 libG4processes.dylib`G4ParticleChange::ProposeLocalTime(this=0x00000001083b1ef0, t=2.0278695996039202) + 17 at G4ParticleChange.icc:162, queue = 'com.apple.main-thread', stop reason = breakpoint 3.1
      * frame #0: 0x0000000102a2cde1 libG4processes.dylib`G4ParticleChange::ProposeLocalTime(this=0x00000001083b1ef0, t=2.0278695996039202) + 17 at G4ParticleChange.icc:162
        frame #1: 0x0000000103b76e43 libG4processes.dylib`G4Transportation::AlongStepDoIt(this=0x00000001083b1cd0, track=0x00000001098cba70, stepData=0x0000000108372840) + 467 at G4Transportation.cc:561
        frame #2: 0x000000010271294f libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x00000001083726b0) + 223 at G4SteppingManager2.cc:417
        frame #3: 0x000000010270e168 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x00000001083726b0) + 504 at G4SteppingManager.cc:191
        frame #4: 0x000000010272592d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000108372670, apValueG4Track=0x00000001098cba70) + 1357 at G4TrackingManager.cc:126
        frame #5: 0x0000000102602e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x00000001083725e0, anEvent=0x0000000109151200) + 3188 at G4EventManager.cc:185
        frame #6: 0x0000000102603b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x00000001083725e0, anEvent=0x0000000109151200) + 47 at G4EventManager.cc:336
        frame #7: 0x0000000102530c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x00000001082676f0, i_event=0) + 69 at G4RunManager.cc:399
        frame #8: 0x0000000102530ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x00000001082676f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #9: 0x000000010252f8e4 libG4run.dylib`G4RunManager::BeamOn(this=0x00000001082676f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #10: 0x0000000100f140dd libcfg4.dylib`CfG4::propagate(this=0x0000000108212ec0) + 685 at CfG4.cc:163
        frame #11: 0x000000010000cd45 cfg4Test`main(argc=21, argv=0x00007fff5fbfdd38) + 101 at cfg4Test.cc:9
        frame #12: 0x00007fff8568c5fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 1
    frame #1: 0x0000000103b76e43 libG4processes.dylib`G4Transportation::AlongStepDoIt(this=0x00000001083b1cd0, track=0x00000001098cba70, stepData=0x0000000108372840) + 467 at G4Transportation.cc:561
       558       if ( initialVelocity > 0.0 )  { deltaTime = stepLength/initialVelocity; }
       559  
       560       fCandidateEndGlobalTime   = startTime + deltaTime ;
    -> 561       fParticleChange.ProposeLocalTime(  track.GetLocalTime() + deltaTime) ;
       562    }
       563    else
       564    {
    (lldb) p stepLength
    (G4double) $7 = 200
    (lldb) p initialVelocity
    (G4double) $8 = 205.61897762237669
    (lldb) p deltaTime
    (G4double) $9 = 0.97267286469687608
    (lldb) p startTime
    (G4double) $10 = 1.15519673639716
    (lldb) p track.GetLocalTime()
    (G4double) $11 = 1.0551967349070439
    (lldb) 


::

    In [1]: 1.055196+0.97267
    Out[1]: 2.027866

::

    525 G4VParticleChange* G4Transportation::AlongStepDoIt( const G4Track& track,
    526                                                     const G4Step&  stepData )
    527 {
    ...
    542   G4double deltaTime = 0.0 ;
    548   G4double startTime = track.GetGlobalTime() ;
    549  
    550   if (!fEndGlobalTimeComputed)
    551   {
    552      // The time was not integrated .. make the best estimate possible
    553      //
    554      G4double initialVelocity = stepData.GetPreStepPoint()->GetVelocity();
    555      G4double stepLength      = track.GetStepLength();
    556 
    557      deltaTime= 0.0;  // in case initialVelocity = 0 
    558      if ( initialVelocity > 0.0 )  { deltaTime = stepLength/initialVelocity; }
    559 
    560      fCandidateEndGlobalTime   = startTime + deltaTime ;
    561      fParticleChange.ProposeLocalTime(  track.GetLocalTime() + deltaTime) ;
    562   }
    563   else
    564   {
    565      deltaTime = fCandidateEndGlobalTime - startTime ;
    566      fParticleChange.ProposeGlobalTime( fCandidateEndGlobalTime ) ;
    567   }
    568 
    569 
    570   // Now Correct by Lorentz factor to get delta "proper" Time
    571  
    572   G4double  restMass       = track.GetDynamicParticle()->GetMass() ;
    573   G4double deltaProperTime = deltaTime*( restMass/track.GetTotalEnergy() ) ;
    574 
    575   fParticleChange.ProposeProperTime(track.GetProperTime() + deltaProperTime) ;
    576   //fParticleChange. ProposeTrueStepLength( track.GetStepLength() ) ;



    (lldb) b "G4Track::SetVelocity(double)"
    Breakpoint 7: 2 locations.
    (lldb) c
    Process 62039 resuming
    Process 62039 stopped
    * thread #1: tid = 0x2ca879, 0x0000000102710ef1 libG4tracking.dylib`G4Track::SetVelocity(this=0x00000001098cba70, val=205.61897762237669) + 17 at G4Track.icc:124, queue = 'com.apple.main-thread', stop reason = breakpoint 7.1
        frame #0: 0x0000000102710ef1 libG4tracking.dylib`G4Track::SetVelocity(this=0x00000001098cba70, val=205.61897762237669) + 17 at G4Track.icc:124
       121     { return fVelocity; }
       122  
       123     inline void  G4Track::SetVelocity(G4double val)
    -> 124     { fVelocity = val; } 
       125  
       126     inline G4bool   G4Track::UseGivenVelocity() const
       127     { return  useGivenVelocity;}
    (lldb) p val
    (G4double) $12 = 205.61897762237669


    




    (lldb) bt
    * thread #1: tid = 0x2ca879, 0x0000000102710ef1 libG4tracking.dylib`G4Track::SetVelocity(this=0x00000001098cba70, val=205.61897762237669) + 17 at G4Track.icc:124, queue = 'com.apple.main-thread', stop reason = breakpoint 7.1
      * frame #0: 0x0000000102710ef1 libG4tracking.dylib`G4Track::SetVelocity(this=0x00000001098cba70, val=205.61897762237669) + 17 at G4Track.icc:124
        frame #1: 0x000000010270f4fe libG4tracking.dylib`G4Step::UpdateTrack(this=0x0000000108372840) + 462 at G4Step.icc:251
        frame #2: 0x0000000102712f56 libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x00000001083726b0) + 1766 at G4SteppingManager2.cc:471
        frame #3: 0x000000010270e168 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x00000001083726b0) + 504 at G4SteppingManager.cc:191
        frame #4: 0x000000010272592d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000108372670, apValueG4Track=0x00000001098cba70) + 1357 at G4TrackingManager.cc:126
        frame #5: 0x0000000102602e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x00000001083725e0, anEvent=0x0000000109151200) + 3188 at G4EventManager.cc:185
        frame #6: 0x0000000102603b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x00000001083725e0, anEvent=0x0000000109151200) + 47 at G4EventManager.cc:336
        frame #7: 0x0000000102530c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x00000001082676f0, i_event=0) + 69 at G4RunManager.cc:399
        frame #8: 0x0000000102530ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x00000001082676f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #9: 0x000000010252f8e4 libG4run.dylib`G4RunManager::BeamOn(this=0x00000001082676f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #10: 0x0000000100f140dd libcfg4.dylib`CfG4::propagate(this=0x0000000108212ec0) + 685 at CfG4.cc:163
        frame #11: 0x000000010000cd45 cfg4Test`main(argc=21, argv=0x00007fff5fbfdd38) + 101 at cfg4Test.cc:9
        frame #12: 0x00007fff8568c5fd libdyld.dylib`start + 1
    (lldb) 
    (lldb) 
    frame #1: 0x000000010270f4fe libG4tracking.dylib`G4Step::UpdateTrack(this=0x0000000108372840) + 462 at G4Step.icc:251
       248  
       249  
       250     // set velocity 
    -> 251     fpTrack->SetVelocity(fpPostStepPoint->GetVelocity());
       252  }
       253  
       254  inline  G4int G4Step::GetNumberOfSecondariesInCurrentStep() const
    (lldb) 

    (lldb) c
    Process 62039 resuming
    Process 62039 stopped
    * thread #1: tid = 0x2ca879, 0x000000010571bdd1 libG4track.dylib`G4StepPoint::SetVelocity(this=0x0000000108372970, v=189.53811491619024) + 17 at G4StepPoint.icc:124, queue = 'com.apple.main-thread', stop reason = breakpoint 8.3
        frame #0: 0x000000010571bdd1 libG4track.dylib`G4StepPoint::SetVelocity(this=0x0000000108372970, v=189.53811491619024) + 17 at G4StepPoint.icc:124
       121  
       122  inline
       123   void G4StepPoint::SetVelocity(G4double v)
    -> 124   {  fVelocity = v; }
       125    
       126  inline
       127   G4double G4StepPoint::GetBeta() const
    (lldb) p v
    (G4double) $15 = 189.53811491619024

     Huh thats very slow...  and the right order to explain the mismatch

     In [8]: 1-189./205.
     Out[8]: 0.07804878048780484

     In [9]: 1-189./202.
     Out[9]: 0.0643564356435643



    (lldb) bt
    * thread #1: tid = 0x2ca879, 0x000000010571bdd1 libG4track.dylib`G4StepPoint::SetVelocity(this=0x0000000108372970, v=189.53811491619024) + 17 at G4StepPoint.icc:124, queue = 'com.apple.main-thread', stop reason = breakpoint 8.3
      * frame #0: 0x000000010571bdd1 libG4track.dylib`G4StepPoint::SetVelocity(this=0x0000000108372970, v=189.53811491619024) + 17 at G4StepPoint.icc:124
        frame #1: 0x00000001057185bc libG4track.dylib`G4ParticleChange::UpdateStepForPostStep(this=0x00000001083ca298, pStep=0x0000000108372840) + 316 at G4ParticleChange.cc:377
        frame #2: 0x00000001027132ba libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x00000001083726b0, np=4) + 186 at G4SteppingManager2.cc:533
        frame #3: 0x00000001027130d8 libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x00000001083726b0) + 232 at G4SteppingManager2.cc:502
        frame #4: 0x000000010270e28e libG4tracking.dylib`G4SteppingManager::Stepping(this=0x00000001083726b0) + 798 at G4SteppingManager.cc:209
        frame #5: 0x000000010272592d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000108372670, apValueG4Track=0x00000001098cba70) + 1357 at G4TrackingManager.cc:126
        frame #6: 0x0000000102602e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x00000001083725e0, anEvent=0x0000000109151200) + 3188 at G4EventManager.cc:185
        frame #7: 0x0000000102603b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x00000001083725e0, anEvent=0x0000000109151200) + 47 at G4EventManager.cc:336
        frame #8: 0x0000000102530c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x00000001082676f0, i_event=0) + 69 at G4RunManager.cc:399
        frame #9: 0x0000000102530ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x00000001082676f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #10: 0x000000010252f8e4 libG4run.dylib`G4RunManager::BeamOn(this=0x00000001082676f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #11: 0x0000000100f140dd libcfg4.dylib`CfG4::propagate(this=0x0000000108212ec0) + 685 at CfG4.cc:163
        frame #12: 0x000000010000cd45 cfg4Test`main(argc=21, argv=0x00007fff5fbfdd38) + 101 at cfg4Test.cc:9
        frame #13: 0x00007fff8568c5fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 1
    frame #1: 0x00000001057185bc libG4track.dylib`G4ParticleChange::UpdateStepForPostStep(this=0x00000001083ca298, pStep=0x0000000108372840) + 316 at G4ParticleChange.cc:377
       374        theVelocityChange = 0.0;
       375      }
       376    }
    -> 377    pPostStepPoint->SetVelocity(theVelocityChange);
       378   
       379     // update polarization
       380    pPostStepPoint->SetPolarization( thePolarizationChange );
    (lldb) 



From Optixrap::

    // mm/ns
    #define SPEED_OF_LIGHT 299.792458f


    In [5]: 299.792458/1.48264   # MO RINDEX(380nm)  // ggv --mat 3   #(0 based)
    Out[5]: 202.20178735229052

    In [6]: 299.792458/1.458     # Py RINDEX(380nm) // ggv --mat 13   #(0 based)
    Out[6]: 205.61896982167355


::

    (lldb) c
    Process 62039 resuming
    Process 62039 stopped
    * thread #1: tid = 0x2ca879, 0x0000000102710831 libG4tracking.dylib`G4StepPoint::SetVelocity(this=0x00000001083728a0, v=189.53811491619024) + 17 at G4StepPoint.icc:124, queue = 'com.apple.main-thread', stop reason = breakpoint 8.1
        frame #0: 0x0000000102710831 libG4tracking.dylib`G4StepPoint::SetVelocity(this=0x00000001083728a0, v=189.53811491619024) + 17 at G4StepPoint.icc:124
       121  
       122  inline
       123   void G4StepPoint::SetVelocity(G4double v)
    -> 124   {  fVelocity = v; }
       125    
       126  inline
       127   G4double G4StepPoint::GetBeta() const
    (lldb) p v
    (G4double) $24 = 189.53811491619024
    (lldb) bt
    * thread #1: tid = 0x2ca879, 0x0000000102710831 libG4tracking.dylib`G4StepPoint::SetVelocity(this=0x00000001083728a0, v=189.53811491619024) + 17 at G4StepPoint.icc:124, queue = 'com.apple.main-thread', stop reason = breakpoint 8.1
      * frame #0: 0x0000000102710831 libG4tracking.dylib`G4StepPoint::SetVelocity(this=0x00000001083728a0, v=189.53811491619024) + 17 at G4StepPoint.icc:124
        frame #1: 0x000000010270fcd1 libG4tracking.dylib`G4Step::InitializeStep(this=0x0000000108372840, aValue=0x00000001098cb880) + 801 at G4Step.icc:219
        frame #2: 0x000000010270eb5e libG4tracking.dylib`G4SteppingManager::SetInitialStep(this=0x00000001083726b0, valueTrack=0x00000001098cb880) + 1774 at G4SteppingManager.cc:351
        frame #3: 0x00000001027255fa libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000108372670, apValueG4Track=0x00000001098cb880) + 538 at G4TrackingManager.cc:89
        frame #4: 0x0000000102602e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x00000001083725e0, anEvent=0x0000000109151200) + 3188 at G4EventManager.cc:185
        frame #5: 0x0000000102603b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x00000001083725e0, anEvent=0x0000000109151200) + 47 at G4EventManager.cc:336
        frame #6: 0x0000000102530c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x00000001082676f0, i_event=0) + 69 at G4RunManager.cc:399
        frame #7: 0x0000000102530ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x00000001082676f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #8: 0x000000010252f8e4 libG4run.dylib`G4RunManager::BeamOn(this=0x00000001082676f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #9: 0x0000000100f140dd libcfg4.dylib`CfG4::propagate(this=0x0000000108212ec0) + 685 at CfG4.cc:163
        frame #10: 0x000000010000cd45 cfg4Test`main(argc=21, argv=0x00007fff5fbfdd38) + 101 at cfg4Test.cc:9
        frame #11: 0x00007fff8568c5fd libdyld.dylib`start + 1
    (lldb) f 1
    frame #1: 0x000000010270fcd1 libG4tracking.dylib`G4Step::InitializeStep(this=0x0000000108372840, aValue=0x00000001098cb880) + 801 at G4Step.icc:219
       216  
       217     // Set Velocity
       218     //  should be placed after SetMaterial for preStep point 
    -> 219      fpPreStepPoint->SetVelocity(fpTrack->CalculateVelocity());
       220    
       221     (*fpPostStepPoint) = (*fpPreStepPoint);
       222   }
    (lldb) 


::

    (lldb) b "G4Track::CalculateVelocity
    Available completions:
        G4Track::CalculateVelocity() const
        G4Track::CalculateVelocityForOpticalPhoton() const
    (lldb) b "G4Track::CalculateVelocityForOpticalPhoton() const"
    Breakpoint 10: where = libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton() const + 24 at G4Track.cc:259, address = 0x0000000105728188
    (lldb) b "G4Track::CalculateVelocity() const"
    Breakpoint 11: where = libG4track.dylib`G4Track::CalculateVelocity() const + 16 at G4Track.cc:225, address = 0x00000001057272f0
    (lldb) 

::

    (lldb) c
    Process 62039 resuming
    Process 62039 stopped
    * thread #1: tid = 0x2ca879, 0x0000000102a2cc31 libG4processes.dylib`G4ParticleChange::ProposeVelocity(this=0x00000001083ca298, finalVelocity=189.53811491619024) + 17 at G4ParticleChange.icc:57, queue = 'com.apple.main-thread', stop reason = breakpoint 9.1
        frame #0: 0x0000000102a2cc31 libG4processes.dylib`G4ParticleChange::ProposeVelocity(this=0x00000001083ca298, finalVelocity=189.53811491619024) + 17 at G4ParticleChange.icc:57
       54   inline
       55     void G4ParticleChange::ProposeVelocity(G4double finalVelocity)
       56   {
    -> 57      theVelocityChange = finalVelocity;
       58      isVelocityChanged = true;
       59   }
       60   
    (lldb) p finalVelocity
    (G4double) $29 = 189.53811491619024
    (lldb) bt
    * thread #1: tid = 0x2ca879, 0x0000000102a2cc31 libG4processes.dylib`G4ParticleChange::ProposeVelocity(this=0x00000001083ca298, finalVelocity=189.53811491619024) + 17 at G4ParticleChange.icc:57, queue = 'com.apple.main-thread', stop reason = breakpoint 9.1
      * frame #0: 0x0000000102a2cc31 libG4processes.dylib`G4ParticleChange::ProposeVelocity(this=0x00000001083ca298, finalVelocity=189.53811491619024) + 17 at G4ParticleChange.icc:57
        frame #1: 0x0000000103b2dfcc libG4processes.dylib`G4OpBoundaryProcess::PostStepDoIt(this=0x00000001083ca280, aTrack=0x00000001098cb880, aStep=0x0000000108372840) + 140 at G4OpBoundaryProcess.cc:171
        frame #2: 0x000000010271327b libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x00000001083726b0, np=4) + 123 at G4SteppingManager2.cc:530
        frame #3: 0x00000001027130d8 libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x00000001083726b0) + 232 at G4SteppingManager2.cc:502
        frame #4: 0x000000010270e28e libG4tracking.dylib`G4SteppingManager::Stepping(this=0x00000001083726b0) + 798 at G4SteppingManager.cc:209
        frame #5: 0x000000010272592d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000108372670, apValueG4Track=0x00000001098cb880) + 1357 at G4TrackingManager.cc:126
        frame #6: 0x0000000102602e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x00000001083725e0, anEvent=0x0000000109151200) + 3188 at G4EventManager.cc:185
        frame #7: 0x0000000102603b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x00000001083725e0, anEvent=0x0000000109151200) + 47 at G4EventManager.cc:336
        frame #8: 0x0000000102530c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x00000001082676f0, i_event=0) + 69 at G4RunManager.cc:399
        frame #9: 0x0000000102530ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x00000001082676f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #10: 0x000000010252f8e4 libG4run.dylib`G4RunManager::BeamOn(this=0x00000001082676f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #11: 0x0000000100f140dd libcfg4.dylib`CfG4::propagate(this=0x0000000108212ec0) + 685 at CfG4.cc:163
        frame #12: 0x000000010000cd45 cfg4Test`main(argc=21, argv=0x00007fff5fbfdd38) + 101 at cfg4Test.cc:9
        frame #13: 0x00007fff8568c5fd libdyld.dylib`start + 1
    (lldb) 


    (lldb) c
    Process 62039 resuming
    Process 62039 stopped
    * thread #1: tid = 0x2ca879, 0x0000000105727321 libG4track.dylib`G4Track::CalculateVelocity(this=0x00000001098cb770) const + 65 at G4Track.cc:229, queue = 'com.apple.main-thread', stop reason = breakpoint 13.1
        frame #0: 0x0000000105727321 libG4track.dylib`G4Track::CalculateVelocity(this=0x00000001098cb770) const + 65 at G4Track.cc:229
       226  
       227    G4double velocity = c_light ;
       228    
    -> 229    G4double mass = fpDynamicParticle->GetMass();
       230  
       231    // special case for photons
       232    if ( is_OpticalPhoton ) return CalculateVelocityForOpticalPhoton();
    (lldb) p velocity
    (G4double) $39 = 299.79245800000001
    (lldb) 




    (lldb) c
    Process 62039 resuming
    Process 62039 stopped
    * thread #1: tid = 0x2ca879, 0x00000001057282b6 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001098cb770) const + 326 at G4Track.cc:281, queue = 'com.apple.main-thread', stop reason = breakpoint 15.1
        frame #0: 0x00000001057282b6 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001098cb770) const + 326 at G4Track.cc:281
       278    }
       279    prev_mat = mat;
       280    
    -> 281    if  (groupvel != 0 ) {
       282      // light velocity = c/(rindex+d(rindex)/d(log(E_phot)))
       283      // values stored in GROUPVEL material properties vector
       284      velocity =  prev_velocity;
    (lldb) p groupvel
    (G4MaterialPropertyVector *) $45 = 0x0000000109900000
    (lldb) p *groupvel
    (G4MaterialPropertyVector) $46 = {
      G4PhysicsVector = {
        type = T_G4PhysicsOrderedFreeVector
        edgeMin = 0.0000015120022870975581
        edgeMax = 0.000020664031256999959
        numberOfNodes = 39
        dataVector = size=39 {
          [0] = 205.84485747217411
          [1] = 205.84485747217411
          [2] = 204.10812440289433
          [3] = 204.10037473605965
          [4] = 204.09974782245047
          [5] = 204.09975045699542
          [6] = 204.09975323998844
          [7] = 202.92166404329791
          [8] = 201.93672443399282
          [9] = 201.93673242076844
          [10] = 201.93674091474074
          [11] = 201.93674996581169
          [12] = 201.21267211183462
          [13] = 200.36385790387294
          [14] = 200.28479911876505
          [15] = 200.10626742977158
          [16] = 200.10628775075023
          [17] = 199.46795765489946


Where did GROUPVEL come from ?::

    simon:source blyth$ find . -name '*.cc' -exec grep -H GROUPVEL {} \;
    ./materials/src/G4MaterialPropertiesTable.cc:// Updated:     2005-05-12 add SetGROUPVEL(), courtesy of
    ./materials/src/G4MaterialPropertiesTable.cc:G4MaterialPropertyVector* G4MaterialPropertiesTable::SetGROUPVEL()
    ./materials/src/G4MaterialPropertiesTable.cc:  // check if "GROUPVEL" already exists
    ./materials/src/G4MaterialPropertiesTable.cc:  itr = MPT.find("GROUPVEL");
    ./materials/src/G4MaterialPropertiesTable.cc:  // add GROUPVEL vector
    ./materials/src/G4MaterialPropertiesTable.cc:  // fill GROUPVEL vector using RINDEX values
    ./materials/src/G4MaterialPropertiesTable.cc:    G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    ./materials/src/G4MaterialPropertiesTable.cc:      G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    ./materials/src/G4MaterialPropertiesTable.cc:        G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    ./materials/src/G4MaterialPropertiesTable.cc:  this->AddProperty( "GROUPVEL", groupvel );
    ./processes/optical/src/G4OpBoundaryProcess.cc:           Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    ./track/src/G4Track.cc:    //  and get new GROUPVELOCITY table if necessary 
    ./track/src/G4Track.cc:      groupvel = mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    ./track/src/G4Track.cc:    // values stored in GROUPVEL material properties vector
    simon:source blyth$ 
    simon:source blyth$ 


     materials/src/G4MaterialPropertiesTable.cc

     38 // Updated:     2005-05-12 add SetGROUPVEL(), courtesy of
     39 //              Horton-Smith (bug report #741), by P. Gumplinger

     G4MaterialPropertyVector* G4MaterialPropertiesTable::SetGROUPVEL()  


Huh does proper timing for photons require GROUPVEL ? Looks like this is auto added by G4MaterialPropertiesTable::SetGROUPVEL()::

     G4OpBoundaryStatus.cc

     533         if ( theStatus == FresnelRefraction || theStatus == Transmission ) {
     534            G4MaterialPropertyVector* groupvel =
     535            Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
     536            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
     537            aParticleChange.ProposeVelocity(finalVelocity);
     538         }

::

    [2016-Mar-04 16:09:06.882047]:info: CfG4::propagate num_g4event 1 num_photons 1 steps_per_photon 10 bounce_max 9
    Process 76399 stopped
    * thread #1: tid = 0x2d1e62, 0x0000000105e2c498 libG4materials.dylib`G4MaterialPropertiesTable::SetGROUPVEL(this=0x000000010858df10) + 56 at G4MaterialPropertiesTable.cc:127, queue = 'com.apple.main-thread', stop reason = breakpoint 16.1
        frame #0: 0x0000000105e2c498 libG4materials.dylib`G4MaterialPropertiesTable::SetGROUPVEL(this=0x000000010858df10) + 56 at G4MaterialPropertiesTable.cc:127
       124  
       125    // check if "GROUPVEL" already exists
       126    MPTiterator itr;
    -> 127    itr = MPT.find("GROUPVEL");
       128    if(itr != MPT.end()) return itr->second;
       129  
       130    // fetch RINDEX data, give up if unavailable
    (lldb) bt
    * thread #1: tid = 0x2d1e62, 0x0000000105e2c498 libG4materials.dylib`G4MaterialPropertiesTable::SetGROUPVEL(this=0x000000010858df10) + 56 at G4MaterialPropertiesTable.cc:127, queue = 'com.apple.main-thread', stop reason = breakpoint 16.1
      * frame #0: 0x0000000105e2c498 libG4materials.dylib`G4MaterialPropertiesTable::SetGROUPVEL(this=0x000000010858df10) + 56 at G4MaterialPropertiesTable.cc:127
        frame #1: 0x000000010572943c libG4track.dylib`G4MaterialPropertiesTable::GetProperty(this=0x000000010858df10, key=0x000000010573818a) + 876 at G4MaterialPropertiesTable.icc:123
        frame #2: 0x0000000105728298 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x000000010b31abc0) const + 296 at G4Track.cc:276
        frame #3: 0x000000010572734d libG4track.dylib`G4Track::CalculateVelocity(this=0x000000010b31abc0) const + 109 at G4Track.cc:232
        frame #4: 0x000000010270fcc5 libG4tracking.dylib`G4Step::InitializeStep(this=0x0000000108556d10, aValue=0x000000010b31abc0) + 789 at G4Step.icc:219
        frame #5: 0x000000010270eb5e libG4tracking.dylib`G4SteppingManager::SetInitialStep(this=0x0000000108556b80, valueTrack=0x000000010b31abc0) + 1774 at G4SteppingManager.cc:351
        frame #6: 0x00000001027255fa libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000108556b40, apValueG4Track=0x000000010b31abc0) + 538 at G4TrackingManager.cc:89
        frame #7: 0x0000000102602e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000108556ab0, anEvent=0x000000010a39c4a0) + 3188 at G4EventManager.cc:185
        frame #8: 0x0000000102603b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x0000000108556ab0, anEvent=0x000000010a39c4a0) + 47 at G4EventManager.cc:336
        frame #9: 0x0000000102530c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010844bb80, i_event=0) + 69 at G4RunManager.cc:399
        frame #10: 0x0000000102530ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010844bb80, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #11: 0x000000010252f8e4 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010844bb80, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #12: 0x0000000100f140dd libcfg4.dylib`CfG4::propagate(this=0x0000000108212ec0) + 685 at CfG4.cc:163
        frame #13: 0x000000010000cd45 cfg4Test`main(argc=21, argv=0x00007fff5fbfdd38) + 101 at cfg4Test.cc:9
        frame #14: 0x00007fff8568c5fd libdyld.dylib`start + 1
    (lldb) 


    key GROUPVEL is special cased

    (lldb) f 1
    frame #1: 0x000000010572943c libG4track.dylib`G4MaterialPropertiesTable::GetProperty(this=0x000000010858df10, key=0x000000010573818a) + 876 at G4MaterialPropertiesTable.icc:123
       120    MPTiterator i;
       121    i = MPT.find(G4String(key));
       122    if ( i != MPT.end() ) return i->second;
    -> 123    if (G4String(key) == "GROUPVEL") return SetGROUPVEL();
       124    return NULL;
       125  }

    (lldb) f 2
    frame #2: 0x0000000105728298 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x000000010b31abc0) const + 296 at G4Track.cc:276
       273    if ((mat != 0) && ((mat != prev_mat)||(groupvel==0))) {
       274      groupvel = 0;
       275      if(mat->GetMaterialPropertiesTable() != 0)
    -> 276        groupvel = mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
       277      update_groupvel = true;
       278    }
       279    prev_mat = mat;
    (lldb) 

    https://en.wikipedia.org/wiki/Group_velocity



EOU
}
g4op-dir(){ echo $(env-home)/geant4/g4op ; }
g4op-cd(){  cd $(g4op-dir); }
g4op-mate(){ mate $(g4op-dir) ; }
g4op-get(){
    g4op-cd

    g4-
    local klss="G4OpBoundaryProcess"
    for kls in $klss ; do 
       cp $(g4-dir)/source/processes/optical/src/$kls.cc  .
       cp $(g4-dir)/source/processes/optical/include/$kls.hh  .
    done 
}


g4op-boundary()
{
   g4op-kls G4OpBoundaryProcess
}

g4op-kls()
{
    g4-
    local kls=${1:-G4OpBoundaryProcess} 
    local base=$(g4-dir)/source/processes/optical
    vi $base/src/$kls.cc $base/include/$kls.hh 
}

