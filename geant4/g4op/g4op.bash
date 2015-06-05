# === func-gen- : geant4/g4op/g4op fgp geant4/g4op/g4op.bash fgn g4op fgh geant4/g4op
g4op-src(){      echo geant4/g4op/g4op.bash ; }
g4op-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4op-src)} ; }
g4op-vi(){       vi $(g4op-source) ; }
g4op-env(){      elocal- ; }
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




vector treatment of fresnel eqns with polrization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.classe.cornell.edu/~ib38/tmp/reading/syn_rad/wigner_dist/polarized_light/DKE498_ch8.pdf
* http://iqst.ca/quantech/pubs/2013/fresnel-eoe.pdf

* http://en.wikipedia.org/wiki/Mueller_calculus
* http://en.wikipedia.org/wiki/Stokes_parameters


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
     ...
     914           sint1 = std::sqrt(1.-cost1*cost1);
     915           sint2 = sint1*Rindex1/Rindex2;     // *** Snell's Law ***
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
    ////
    ////
    ////        (P-polarization: E field within plane of incidence) 
    ////         magnetic field continuity at boundary (sign from triad convention wrt field directions)
    ////
    ////              Hi - Hr = Ht 
    ////         n1 (Ei - Er ) = n2 Et      relating to E brings in material characteristics  (6) 
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







EOU
}
g4op-dir(){ echo $(local-base)/env/geant4/g4op/geant4/g4op-g4op ; }
g4op-cd(){  cd $(g4op-dir); }
g4op-mate(){ mate $(g4op-dir) ; }
g4op-get(){
   local dir=$(dirname $(g4op-dir)) &&  mkdir -p $dir && cd $dir

}
