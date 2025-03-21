#pragma once

/*
propagate_to_boundary absorb/scatter/sail ? 
=============================================

see /usr/local/env/chroma_env/src/chroma/chroma/cuda/photon.h

* absorb 

  #. advance .time and .position to absorption point
  #. if BULK_REEMIT(CONTINUE) change .direction .polarization .wavelength
  #. if BULK_ABSORB(BREAK)  .last_hit_triangle -1  

* scatter

  #. advance .time and .position to scattering point
  #. RAYLEIGH_SCATTER(CONTINUE)  .direction .polarization twiddled 

* sail

  #. advance .position .time to boundary 
  #. sail to boundary(PASS)  

Inputs:

* p.time
* p.position
* p.direction

* s.distance_to_boundary
* s.material1.x  refractive_index
* s.material1.y  absorption_length
* s.material1.z  scattering_length
* s.material1.w  reemission_prob

Outputs:

* p.time
* p.position
* p.direction
* p.wavelength
* p.polarization
* p.flags.i.x    (boundary)
* p.flags.i.w    (history)

Returns:

* BREAK(BULK_ABSORB)
* CONTINUE(BULK_REEMIT)
* CONTINUE(RAYLEIGH_SCATTER)
* PASS("SAIL")


*/

__device__ int propagate_to_boundary( Photon& p, State& s, curandState &rng)
{
    float speed = SPEED_OF_LIGHT/s.material1.x ;    // .x:refractive_index
    float absorption_distance = -s.material1.y*logf(curand_uniform(&rng));   // .y:absorption_length
    float scattering_distance = -s.material1.z*logf(curand_uniform(&rng));   // .z:scattering_length

    if (absorption_distance <= scattering_distance) 
    {
        if (absorption_distance <= s.distance_to_boundary) 
        {
            p.time += absorption_distance/speed ;   
            p.position += absorption_distance*p.direction;

            float uniform_sample_reemit = curand_uniform(&rng);
            if (uniform_sample_reemit < s.material1.w)                       // .w:reemission_prob
            {
                // no materialIndex input to reemission_lookup as both scintillators share same CDF 
                // non-scintillators have zero reemission_prob
                p.wavelength = reemission_lookup(curand_uniform(&rng));
                p.direction = uniform_sphere(&rng);
                p.polarization = normalize(cross(uniform_sphere(&rng), p.direction));
                p.flags.i.x = 0 ;   // no-boundary-yet for new direction
                //p.flags.i.w |= BULK_REEMIT;
                s.flag = BULK_REEMIT ;
                return CONTINUE;
            }                           
            else 
            {
                //p.flags.i.w |= BULK_ABSORB;
                s.flag = BULK_ABSORB ;
                return BREAK;
            }                         
        }
        //  otherwise sail to boundary  
    }
    else 
    {
        if (scattering_distance <= s.distance_to_boundary) 
        {
            p.time += scattering_distance/speed ; 
            p.position += scattering_distance*p.direction;

            rayleigh_scatter(p, rng);

            //p.flags.i.w |= RAYLEIGH_SCATTER;
            s.flag = BULK_SCATTER;
            p.flags.i.x = 0 ;  // no-boundary-yet for new direction

            return CONTINUE;
        } 
        //  otherwise sail to boundary  
         
    }     // if scattering_distance < absorption_distance


    p.position += s.distance_to_boundary*p.direction;
    p.time += s.distance_to_boundary/speed ;   // .x:refractive_index

    return PASS;

} // propagate_to_boundary


//
//  fresnel reflect/transmit conventional directions
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                     s1
//                   +----+          
//                    \   .   /      ^
//               c1   i\  .  / r    /|\
//                      \ . /        |                      
//         material1     \./         | n
//         ---------------+----------+----------
//         material2      .\
//                        . \
//                   c2   .  \ t
//                        .   \
//                        +----+
//                          s2
//   i, incident photons 
//      pointing down to interface (from material1 towards material2)
//
//   n, surface normal (s.surface_normal)
//      pointing up from interface (from material2 back into material1)
//      Orientation is arranged by flipping geometric normal 
//      based on photon direction.
//
//   t, transmitted photons
//      from interface into material2
//
//   r, reflected photons
//      from interface back into material1
//
//   c1, costheta_1 
//      cosine of incident angle,  c1 = dot(-i, n) = - dot(i, n)
//      arranged to be positive via normal flipping 
//      and corresponding flip of which material is labelled 1 and 2 
//     
//
//  polarisation
//  ~~~~~~~~~~~~~~~
//                    
//   S polarized : E field perpendicular to plane of incidence
//   P polarized : E field within plane of incidence 
//
//
// normal incidence photons
// ~~~~~~~~~~~~~~~~~~~~~~~~~~ 
// 
// * no unique plane of incidence, 
// * artifically setting incident_plane_normal to initial p.polarisation yields normal_coefficient = 1.0f 
//   so always treated as S polarized 
//   
//
//   initial momentum dir
//            -s.surface_normal 
//
//   final momentum dir (c1 = 1.f)
//            -s.surface_normal + 2.0f*c1*s.surface_normal  = -p.direction 
//                                                    
//
//  minimise use of trancendental functions 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  Obtain c2c2 from Snells Law without lots of expensive function calls.
//  
//        n1 s1 = n2 s2
//
//           s2 = eta * s1       eta = n1/n2
//
//
//   
//         c2c2 = 1 - s2s2 
//              = 1 - eta eta s1 s1  
//              = 1 - eta eta (1 - c1c1) 
//
//
//         c2c2 - 1 = (c1c1 - 1) eta eta
//
//        
//
//  TIR
//  ~~~~
//
//  Total internal reflection, occurs when c2c2 < 0.f  (c2 imaginary)
//
//  Handled by: 
//
//  * artificially setting c2 = 0.f 
//  * results in reflection_coefficient = 1.0f so will always reflect for both S and P cases
//
//
//

/*
propagate_at_boundary
======================

See g4op- for comparison of Geant4/Chroma/OptiX-refract

Inputs:

* p.direction
* p.polarization

* s.material1.x    : refractive index 
* s.material2.x    : refractive index
* s.surface_normal 
* s.cos_theta       (for noting normal incidence)

Outputs:

* p.direction
* p.polarization
* p.flags.i.x     (boundary) 
* p.flags.i.w     (history)

Tacitly returns CONTINUE

*/




__device__ void propagate_at_boundary_geant4_style( Photon& p, State& s, curandState &rng)
{
    // see g4op-/G4OpBoundaryProcess.cc annotations to follow this

    const float n1 = s.material1.x ;
    const float n2 = s.material2.x ;   
    const float eta = n1/n2 ; 

    const float c1 = -dot(p.direction, s.surface_normal ); // c1 arranged to be +ve   
    const float eta_c1 = eta * c1 ; 

    const float c2c2 = 1.f - eta*eta*(1.f - c1 * c1 ) ;   // Snells law 
     
    bool tir = c2c2 < 0.f ; 
    const float EdotN = dot(p.polarization , s.surface_normal ) ;  // used for TIR polarization

    const float c2 = tir ? 0.f : sqrtf(c2c2) ;   // c2 chosen +ve, set to 0.f for TIR => reflection_coefficient = 1.0f : so will always reflect

    const float n1c1 = n1*c1 ; 
    const float n2c2 = n2*c2 ; 
    const float n2c1 = n2*c1 ; 
    const float n1c2 = n1*c2 ; 

    const float3 A_trans = fabs(c1) > 0.999999f ? p.polarization : normalize(cross(p.direction, s.surface_normal)) ;
    
    // decompose p.polarization onto incident orthogonal basis

    const float E1_perp = dot(p.polarization, A_trans);   // fraction of E vector perpendicular to plane of incidence, ie S polarization
    const float3 E1pp = E1_perp * A_trans ;               // S-pol transverse component   
    const float3 E1pl = p.polarization - E1pp ;           // P-pol parallel component 
    const float E1_parl = length(E1pl) ;
  
    // G4OpBoundaryProcess at normal incidence, mentions Jackson and uses 
    //      A_trans  = OldPolarization; E1_perp = 0. E1_parl = 1. 
    // but that seems inconsistent with the above dot product, above is swapped cf that

    const float E2_perp_t = 2.f*n1c1*E1_perp/(n1c1+n2c2);  // Fresnel S-pol transmittance
    const float E2_parl_t = 2.f*n1c1*E1_parl/(n2c1+n1c2);  // Fresnel P-pol transmittance

    const float E2_perp_r = E2_perp_t - E1_perp;           // Fresnel S-pol reflectance
    const float E2_parl_r = (n2*E2_parl_t/n1) - E1_parl ;  // Fresnel P-pol reflectance

    const float2 E2_t = make_float2( E2_perp_t, E2_parl_t ) ;
    const float2 E2_r = make_float2( E2_perp_r, E2_parl_r ) ;

    const float  E2_total_t = dot(E2_t,E2_t) ; 

    const float2 T = normalize(E2_t) ; 
    const float2 R = normalize(E2_r) ; 

    const float TransCoeff =  tir ? 0.0f : n2c2*E2_total_t/n1c1 ; 
    //  above 0.0f was until 2016/3/4 incorrectly a 1.0f 
    //  resulting in TIR yielding BT where BR is expected

    bool reflect = curand_uniform(&rng) > TransCoeff  ;

    p.direction = reflect 
                    ? 
                       p.direction + 2.0f*c1*s.surface_normal 
                    : 
                       eta*p.direction + (eta_c1 - c2)*s.surface_normal
                    ;   

    const float3 A_paral = normalize(cross(p.direction, A_trans));

    p.polarization = reflect ?
                                ( tir ? 
                                        -p.polarization + 2.f*EdotN*s.surface_normal 
                                      :
                                        R.x*A_trans + R.y*A_paral 
                                )
                             :
                                T.x*A_trans + T.y*A_paral 
                             ;


    s.flag = reflect     ? BOUNDARY_REFLECT : BOUNDARY_TRANSMIT ; 

    p.flags.i.x = 0 ;  // no-boundary-yet for new direction
}



__device__ void propagate_at_boundary( Photon& p, State& s, curandState &rng)
{
    float eta = s.material1.x/s.material2.x ;    // eta = n1/n2   x:refractive_index  PRE-FLIPPED

    float3 incident_plane_normal = fabs(s.cos_theta) < 1e-6f ? p.polarization : normalize(cross(p.direction, s.surface_normal)) ;

    float normal_coefficient = dot(p.polarization, incident_plane_normal);  // fraction of E vector perpendicular to plane of incidence, ie S polarization

    const float c1 = -dot(p.direction, s.surface_normal ); // c1 arranged to be +ve   

    const float c2c2 = 1.f - eta*eta*(1.f - c1 * c1 ) ; 

    bool tir = c2c2 < 0.f ; 

    const float c2 = tir ? 0.f : sqrtf(c2c2) ;   // c2 chosen +ve, set to 0.f for TIR => reflection_coefficient = 1.0f : so will always reflect

    const float eta_c1 = eta * c1 ; 

    const float eta_c2 = eta * c2 ;    

    bool s_polarized = curand_uniform(&rng) < normal_coefficient*normal_coefficient ;

    const float reflection_coefficient = s_polarized 
                      ? 
                         (eta_c1 - c2)/(eta_c1 + c2 )  
                      :
                         (c1 - eta_c2)/(c1 + eta_c2)  
                      ; 

    bool reflect = curand_uniform(&rng) < reflection_coefficient*reflection_coefficient ;

    // need to find new direction first as polarization depends on it for case P

    p.direction = reflect 
                    ? 
                       p.direction + 2.0f*c1*s.surface_normal 
                    : 
                       eta*p.direction + (eta_c1 - c2)*s.surface_normal
                    ;   

    p.polarization = s_polarized 
                       ? 
                          incident_plane_normal
                       :
                          normalize(cross(incident_plane_normal, p.direction))
                       ;
    

    //p.flags.i.w |= reflect     ? BOUNDARY_REFLECT : BOUNDARY_TRANSMIT ;
    //p.flags.i.w |= s_polarized ? BOUNDARY_SPOL    : BOUNDARY_PPOL ;
    //p.flags.i.w |= tir         ? BOUNDARY_TIR     : BOUNDARY_TIR_NOT ; 

    s.flag = reflect     ? BOUNDARY_REFLECT : BOUNDARY_TRANSMIT ; 

    p.flags.i.x = 0 ;  // no-boundary-yet for new direction
}



/*
propagate_at_specular_reflector / propagate_at_diffuse_reflector
===================================================================

Inputs:

* p.direction
* p.polarization

* s.surface_normal
* s.cos_theta

Outputs:

* p.direction
* p.polarization
* p.flags.i.x
* p.flags.i.w

Returns:

CONTINUE


*/

__device__ void propagate_at_specular_reflector(Photon &p, State &s, curandState &rng)
{
    const float c1 = -dot(p.direction, s.surface_normal );     // c1 arranged to be +ve   

    // TODO: make change to c1 for normal incidence detection
 
    float3 incident_plane_normal = fabs(s.cos_theta) < 1e-6f ? p.polarization : normalize(cross(p.direction, s.surface_normal)) ;

    float normal_coefficient = dot(p.polarization, incident_plane_normal);  // fraction of E vector perpendicular to plane of incidence, ie S polarization

    p.direction += 2.0f*c1*s.surface_normal  ;  

    bool s_polarized = curand_uniform(&rng) < normal_coefficient*normal_coefficient ;

    p.polarization = s_polarized 
                       ? 
                          incident_plane_normal
                       :
                          normalize(cross(incident_plane_normal, p.direction))
                       ;

    //p.flags.i.w |= REFLECT_SPECULAR;
 
    p.flags.i.x = 0 ;  // no-boundary-yet for new direction
} 

__device__ void propagate_at_diffuse_reflector(Photon &p, State &s, curandState &rng)
{
    float ndotv;
    do {
	    p.direction = uniform_sphere(&rng);
	    ndotv = dot(p.direction, s.surface_normal);
	    if (ndotv < 0.0f) 
        {
	        p.direction = -p.direction;
	        ndotv = -ndotv;
	    }
    } while (! (curand_uniform(&rng) < ndotv) );

    p.polarization = normalize( cross(uniform_sphere(&rng), p.direction));
    //p.flags.i.w |= REFLECT_DIFFUSE;
    p.flags.i.x = 0 ;  // no-boundary-yet for new direction
}                       


/*
propagate_at_surface
======================

Inputs:

* s.surface.x detect
* s.surface.y absorb              (1.f - reflectivity ) ?
* s.surface.z reflect_specular
* s.surface.w reflect_diffuse

Returns:

* BREAK(SURFACE_ABSORB) 
* BREAK(SURFACE_DETECT) 
* CONTINUE(SURFACE_DREFLECT) 
* CONTINUE(SURFACE_SREFLECT) 


TODO
-----

* arrange values to do equivalent to G4 ?

   absorb + detect + reflect_diffuse + reflect_specular  = 1   ??

* How to handle special casing of some surfaces...

  * SPECULARLOBE...


*/

__device__ int
propagate_at_surface(Photon &p, State &s, curandState &rng)
{

    float u = curand_uniform(&rng);

    if( u < s.surface.y )   // absorb   
    {
        s.flag = SURFACE_ABSORB ;
        return BREAK ;
    }
    else if ( u < s.surface.y + s.surface.x )  // absorb + detect
    {
        s.flag = SURFACE_DETECT ;
        return BREAK ;
    } 
    else if (u  < s.surface.y + s.surface.x + s.surface.w )  // absorb + detect + reflect_diffuse 
    {
        s.flag = SURFACE_DREFLECT ;
        propagate_at_diffuse_reflector(p, s, rng);
        return CONTINUE;
    }
    else
    {
        s.flag = SURFACE_SREFLECT ;
        propagate_at_specular_reflector(p, s, rng );
        return CONTINUE;
    }
}


