Chroma Reemission
====================


`chroma/chroma/cuda/photon.h`::

    335     // absorption 
    336     //   #. advance .time and .position to absorption point
    337     //   #. if BULK_REEMIT(CONTINUE) change .direction .polarization .wavelength
    338     //   #. if BULK_ABSORB(BREAK)  .last_hit_triangle -1  
    339     //
    340     //  huh, branch BULK_REEMIT(CONTINUE) does not set .last_hit_triangle -1 ?
    341     //
    342     if (absorption_distance <= scattering_distance) {
    343         if (absorption_distance <= s.distance_to_boundary)
    344         {
    345             p.time += absorption_distance/(SPEED_OF_LIGHT/s.refractive_index1);
    346             p.position += absorption_distance*p.direction;
    347 
    348             float uniform_sample_reemit = curand_uniform(&rng);
    349             if (uniform_sample_reemit < s.reemission_prob)
    350             {
    351                 p.wavelength = sample_cdf(&rng, s.material1->n,
    352                                           s.material1->wavelength0,
    353                                           s.material1->step,
    354                                           s.material1->reemission_cdf);
                        //
                        // generating a wavelength according to 
                        // the source distribution used in construction 
                        // of the reemission_cdf
                        //
    
    355                 p.direction = uniform_sphere(&rng);
    356                 p.polarization = cross(uniform_sphere(&rng), p.direction);
    357                 p.polarization /= norm(p.polarization);
    358                 p.history |= BULK_REEMIT;
    359                 return CONTINUE;
    360             } // photon is reemitted isotropically
    361             else
    362             {
    363                 p.last_hit_triangle = -1;
    364                 p.history |= BULK_ABSORB;
    365                 return BREAK;
    366             } // photon is absorbed in material1
    367         }
    368     }


See env/geant4/geometry/collada/g4daeview/sample_cdf.py 
for review of `sample_cdf` using inverse CDF sampling to generate distributions. 

`chroma/chroma/cuda/random.h`::

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

            //
            // u, uniform random number between 0 and 1
            //
     38 
     39     int lower = 0;
     40     int upper = ncdf - 1;
     41 
     42     while(lower < upper-1)
     43     {
     44         int half = (lower + upper) / 2;
     45 
     46         if (u < cdf_y[half])
     47             upper = half;
     48         else
     49             lower = half;
     50     }

            //
            // find bin of cdf_y that straddles u   
            //
            //     * cdf_y needs to range from 0:1 
            //       (ie a cumulative distribution normalized to 1 at RHS)
            //

     51 
     52     float delta_cdf_y = cdf_y[upper] - cdf_y[lower];
     53 

            //  translate bin int into domain float 

     54     return x0 + delta*lower + delta*(u-cdf_y[lower])/delta_cdf_y;
     55 }




