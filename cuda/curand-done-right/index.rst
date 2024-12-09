curand-done-right
===================

::

   cd
   git clone https://github.com/kshitijl/curand-done-right.git

   cp ~/curand-done-right/src/curand-done-right/curanddr.hxx ~/env/cuda/curand-done-right/
   cp ~/curand-done-right/examples/basic-pi.cu ~/env/cuda/curand-done-right/
   cp ~/curand-done-right/Makefile ~/env/cuda/curand-done-right/



::

    P[blyth@localhost curand-done-right]$ ./basic-pi.sh 
    //estimate_pi  index 0  xx  0.178931 yy  0.075331 
    //estimate_pi  index 1  xx  0.072204 yy  0.117255 
    //estimate_pi  index 2  xx  0.312774 yy  0.602896 
    //estimate_pi  index 3  xx  0.081673 yy  0.547574 
    //estimate_pi  index 4  xx  0.944169 yy  0.364360 
    //estimate_pi  index 5  xx  0.278512 yy  0.287804 
    //estimate_pi  index 6  xx  0.111264 yy  0.254863 
    //estimate_pi  index 7  xx  0.838473 yy  0.444990 
    //estimate_pi  index 8  xx  0.947367 yy  0.443467 
    //estimate_pi  index 9  xx  0.853467 yy  0.653512 
    785338 448
    3.141352


    P[blyth@localhost curand-done-right]$ ./basic-pi.sh 
    //estimate_pi  index 0  xx  0.178931 yy  0.075331 zz  0.988173 ww  0.634883 
    //estimate_pi  index 1  xx  0.072204 yy  0.117255 zz  0.283267 ww  0.039935 
    //estimate_pi  index 2  xx  0.312774 yy  0.602896 zz  0.603033 ww  0.364543 
    //estimate_pi  index 3  xx  0.081673 yy  0.547574 zz  0.456981 ww  0.642444 
    //estimate_pi  index 4  xx  0.944169 yy  0.364360 zz  0.667021 ww  0.933453 
    //estimate_pi  index 5  xx  0.278512 yy  0.287804 zz  0.692024 ww  0.071551 
    //estimate_pi  index 6  xx  0.111264 yy  0.254863 zz  0.414897 ww  0.567098 
    //estimate_pi  index 7  xx  0.838473 yy  0.444990 zz  0.229636 ww  0.167966 
    //estimate_pi  index 8  xx  0.947367 yy  0.443467 zz  0.803773 ww  0.519327 
    //estimate_pi  index 9  xx  0.853467 yy  0.653512 zz  0.157435 ww  0.020901 
    785338 448
    3.141352
    P[blyth@localhost curand-done-right]$ 


Getting 2 or 4 doesnt change the first 2::

     36     //curanddr::vector_t<2,float> randoms = curanddr::uniforms<2>(uint4{0,0,0,seed}, index); 
     37     curanddr::vector_t<4,float> randoms = curanddr::uniforms<4>(uint4{0,0,0,seed}, index);





Second call with same index gives same values::

    37     curanddr::vector_t<4,float> randoms = curanddr::uniforms<4>(uint4{0,0,0,seed}, index);
    38     curanddr::vector_t<4,float> randoms1 = curanddr::uniforms<4>(uint4{0,0,0,seed}, index);

    P[blyth@localhost curand-done-right]$ ./basic-pi.sh 
    //estimate_pi  index 0  xx  0.178931 yy  0.075331 zz  0.988173 ww  0.634883 
    //estimate_pi  index 1  xx  0.072204 yy  0.117255 zz  0.283267 ww  0.039935 
    //estimate_pi  index 2  xx  0.312774 yy  0.602896 zz  0.603033 ww  0.364543 
    //estimate_pi  index 3  xx  0.081673 yy  0.547574 zz  0.456981 ww  0.642444 
    //estimate_pi  index 4  xx  0.944169 yy  0.364360 zz  0.667021 ww  0.933453 
    //estimate_pi  index 5  xx  0.278512 yy  0.287804 zz  0.692024 ww  0.071551 
    //estimate_pi  index 6  xx  0.111264 yy  0.254863 zz  0.414897 ww  0.567098 
    //estimate_pi  index 7  xx  0.838473 yy  0.444990 zz  0.229636 ww  0.167966 
    //estimate_pi  index 8  xx  0.947367 yy  0.443467 zz  0.803773 ww  0.519327 
    //estimate_pi  index 9  xx  0.853467 yy  0.653512 zz  0.157435 ww  0.020901 
    //estimate_pi  index 0  xx1  0.178931 yy1  0.075331 zz1  0.988173 ww1  0.634883 
    //estimate_pi  index 1  xx1  0.072204 yy1  0.117255 zz1  0.283267 ww1  0.039935 
    //estimate_pi  index 2  xx1  0.312774 yy1  0.602896 zz1  0.603033 ww1  0.364543 
    //estimate_pi  index 3  xx1  0.081673 yy1  0.547574 zz1  0.456981 ww1  0.642444 
    //estimate_pi  index 4  xx1  0.944169 yy1  0.364360 zz1  0.667021 ww1  0.933453 
    //estimate_pi  index 5  xx1  0.278512 yy1  0.287804 zz1  0.692024 ww1  0.071551 
    //estimate_pi  index 6  xx1  0.111264 yy1  0.254863 zz1  0.414897 ww1  0.567098 
    //estimate_pi  index 7  xx1  0.838473 yy1  0.444990 zz1  0.229636 ww1  0.167966 
    //estimate_pi  index 8  xx1  0.947367 yy1  0.443467 zz1  0.803773 ww1  0.519327 
    //estimate_pi  index 9  xx1  0.853467 yy1  0.653512 zz1  0.157435 ww1  0.020901 
    785338 448
    3.141352
    P[blyth@localhost curand-done-right]$


Second call with index+1 repeats values from other thread::

    37     curanddr::vector_t<4,float> randoms = curanddr::uniforms<4>(uint4{0,0,0,seed}, index);
    38     curanddr::vector_t<4,float> randoms1 = curanddr::uniforms<4>(uint4{0,0,0,seed}, index+1);

    P[blyth@localhost curand-done-right]$ ./basic-pi.sh 
    //estimate_pi  index 0  xx  0.178931 yy  0.075331 zz  0.988173 ww  0.634883 
    //estimate_pi  index 1  xx  0.072204 yy  0.117255 zz  0.283267 ww  0.039935 
    //estimate_pi  index 2  xx  0.312774 yy  0.602896 zz  0.603033 ww  0.364543 
    //estimate_pi  index 3  xx  0.081673 yy  0.547574 zz  0.456981 ww  0.642444 
    //estimate_pi  index 4  xx  0.944169 yy  0.364360 zz  0.667021 ww  0.933453 
    //estimate_pi  index 5  xx  0.278512 yy  0.287804 zz  0.692024 ww  0.071551 
    //estimate_pi  index 6  xx  0.111264 yy  0.254863 zz  0.414897 ww  0.567098 
    //estimate_pi  index 7  xx  0.838473 yy  0.444990 zz  0.229636 ww  0.167966 
    //estimate_pi  index 8  xx  0.947367 yy  0.443467 zz  0.803773 ww  0.519327 
    //estimate_pi  index 9  xx  0.853467 yy  0.653512 zz  0.157435 ww  0.020901 
    //estimate_pi  index 0  xx1  0.072204 yy1  0.117255 zz1  0.283267 ww1  0.039935 
    //estimate_pi  index 1  xx1  0.312774 yy1  0.602896 zz1  0.603033 ww1  0.364543 
    //estimate_pi  index 2  xx1  0.081673 yy1  0.547574 zz1  0.456981 ww1  0.642444 
    //estimate_pi  index 3  xx1  0.944169 yy1  0.364360 zz1  0.667021 ww1  0.933453 
    //estimate_pi  index 4  xx1  0.278512 yy1  0.287804 zz1  0.692024 ww1  0.071551 
    //estimate_pi  index 5  xx1  0.111264 yy1  0.254863 zz1  0.414897 ww1  0.567098 
    //estimate_pi  index 6  xx1  0.838473 yy1  0.444990 zz1  0.229636 ww1  0.167966 
    //estimate_pi  index 7  xx1  0.947367 yy1  0.443467 zz1  0.803773 ww1  0.519327 
    //estimate_pi  index 8  xx1  0.853467 yy1  0.653512 zz1  0.157435 ww1  0.020901 
    //estimate_pi  index 9  xx1  0.030141 yy1  0.643481 zz1  0.333829 ww1  0.758343 
    785338 448
    3.141352
    P[blyth@localhost curand-done-right]$ 



TO CHECK: DOES INCREMENTING ANY OF THE FIVE BEHAVE THE SAME ?
------------------------------------------------------------------






check what curanddr.hxx does
-----------------------------

Usage::

    37     curanddr::vector_t<4,float> randoms = curanddr::uniforms<4>(uint4{0,0,0,seed}, index);


/home/blyth/env/cuda/curand-done-right/curanddr.hxx::

    076   template<int Arity>
     77   __device__ vector_t<Arity> uniforms(uint4 counter, uint key) {
     78     enum { n_blocks = (Arity + 4 - 1)/4 };
     79 
     80     float scratch[n_blocks * 4];
     81 
     82     iterate<n_blocks>([&](uint index) {
     83         uint2 local_key{key, index};
     84         uint4 result = curand_Philox4x32_10(counter, local_key);
     85 
     86         uint ii = index*4;
     87         scratch[ii]   = _curand_uniform(result.x);
     88         scratch[ii+1] = _curand_uniform(result.y);
     89         scratch[ii+2] = _curand_uniform(result.z);
     90         scratch[ii+3] = _curand_uniform(result.w);
     91       });
     92 
     93     vector_t<Arity> answer;
     94 
     95     iterate<Arity>([&](uint index) {
     96         answer.values[index] = scratch[index];
     97       });
     98 
     99     return answer;
    100   }


::

    In [10]: (np.arange(1,20) + 4 - 1)//4
    Out[10]: array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5])



curand_kernel.h::

     140 struct curandStateXORWOW {
     141     unsigned int d, v[5];
     142     int boxmuller_flag;
     143     int boxmuller_flag_double;
     144     float boxmuller_extra;
     145     double boxmuller_extra_double;
     146 };



/usr/local/cuda/include/curand_philox4x32_x.h::

    092 struct curandStatePhilox4_32_10 {

     93    uint4 ctr;                      // 16 
     94    uint4 output;                   // 16   < also used to fake 1-by-1 when actually 4-by-4   
     95    uint2 key;                      //  8
     96    unsigned int STATE;             //  4  < 0,1,2,3,(4): used to fake 1-by-1 when actually 4-by-4 

     97    int boxmuller_flag;              // 4  < used by curand_normal faking 1-by-1 when 2-by-2
     98    int boxmuller_flag_double;       // 4  
     99    float boxmuller_extra;           // 4 
    100    double boxmuller_extra_double;   // 8 
    101 };                                // ------
    102                                       64    total bytes

                                              24    (16+8 uint4 ctr, uint2 key)  counters  

                                              20    used to fake 1-by-1 when actually 4-by-4 for curand_uniform
                                              20    used to fake 1-by-1 when actually 2-by-2 for curand_normal



::

    342 QUALIFIERS float curand_normal(curandStatePhilox4_32_10_t *state)
    343 {
    344     if(state->boxmuller_flag != EXTRA_FLAG_NORMAL) {
    345         unsigned int x, y;
    346         x = curand(state);
    347         y = curand(state);
    348         float2 v = _curand_box_muller(x, y);
    349         state->boxmuller_extra = v.y;
    350         state->boxmuller_flag = EXTRA_FLAG_NORMAL;
    351         return v.x;
    352     }
    353     state->boxmuller_flag = 0;
    354     return state->boxmuller_extra;
    355 }

    /// AHHA : the boxmuller_extra and boxmuller_flag is again
    ///        making something that naturally gives 2 normally 
    ///        distrib values look like it gives 1 without 
    ///        ... and costs 20 bytes for this "fib"

    402 QUALIFIERS float2 curand_normal2(curandStateXORWOW_t *state)
    403 {
    404     return curand_box_muller(state);
    405 }

    151 template <typename R>
    152 QUALIFIERS float2 curand_box_muller(R *state)
    153 {
    154     float2 result;
    155     unsigned int x = curand(state);
    156     unsigned int y = curand(state);
    157     result = _curand_box_muller(x, y);
    158     return result;
    159 }

     69 QUALIFIERS float2 _curand_box_muller(unsigned int x, unsigned int y)
     70 {
     71     float2 result;
     72     float u = x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2);
     73     float v = y * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI/2);
     74 #if __CUDA_ARCH__ > 0
     75     float s = sqrtf(-2.0f * logf(u));
     76     __sincosf(v, &result.x, &result.y);
     77 #else
     78     float s = sqrtf(-2.0f * logf(u));
     79     result.x = sinf(v);
     80     result.y = cosf(v);
     81 #endif
     82     result.x *= s;
     83     result.y *= s;
     84     return result;
     85 }







::

    1012 QUALIFIERS void curand_init(unsigned long long seed,
    1013                                  unsigned long long subsequence,
    1014                                  unsigned long long offset,
    1015                                  curandStatePhilox4_32_10_t *state)
    1016 {
    1017     state->ctr = make_uint4(0, 0, 0, 0);
    1018     state->key.x = (unsigned int)seed;
    1019     state->key.y = (unsigned int)(seed>>32);
    1020     state->STATE = 0;
    1021     state->boxmuller_flag = 0;
    1022     state->boxmuller_flag_double = 0;
    1023     state->boxmuller_extra = 0.f;
    1024     state->boxmuller_extra_double = 0.;
    1025     skipahead_sequence(subsequence, state);
    1026     skipahead(offset, state);
    1027 }

    0961 QUALIFIERS void skipahead(unsigned long long n, curandStatePhilox4_32_10_t *state)
     962 {
     963     state->STATE += (n & 3);
     964     n /= 4;
     965     if( state->STATE > 3 ){
     966         n += 1;
     967         state->STATE -= 4;
     968     }
     969     Philox_State_Incr(state, n);
     970     state->output = curand_Philox4x32_10(state->ctr,state->key);
     971 }
    /// looks to be fiddling to enable generator that returns sets of four random uint 
    /// to look like it can be skipped ahead not in steps of four by item by item 
    /// [n & 3 is 0,1,2,3 only, whats range if STATE? 0,1,2,3 only ? ]

    106 QUALIFIERS void Philox_State_Incr(curandStatePhilox4_32_10_t* s, unsigned long long n)
    107 {
    108    unsigned int nlo = (unsigned int)(n);
    109    unsigned int nhi = (unsigned int)(n>>32);
    110 
    111    s->ctr.x += nlo;
    112    if( s->ctr.x < nlo )
    113       nhi++;
    114 
    115    s->ctr.y += nhi;
    116    if(nhi <= s->ctr.y)
    117       return;
    118    if(++s->ctr.z) return;
    119    ++s->ctr.w;
    120 }


    170 QUALIFIERS uint4 curand_Philox4x32_10( uint4 c, uint2 k)
    171 {
    172    c = _philox4x32round(c, k);                           // 1 
    173    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    174    c = _philox4x32round(c, k);                           // 2
    175    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    176    c = _philox4x32round(c, k);                           // 3 
    177    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    178    c = _philox4x32round(c, k);                           // 4 
    179    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    180    c = _philox4x32round(c, k);                           // 5 
    181    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    182    c = _philox4x32round(c, k);                           // 6 
    183    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    184    c = _philox4x32round(c, k);                           // 7 
    185    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    186    c = _philox4x32round(c, k);                           // 8 
    187    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    188    c = _philox4x32round(c, k);                           // 9 
    189    k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    190    return _philox4x32round(c, k);                        // 10
    191 }

* Notice arg structs used as workspace




/usr/local/cuda/include/curand_kernel.h::

    255 QUALIFIERS float curand_uniform(curandStatePhilox4_32_10_t *state)
    256 {
    257    return _curand_uniform(curand(state));
    258 }

/usr/local/cuda/include/curand_uniform.h::

     69 QUALIFIERS float _curand_uniform(unsigned int x)
     70 {
     71     return x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
     72 }





     878 QUALIFIERS unsigned int curand(curandStatePhilox4_32_10_t *state)
     879 {
     880     // Maintain the invariant: output[STATE] is always "good" and
     881     //  is the next value to be returned by curand.
     882     unsigned int ret;
     883     switch(state->STATE++){
     884     default:
     885         ret = state->output.x;
     886         break;
     887     case 1:
     888         ret = state->output.y;
     889         break;
     890     case 2:
     891         ret = state->output.z;
     892         break;
     893     case 3:
     894         ret = state->output.w;
     895         break;
     896     }
     897     if(state->STATE == 4){
     898         Philox_State_Incr(state);
     899         state->output = curand_Philox4x32_10(state->ctr,state->key);
     900         state->STATE = 0;
     901     }
     902     return ret;
     903 }


/home/blyth/env/cuda/curand-done-right/curanddr.hxx::

    076   template<int Arity>
     77   __device__ vector_t<Arity> uniforms(uint4 counter, uint key) {
     78     enum { n_blocks = (Arity + 4 - 1)/4 };
     79 
     80     float scratch[n_blocks * 4];
     81 
     82     iterate<n_blocks>([&](uint index) {
     83         uint2 local_key{key, index};
     84         uint4 result = curand_Philox4x32_10(counter, local_key);
     85 
     86         uint ii = index*4;
     87         scratch[ii]   = _curand_uniform(result.x);
     88         scratch[ii+1] = _curand_uniform(result.y);
     89         scratch[ii+2] = _curand_uniform(result.z);
     90         scratch[ii+3] = _curand_uniform(result.w);
     91       });
     92 
     93     vector_t<Arity> answer;
     94 
     95     iterate<Arity>([&](uint index) {
     96         answer.values[index] = scratch[index];
     97       });
     98 
     99     return answer;
    100   }



::

     905 /**
     906  * \brief Return tuple of 4 32-bit pseudorandoms from a Philox4_32_10 generator.
     907  *
     908  * Return 128 bits of pseudorandomness from the Philox4_32_10 generator in \p state,
     909  * increment position of generator by four.
     910  *
     911  * \param state - Pointer to state to update
     912  *
     913  * \return 128-bits of pseudorandomness as a uint4, all bits valid to use.
     914  */
     915 
     916 QUALIFIERS uint4 curand4(curandStatePhilox4_32_10_t *state)
     917 {
     918     uint4 r;
     919 
     920     uint4 tmp = state->output;
     921     Philox_State_Incr(state);
     922     state->output= curand_Philox4x32_10(state->ctr,state->key);
     923     switch(state->STATE){
     924     case 0:
     925         return tmp;
     926     case 1:
     927         r.x = tmp.y;
     928         r.y = tmp.z;
     929         r.z = tmp.w;
     930         r.w = state->output.x;
     931         break;
     932     case 2:
     933         r.x = tmp.z;
     934         r.y = tmp.w;
     935         r.z = state->output.x;
     936         r.w = state->output.y;
     937         break;
     938     case 3:
     939         r.x = tmp.w;
     940         r.y = state->output.x;
     941         r.z = state->output.y;
     942         r.w = state->output.z;
     943         break;
     944     default:
     945         // NOT possible but needed to avoid compiler warnings
     946         return tmp;
     947     }
     948     return r;
     949 }


::

    P[blyth@localhost include]$ pwd
    /usr/local/cuda/include
    P[blyth@localhost include]$ grep curand_Philox4x32_10 *.h
    curand_kernel.h:        state->output = curand_Philox4x32_10(state->ctr,state->key);
    curand_kernel.h:    state->output= curand_Philox4x32_10(state->ctr,state->key);
    curand_kernel.h:    state->output = curand_Philox4x32_10(state->ctr,state->key);
    curand_kernel.h:    state->output = curand_Philox4x32_10(state->ctr,state->key);
    curand_philox4x32_x.h:QUALIFIERS uint4 curand_Philox4x32_10( uint4 c, uint2 k)
    P[blyth@localhost include]$ 



How to make normal curand API with curandState use this ? 
------------------------------------------------------------


::

    287 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    288 {
    289     sevent* evt = params.evt ;
    290     if (launch_idx.x >= evt->num_photon) return;
    291 
    292     unsigned idx = launch_idx.x ;  // aka photon_idx
    293     unsigned genstep_idx = evt->seed[idx] ;
    294     const quad6& gs = evt->genstep[genstep_idx] ;
    295 
    296     qsim* sim = params.sim ;
    297 
    298 //#define OLD_WITHOUT_SKIPAHEAD 1
    299 #ifdef OLD_WITHOUT_SKIPAHEAD
    300     curandState rng = sim->rngstate[idx] ;
    301 #else
    302     curandState rng ;
    303     sim->rng->get_rngstate_with_skipahead( rng, sim->evt->index, idx );
    304 #endif
    305 
    306 

::

     53 inline QRNG_METHOD void qrng::get_rngstate_with_skipahead(curandStateXORWOW& rng, unsigned event_idx, unsigned photon_idx )
     54 {
     55     unsigned long long skipahead_ = skipahead_event_offset*event_idx ;
     56     rng = *(rng_states + photon_idx) ;
     57     skipahead( skipahead_, &rng );
     58 }





