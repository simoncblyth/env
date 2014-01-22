Chroma CUDA Photon handling
==============================


geometry from CUDA photon propagation, in `photon.h`::

    584 __device__ int   
    585 propagate_at_surface(Photon &p, State &s, curandState &rng, Geometry *geometry,
    586                      bool use_weights=false)
    587 {
    588     Surface *surface = geometry->surfaces[s.surface_index];
    589 
    590     if (surface->model == SURFACE_COMPLEX)
    591         return propagate_complex(p, s, rng, surface, use_weights);
    592     else if (surface->model == SURFACE_WLS)
    593         return propagate_at_wls(p, s, rng, surface, use_weights);
    594     else {
    595         // use default surface model: do a combination of specular and
    596         // diffuse reflection, detection, and absorption based on relative
    597         // probabilties
    598 
    599         // since the surface properties are interpolated linearly, we are
    600         // guaranteed that they still sum to 1.0.
    601         float detect = interp_property(surface, p.wavelength, surface->detect);
    602         float absorb = interp_property(surface, p.wavelength, surface->absorb);
    603         float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    604         float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    605 
    606         float uniform_sample = curand_uniform(&rng);



::

    simon:cuda blyth$ grep __shared__ *.*
    bvh.cu:    __shared__ unsigned long long min_area[128];
    bvh.cu:    __shared__ unsigned long long adjacent_area;
    daq.cu:    __shared__ int photon_id;
    daq.cu:    __shared__ int triangle_id;
    daq.cu:    __shared__ int solid_id;
    daq.cu:    __shared__ int channel_index;
    daq.cu:    __shared__ unsigned int history;
    daq.cu:    __shared__ float photon_time;
    daq.cu:    __shared__ float weight;
    mesh.h:    __shared__ Geometry sg;
    pdf.cu:    __shared__ float distance_table[1000];
    pdf.cu:    __shared__ unsigned int *work_queue;
    pdf.cu:    __shared__ int queue_items;
    pdf.cu:    __shared__ int channel_id;
    pdf.cu:    __shared__ float channel_event_time;
    pdf.cu:    __shared__ int distance_table_len;
    pdf.cu:    __shared__ int offset;
    propagate.cu:    __shared__ unsigned int counter;
    propagate.cu:    __shared__ Geometry sg;
    render.cu:    __shared__ Geometry sg;
    simon:cuda blyth$ 


::

    simon:cuda blyth$ grep sync *.*      
    bvh.cu:    __syncthreads();
    bvh.cu:    __syncthreads();
    bvh.cu:    __syncthreads();
    daq.cu:    __syncthreads();
    mesh.h:    __syncthreads();
    pdf.cu:    __syncthreads();
    pdf.cu:    __syncthreads();
    pdf.cu:    __syncthreads();
    pdf.cu:    __syncthreads();
    propagate.cu:    __syncthreads();
    propagate.cu:    __syncthreads();
    propagate.cu:    __syncthreads();
    render.cu:    __syncthreads();
    simon:cuda blyth$ 



