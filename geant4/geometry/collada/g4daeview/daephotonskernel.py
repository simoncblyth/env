#!/usr/bin/env python
"""

NEXT:

#. Current DAEVertexBuffer machinery relies on specific names, 
   "position" and "color" for appropriate VertexAttribute setup. 
   Maybe can get away from that via GLSL attribute fed into gl_Position ?

#. Oops forgot to GL_DYNAMIC_DRAW, but its working anyhow


Extending interop to Chroma propagation ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. all the photon data could conceivably be visualized or at least used by shaders

::

    __global__ void
    propagate(

          /* control arguments */
 
          int first_photon, 
          int nthreads, 
          unsigned int *input_queue,
          unsigned int *output_queue, 
          curandState *rng_states,

          /* photon data */

          float3*       positions,          ##
          float3*       directions,         ##
          float*        wavelengths,        ##
          float3*       polarizations,      ##
          float*        times,              ##
          unsigned int* histories,          ##
          int*          last_hit_triangles, ##
          float*        weights,            ##

          /* configuration */

          int max_steps, 
          int use_weights, 
          int scatter_first,
          Geometry *g
          ) 
    {




float4 coalesceing ?
~~~~~~~~~~~~~~~~~~~~~

* http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

  * regards casting and alignment

::

    float4*       positions_weights,         ##
    float4*       directions_wavelengths,    ##
    float4*       polarizations_times,       ##
    unsigned int* histories,                 ##   can this be stuffed into an int ?
    int*          last_hit_triangles,        ##


::

    struct VPhoton { 
       float4 position_weight,                 # 4*4 = 16  
       float4 direction_wavelength,            # 4*4 = 16     
       float4 polarization_time,               # 4*4 = 16  48
       int4 history_last_hit_triangle_sp_sp }  # 4*4 = 16  64      ## 2 spare 4 byte int 



cuda casting
~~~~~~~~~~~~~~

* :google:`cuda union struct float4 int4`
* http://stackoverflow.com/questions/9044139/accessing-float4-int4-elements-using-a-loop-in-cuda
* https://code.google.com/p/hydrazine/source/browse/trunk/hydrazine/cuda/include/vector_types.h?r=23

* https://svn.ece.lsu.edu/svn/gp/proj-base/boxes/render.cc

  * particle system example using lots of structs

::

    /*DEVICE_BUILTIN*/
    struct __builtin_align__(16) float4
    {
      float x, y, z, w;
    };

    /*DEVICE_BUILTIN*/
    struct __builtin_align__(16) int4
    {
      int x, y, z, w;
    };




"""
import logging
log = logging.getLogger(__name__)

from operator import mul
mul_ = lambda _:reduce(mul, _)          # product of elements 
div_ = lambda num,den:(num+den-1)//den  # integer division trick, rounding up without iffing around


from pycuda.compiler import SourceModule


kernel_source = r"""
//
// CUDA kernel checking the modification of OpenGL VBO 
//
//  #. depends on the simple quad*numquad structure of the VBO 
//     created by DAEPhotonsData.create_data 
//
#include <stdio.h>

union quad
{
   float4 f ;
   int4   i ;
   uint4  u ;
};

#define TEST qlht.i.x 

__global__ void jump(float4* vbo, int items )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x; 
    if (id >= items ) return ;

    float4 posw = vbo[id*%(numquad)s] ;    // position_weight 
    float4 dirw = vbo[id*%(numquad)s+1] ;  // direction_wavelength
    float4 polt = vbo[id*%(numquad)s+2] ;  // polarization_time
    
    union quad qflags ; 
    qflags.f = vbo[id*%(numquad)s+3] ; 

    union quad qlht ;                      // last_hit_triangle
    qlht.f = vbo[id*%(numquad)s+4] ;    

    if( id %% 100 == 0){
       printf( "id %%d \n", id );
       printf( "posw.x %%f \n", posw.x );

       printf( "qflags.f.x %%f \n", qflags.f.x );
       printf( "qflags.i.x %%d \n", qflags.i.x );
       printf( "qflags.u.x %%d \n", qflags.u.x );

       printf( "qlht.f.x %%f \n", qlht.f.x );
       printf( "qlht.i.x %%d \n", qlht.i.x );
       printf( "qlht.u.x %%d \n", qlht.u.x );

       for( int n=0 ; n < 32 ; ++n ){
          if (( TEST & ( 1 << n )) == ( 1 << n )){
             printf(" TEST %%d\n", n );
          }  
       }

    }

    // modify x of slot0 float4, position 
    // vbo[id*%(numquad)s+0] = make_float4( posw.x + 10. , posw.y , posw.z, posw.w );  
    
    // grow the direction
    // vbo[id*%(numquad)s+1] = make_float4( dirw.x*1.01 , dirw.y*1.01 , dirw.z*1.01, dirw.w ); 

    // set a constant momdir for all photons
    // vbo[id*%(numquad)s+1] = make_float4( 100. , 100. , 100. , 0. ); 
    
}
"""

class DAEPhotonsKernel(object):
    """
    NEXT: 

    #. test not-so-simple VBO structure 
    #. adopt ~/e/cuda/cuda_launch approach
    """
    def __init__(self, dphotons):

        ctx = { 'numquad':dphotons.numquad }
        self.kernel_source = kernel_source % ctx 

        module = SourceModule(self.kernel_source )
        kernel = module.get_function("jump")
        kernel.prepare("Pi")
        self.kernel = kernel

    def __call__(self, vbo_dev_ptr, workitems ):
        """
        Assuming a single launch 

        """
        block = (64,1,1)
        threads_per_block = reduce(mul, block)
        grid = ( div_(workitems,threads_per_block), 1 )
        #log.debug("grid %s block %s workitems %s " % ( repr(grid), repr(block), workitems ))

        self.kernel.prepared_call( grid, block, vbo_dev_ptr, workitems  )


    def __str__(self):
        source_ = lambda _:["%2s : %s " % (i, line) for i, line in enumerate(_.split("\n"))]
        return "%s\n%s" % ( self.__class__.__name__,  "\n".join(source_(self.kernel_source)))


if __name__ == '__main__':
    pass



