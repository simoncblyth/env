#!/usr/bin/env python
"""
NEXT

#. Oops forgot to GL_DYNAMIC_DRAW, but its working anyhow
#. Extending interop to Chroma propagation ? 
#. all the photon data could conceivably be visualized or at least used by shaders

cuda casting
~~~~~~~~~~~~~~

* :google:`cuda union struct float4 int4`
* http://stackoverflow.com/questions/9044139/accessing-float4-int4-elements-using-a-loop-in-cuda
* https://code.google.com/p/hydrazine/source/browse/trunk/hydrazine/cuda/include/vector_types.h?r=23
* https://svn.ece.lsu.edu/svn/gp/proj-base/boxes/render.cc

  * particle system example using lots of structs




Confirmed no buffer recreation and correct propagation into this kernel
via external commands::

    udp.py --mask RAYLEIGH_SCATTER
    udp.py --mask 0
    udp.py --mask 1
 



"""
import logging
log = logging.getLogger(__name__)

from operator import mul
mul_ = lambda _:reduce(mul, _)          # product of elements 
div_ = lambda num,den:(num+den-1)//den  # integer division trick, rounding up without iffing around

from env.graphics.color.wav2RGB import wav2RGB_cuda
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as ga


kernel_debug = r"""

#include <stdio.h>

    if( id %% 100 == 0){
       printf( "id %%d \n", id );
       printf( "posw.x %%f \n", posw.x );

       printf( "qflags.f.x %%f \n", qflags.f.x );
       printf( "qflags.i.x %%d \n", qflags.i.x );
       printf( "qflags.u.x %%d \n", qflags.u.x );

       printf( "qlht.f.x %%f \n", qlht.f.x );
       printf( "qlht.i.x %%d \n", qlht.i.x );
       printf( "qlht.u.x %%d \n", qlht.u.x );

    }

//#define TEST qflags.u.x 
#define TEST qlht.i.x 

    if      (TEST == 0 ) vbo[CCOLOR] = make_float4( 1., 0., 0., 1.);
    else if (TEST == 1 ) vbo[CCOLOR] = make_float4( 0., 1., 0., 1.);
    else if (TEST == 2 ) vbo[CCOLOR] = make_float4( 0., 0., 1., 1.);
    else                 vbo[CCOLOR] = make_float4( 1., 1., 1., 1.);


    // modify x of slot0 float4, position 
    // vbo[POSITION_WEIGHT] = make_float4( posw.x + 10. , posw.y , posw.z, posw.w );  
    
    // grow the direction
    // vbo[DIRECTION_WAVELENGTH] = make_float4( dirw.x*1.01 , dirw.y*1.01 , dirw.z*1.01, dirw.w ); 

    // set a constant momdir for all photons
    // vbo[DIRECTION_WAVELENGTH] = make_float4( 100. , 100. , 100. , 0. ); 
 

"""



kernel_source = r"""
//
// CUDA kernel checking the modification of OpenGL VBO 
//
//  #. depends on the simple quad*numquad structure of the VBO 
//     created by DAEPhotonsData.create_data 
// 
//  #. misuses of quad union to interpret float4 as uint4 or int4  
//

""" + wav2RGB_cuda + r"""


__constant__ int4 g_mask ;

union quad
{
   float4 f ;
   int4   i ;
   uint4  u ;
};

#define POSITION_WEIGHT      (id*%(numquad)s)
#define DIRECTION_WAVELENGTH (id*%(numquad)s+1)
#define POLARIZATION_TIME    (id*%(numquad)s+2)
#define CCOLOR               (id*%(numquad)s+3)
#define FLAGS                (id*%(numquad)s+4)
#define LAST_HIT_TRIANGLE    (id*%(numquad)s+5)

__global__ void jump(float4* vbo, int items )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x; 
    if (id >= items ) return ;

    float4 posw, dirw, polt, ccol ; 
    union quad qflags, qlht  ; 

    posw     = vbo[POSITION_WEIGHT];     
    dirw     = vbo[DIRECTION_WAVELENGTH]; 
    polt     = vbo[POLARIZATION_TIME];  
    ccol     = vbo[CCOLOR];  
    qflags.f = vbo[FLAGS] ; 
    qlht.f   = vbo[LAST_HIT_TRIANGLE] ;    

    // skip only when bitwise and quality masks are "enabled" by not being -1
    bool skip = ((g_mask.x > -1 ) && ( qflags.u.x & g_mask.x ) == 0 ) ||
                ((g_mask.y > -1 ) && ( qflags.u.x != g_mask.y )) ;     
 
    vbo[CCOLOR] = skip ? make_float4( 0.5, 0.5, 0.5, 0.) : wav2color( dirw.w ) ; // greyed out OR color from wavelength 
}
"""



class DAEPhotonsKernel(object):
    """
    #. adopt ~/e/cuda/cuda_launch approach
    #. hmm would be good to be able to change the kernel source without restarting 
    """
    def __init__(self, dphotons):
        self.dphotons = dphotons
        self.compile_kernel()
        self.initialize_constants()

    def compile_kernel(self):
        """
        #. compile kernel and extract __constant__ symbol addresses
        """
        ctx = { 'numquad':self.dphotons.numquad }
        self.kernel_source = kernel_source % ctx 

        module = SourceModule(self.kernel_source )
        kernel = module.get_function("jump")
        kernel.prepare("Pi")

        self.g_mask = module.get_global("g_mask")[0]  
        self._mask = None
        self.kernel = kernel

    def initialize_constants(self):
        self.mask = [-1,-1,-1,-1]

    def update_constants(self):
        self.mask = self.dphotons.param.kernel_mask

    def _get_mask(self):
        return self._mask 
    def _set_mask(self, mask):
        if mask == self._mask:return
        self._mask = mask
        log.info("_set_mask : memcpy_htod %s " % repr(mask))
        cuda.memcpy_htod(self.g_mask, ga.vec.make_int4(*mask))
    mask = property(_get_mask, _set_mask, doc="setter copies to device __constant__ memory, getter returns cached value") 


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



