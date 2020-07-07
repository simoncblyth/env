# === func-gen- : cuda/texture/cudatex fgp cuda/texture/cudatex.bash fgn cudatex fgh cuda/texture
cudatex-src(){      echo cuda/texture/cudatex.bash ; }
cudatex-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cudatex-src)} ; }
cudatex-vi(){       vi $(cudatex-source) ; }
cudatex-usage(){ cat << EOU

CUDATEX
========


Ref
----

* http://cuda-programming.blogspot.tw/2013/04/texture-references-object-in-cuda.html

Next Steps
-----------

* arrange to wavelength dependent props handling in ggeo- so have some
  real numbers to experiment with 


Cubic Spline Interpolation Filtering
---------------------------------------

* :google:`GPU texture Cubic Spline Interpolation`

* https://github.com/DannyRuijters/CubicInterpolationCUDA/blob/master/code/internal/bspline_kernel.cu
* http://dannyruijters.nl/cubicinterpolation/
* http://dannyruijters.nl/docs/cudaPrefilter3.pdf

* https://forums.developer.nvidia.com/t/new-cubic-interpolation-in-cuda-cubic-b-spline-interpolation/5839/7

  * CUDA SDK includes a bicubic filtering sample that implements the method in GPU Gems. 
  * Thread discussion between dannyruijters and Simon_Green (author of CUDA SDK example?) 




Interpolation filter modes : excellent explanation from Roger Dahl
--------------------------------------------------------------------

* https://stackoverflow.com/questions/10643790/texture-memory-tex2d-basics

Why the 0.5f ?::

    uint f = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint c = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint read = tex2D( refTex, c+0.5f, f+0.5f);

Roger Dahl:

In graphics, a texture is a set of samples that describes the visual appearance
of a surface. A sample is a point. That is, it has no size (as opposed to a
pixel that has a physical size). When using samples to determine the colors of
pixels, each sample is positioned in the exact center of its corresponding
pixel. When addressing pixels with whole number coordinates, the exact center
for a given pixel becomes its whole number coordinate plus an offset of 0.5 (in
each dimension).

In other words, adding 0.5 to texture coordinates ensures that, when reading
from those coordinates, the exact value of the sample for that pixel is
returned.

However, it is only when filterMode for the texture has been set to
cudaFilterModeLinear that the value that is read from a texture varies within a
pixel. In that mode, reading from coordinates that are not in the exact center
of a pixel returns values that are interpolated between the sample for the
given pixel and the samples for neighboring pixels. So, adding 0.5 to whole
number coordinates effectively negates the cudaFilterModeLinear mode. But,
since adding 0.5 to the texture coordinates takes up cycles in the kernel, it
is better to simply turn off the interpolation by setting filterMode to
cudaFilterModePoint. Then, reading from any coordinate within a pixel returns
the exact texture sample value for that pixel, and so, texture samples can be
read directly by using whole numbers.


* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory

The filtering mode which specifies how the value returned when fetching the
texture is computed based on the input texture coordinates. Linear texture
filtering may be done only for textures that are configured to return
floating-point data. It performs low-precision interpolation between
neighboring texels. When enabled, the texels surrounding a texture fetch
location are read and the return value of the texture fetch is interpolated
based on where the texture coordinates fell between the texels. Simple linear
interpolation is performed for one-dimensional textures, bilinear interpolation
for two-dimensional textures, and trilinear interpolation for three-dimensional
textures. Texture Fetching gives more details on texture fetching. The
filtering mode is equal to cudaFilterModePoint or cudaFilterModeLinear. If it
is cudaFilterModePoint, the returned value is the texel whose texture
coordinates are the closest to the input texture coordinates. If it is
cudaFilterModeLinear, the returned value is the linear interpolation of the two
(for a one-dimensional texture), four (for a two dimensional texture), or eight
(for a three dimensional texture) texels whose texture coordinates are the
closest to the input texture coordinates. cudaFilterModeLinear is only valid
for returned values of floating-point type.

* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching


Wavelength dependent material/surface property lookups via texture objects ?
-------------------------------------------------------------------------------

* one dimensional, as interpolating off single qty: wavelength
* hardware linear interpolation 
* values can be float4 (double4 ?) 

  * pack four properties into one texture, maybe single
    texture for each material and surface would be sufficient

Compare with Chroma photon.h:fill_state::

    223 
    224     s.refractive_index1 = interp_property(material1, p.wavelength,
    225                                           material1->refractive_index);
    226     s.refractive_index2 = interp_property(material2, p.wavelength,
    227                                           material2->refractive_index);
    228     s.absorption_length = interp_property(material1, p.wavelength,
    229                                           material1->absorption_length);
    230     s.scattering_length = interp_property(material1, p.wavelength,
    231                                           material1->scattering_length);
    232     s.reemission_prob = interp_property(material1, p.wavelength,
    233                                         material1->reemission_prob);
    234 

    (chroma_env)delta:cuda blyth$ grep -l interp_property *.cu *.h
    cerenkov.h
    geometry.h
    photon.h

::

    741 __device__ int
    742 propagate_at_surface(Photon &p, State &s, curandState &rng, Geometry *geometry,
    743                      bool use_weights=false)
    744 {
    745     Surface *surface = geometry->surfaces[s.surface_index];
    746 
    747     if (surface->model == SURFACE_COMPLEX)
    748         return propagate_complex(p, s, rng, surface, use_weights);
    749     else if (surface->model == SURFACE_WLS)
    750         return propagate_at_wls(p, s, rng, surface, use_weights);
    751     else
    752     {
    753         // use default surface model: do a combination of specular and
    754         // diffuse reflection, detection, and absorption based on relative
    755         // probabilties
    756 
    757         // since the surface properties are interpolated linearly, we are
    758         // guaranteed that they still sum to 1.0.
    759         float detect = interp_property(surface, p.wavelength, surface->detect);
    760         float absorb = interp_property(surface, p.wavelength, surface->absorb);
    761         float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    762         float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    763 








EOU
}
cudatex-bdir(){ echo $(local-base)/env/cuda/texture ; }
cudatex-sdir(){ echo $(env-home)/cuda/texture ; }
cudatex-scd(){  cd $(cudatex-sdir); }
cudatex-bcd(){  cd $(cudatex-bdir); }
cudatex-cd(){   cudatex-scd ; }

cudatex-env(){      
   elocal- 
   cuda-
}


cudatex-name(){
  case $1 in 
    cudatex-tt) echo cuda_texture_test ;;
    cudatex-to) echo cuda_texture_object ;;
  esac
}


cudatex-tt(){ cudatex-- $FUNCNAME ; }
cudatex-to(){ cudatex-- $FUNCNAME ; }

cudatex-options(){ cat << EOO
-arch=sm_30
EOO
}

cudatex--(){
   local fn=${1:-cudatex-tt}
   local name=$(cudatex-name $fn)

   mkdir -p $(cudatex-bdir)
   local cmd="nvcc -o $(cudatex-bdir)/$name $(cudatex-options)  $(cudatex-sdir)/$name.cu"
   echo $msg $cmd
   eval $cmd

   cmd="$(cudatex-bdir)/$name"

   echo $msg $cmd
   eval $cmd
 
}
