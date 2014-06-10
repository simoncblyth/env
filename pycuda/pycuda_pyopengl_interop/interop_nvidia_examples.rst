NVIDIA Interop Examples
========================


::

    delta:NVIDIA_CUDA-5.5_Samples blyth$ find . -type f -name '*.*' -exec grep -H cuda_gl_interop {} \;
    ./2_Graphics/bindlessTexture/bindlessTexture.cpp:#include <cuda_gl_interop.h>
    ./2_Graphics/Mandelbrot/Mandelbrot.cpp:#include <cuda_gl_interop.h>
    ./2_Graphics/marchingCubes/marchingCubes.cpp:#include <cuda_gl_interop.h>
    ./2_Graphics/simpleGL/simpleGL.cu:#include <cuda_gl_interop.h>
    ./2_Graphics/simpleTexture3D/simpleTexture3D.cpp:#include <cuda_gl_interop.h>
    ./2_Graphics/volumeFiltering/volumeFiltering.cpp:#include <cuda_gl_interop.h>
    ./2_Graphics/volumeRender/volumeRender.cpp:#include <cuda_gl_interop.h>
    ./3_Imaging/bicubicTexture/bicubicTexture.cpp:#include <cuda_gl_interop.h>
    ./3_Imaging/bilateralFilter/bilateralFilter.cpp:#include <cuda_gl_interop.h>
    ./3_Imaging/boxFilter/boxFilter.cpp:#include <cuda_gl_interop.h>
    ./3_Imaging/imageDenoising/imageDenoisingGL.cpp:#include <cuda_gl_interop.h>
    ./3_Imaging/postProcessGL/main.cpp:#include <cuda_gl_interop.h>
    ./3_Imaging/recursiveGaussian/recursiveGaussian.cpp:#include <cuda_gl_interop.h>
    ./3_Imaging/simpleCUDA2GL/main.cpp:#include <cuda_gl_interop.h>
    ./3_Imaging/SobelFilter/SobelFilter.cpp:#include <cuda_gl_interop.h>
    ./5_Simulations/fluidsGL/fluidsGL.cpp:#include <cuda_gl_interop.h>
    ./5_Simulations/nbody/bodysystemcuda.cu:#include <cuda_gl_interop.h>
    ./5_Simulations/nbody/bodysystemcuda_impl.h:#include <cuda_gl_interop.h>
    ./5_Simulations/nbody/nbody.cpp:#include <cuda_gl_interop.h>
    ./5_Simulations/nbody/render_particles.cpp:#include <cuda_gl_interop.h>
    ./5_Simulations/oceanFFT/oceanFFT.cpp:#include <cuda_gl_interop.h>
    ./5_Simulations/particles/particles.cpp:#include <helper_cuda_gl.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h
    ./5_Simulations/particles/particleSystem_cuda.cu:#include <cuda_gl_interop.h>
    ./5_Simulations/smokeParticles/GpuArray.h:#include <cuda_gl_interop.h>
    ./5_Simulations/smokeParticles/particleDemo.cpp:#include <cuda_gl_interop.h>
    ./5_Simulations/smokeParticles/ParticleSystem.cpp:#include <cuda_gl_interop.h>
    ./5_Simulations/smokeParticles/ParticleSystem_cuda.cu:#include <cuda_gl_interop.h>
    ./6_Advanced/FunctionPointers/FunctionPointers.cpp:#include <cuda_gl_interop.h>  // CUDA OpenGL intero
    ./7_CUDALibraries/grabcutNPP/GrabcutMain.cpp:#include <cuda_gl_interop.h>
    ./7_CUDALibraries/randomFog/randomFog.cpp:#include <cuda_gl_interop.h>
    delta:NVIDIA_CUDA-5.5_Samples blyth$ 

