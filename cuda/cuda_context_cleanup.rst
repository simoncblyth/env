CUDA Context Cleanup
====================

Observe that GPU memory usage is high and stays high for minutes
even when no applications are actively using the GPU.  

Is this due to bad actors not cleaning up their CUDA Contexts ?

* does host thread death automatically cleanup any CUDA contexts that were openend ?


Search
-------

* :google:`CUDA Context Cleanup`

https://devblogs.nvidia.com/parallelforall/pro-tip-clean-up-after-yourself-ensure-correct-profiling/ 

    If your application uses the CUDA Runtime API, call cudaDeviceReset() just
    before exiting, or when the application finishes making CUDA calls and using
    device data. If your application uses the CUDA Driver API, call
    cuProfilerStop() on each context to flush the profiling buffers before
    destroying the context with cuCtxDestroy().


