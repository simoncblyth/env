PyCUDA Error Handling
=========================

Active Git Repo
-----------------

So need to follow the history of the code.

* http://git.tiker.net/pycuda.git
* see also :doc:`/pycuda/pycuda_bash`  **pycuda-get**


OSX GPU Panic Symptoms
------------------------

Panics are always preceeded by screen fulls of the below and a few minutes of GUI freeze::

   cuMemFree failed: launch timeout
   PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
   cuMemFree failed: launch timeout
   PyCUDA WARNING: a clean-up operation failed (dead context maybe?)

Normally do not get to see the start of the error due to screen filling with 
the above followed by GUI freeze. But somehow (potentially different conditions) on one occasion::

    INFO:chroma:MOUSEBUTTONDOWN 4 [ 330.98363295  330.98363295  330.98363295] 
    INFO:chroma:MOUSEBUTTONUP 4 
    INFO:chroma:MOUSEBUTTONDOWN 4 [ 330.98363295  330.98363295  330.98363295] 
    Traceback (most recent call last):
      File "./simplecamera.py", line 285, in <module>
        camera._run()
      File "./simplecamera.py", line 271, in _run
        self.process_event(event)
      File "./simplecamera.py", line 136, in process_event
        self.translate(v)
      File "./simplecamera.py", line 119, in translate
        self.update()
      File "./simplecamera.py", line 125, in update
        pixels = self.pixels_gpu.get()
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/gpuarray.py", line 252, in get
        drv.memcpy_dtoh(ary, self.gpudata)
    pycuda._driver.LaunchError: cuMemcpyDtoH failed: launch timeout


Sometime also see the below http://git.tiker.net/pycuda.git/blame/HEAD:/src/cpp/cuda.hpp::

    e52d43d0
    AK  810   inline context_stack::~context_stack()
    811   {
    812     if (!m_stack.empty())
    813     {
    d34ab465    814       std::cerr
    e52d43d0
    AK  815         << "-------------------------------------------------------------------" << std::endl
    816         << "PyCUDA ERROR: The context stack was not empty upon module cleanup." << std::endl
    817         << "-------------------------------------------------------------------" << std::endl
    818         << "A context was still active when the context stack was being" << std::endl
    819         << "cleaned up. At this point in our execution, CUDA may already" << std::endl
    820         << "have been deinitialized, so there is no way we can finish" << std::endl
    821         << "cleanly. The program will be aborted now." << std::endl
    822         << "Use Context.pop() to avoid this problem." << std::endl
    823         << "-------------------------------------------------------------------" << std::endl;
    824       abort();



PyCUDA FAQ Extract regards launch failure
--------------------------------------------

* http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions

Downgrading of former cleanup failures to warnings in 0.93 mentioned
below smells like
it could be contributing to the mishandling of CUDA errors.

Natural questions

#. how can CUDA timeouts be handled so as to ABORT cleanly without bringing the system down in GPU panic ?
#. why is pycuda not doing that ?
#. how can problem be recast to avoid timeouts ?  


My program terminates after a launch failure. Why? (EXTRACT FROM FAQ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* This should not be an issue any more with 0.93 and later, where cleanup failures have been downgraded to warnings.

You're probably seeing something like this::

    Traceback (most recent call last):
      File "fail.py", line 32, in <module>
        cuda.memcpy_dtoh(a_doubled, a_gpu)
    RuntimeError: cuMemcpyDtoH failed: launch failed
    terminate called after throwing an instance of 'std::runtime_error'
      what():  cuMemFree failed: launch failed
    zsh: abort      python fail.py

What's going on here? First of all, recall that launch failures in CUDA are
asynchronous. So the actual traceback does not point to the failed kernel
launch, it points to the next CUDA request after the failed kernel.

Next, as far as I can tell, a CUDA context becomes invalid after a launch
failure, and all following CUDA calls in that context fail. Now, that includes
cleanup (see the cuMemFree in the traceback?) that PyCUDA tries to perform
automatically. Here, a bit of PyCUDA's C++ heritage shows through. While
performing cleanup, we are processing an exception (the launch failure reported
by cuMemcpyDtoH). If another exception occurs during exception processing, C++
gives up and aborts the program with a message.

In principle, this could be handled better. If you're willing to dedicate time
to this, I'll likely take your patch.


where was this downgrade done ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Downgraded cleanup failures from errors to warnings.

* http://git.tiker.net/pycuda.git/commit/7a7150055b18e0a651f719e19f4f6be2d834b601?f=src/cpp/cuda.hpp


::

    Sun, 24 May 2009 00:19:03 +0000 (19:19 -0500)

::

    @@ -908,7 +919,7 @@ namespace cuda
       inline 
       void mem_free(CUdeviceptr devptr)
       {
    -    CUDAPP_CALL_GUARDED(cuMemFree, (devptr));
    +    CUDAPP_CALL_GUARDED_CLEANUP(cuMemFree, (devptr));
       }

::

    #define CUDAPP_CALL_GUARDED(NAME, ARGLIST) \
      { \
        CUDAPP_PRINT_CALL_TRACE(#NAME); \
        CUresult cu_status_code; \
        cu_status_code = NAME ARGLIST; \
        if (cu_status_code != CUDA_SUCCESS) \
          throw cuda::error(#NAME, cu_status_code);\
      }
    #define CUDAPP_CALL_GUARDED_CLEANUP(NAME, ARGLIST) \
      { \
        CUDAPP_PRINT_CALL_TRACE(#NAME); \
        CUresult cu_status_code; \
        cu_status_code = NAME ARGLIST; \
        if (cu_status_code != CUDA_SUCCESS) \
          std::cerr \
            << "PyCUDA WARNING: a clean-up operation failed (dead context maybe?)" \
            << std::endl \
            << cuda::error::make_message(#NAME, cu_status_code) \
            << std::endl; \




PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
-------------------------------------------------------------------

* http://git.tiker.net/pycuda.git/blob/HEAD:/src/cpp/cuda.hpp

::

     146 #define CUDAPP_CALL_GUARDED_CLEANUP(NAME, ARGLIST) \
     147   { \
     148     CUDAPP_PRINT_CALL_TRACE(#NAME); \
     149     CUresult cu_status_code; \
     150     cu_status_code = NAME ARGLIST; \
     151     CUDAPP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
     152     if (cu_status_code != CUDA_SUCCESS) \
     153       std::cerr \
     154         << "PyCUDA WARNING: a clean-up operation failed (dead context maybe?)" \
     155         << std::endl \
     156         << pycuda::error::make_message(#NAME, cu_status_code) \
     157         << std::endl; \
     158   }
     159 #define CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(TYPE) \
     160   catch (pycuda::cannot_activate_out_of_thread_context) \
     161   { } \
     162   catch (pycuda::cannot_activate_dead_context) \
     163   { \
     164     /* PyErr_Warn( \
     165         PyExc_UserWarning, #TYPE " in dead context was implicitly cleaned up");*/ \
     166   }
     167   // In all likelihood, this TYPE's managing thread has exited, and
     168   // therefore its context has already been deleted. No need to harp
     169   // on the fact that we still thought there was cleanup to do.
     170 
     171 // }}}



::

    simon:pycuda blyth$ pwd
    /usr/local/env/pycuda
    simon:pycuda blyth$ find . -name '*.hpp' -exec grep -H CUDAPP_CALL_GUARDED_CLEANUP {} \;
    ./src/cpp/cuda.hpp:#define CUDAPP_CALL_GUARDED_CLEANUP(NAME, ARGLIST) \
    ./src/cpp/cuda.hpp:            CUDAPP_CALL_GUARDED_CLEANUP(cuCtxDetach, (m_context));
    ./src/cpp/cuda.hpp:              CUDAPP_CALL_GUARDED_CLEANUP(cuCtxPushCurrent, (m_context));
    ./src/cpp/cuda.hpp:              CUDAPP_CALL_GUARDED_CLEANUP(cuCtxDetach, (m_context));
    ./src/cpp/cuda.hpp:          CUDAPP_CALL_GUARDED_CLEANUP(cuStreamDestroy, (m_stream));
    ./src/cpp/cuda.hpp:            CUDAPP_CALL_GUARDED_CLEANUP(cuArrayDestroy, (m_array));
    ./src/cpp/cuda.hpp:          CUDAPP_CALL_GUARDED_CLEANUP(cuTexRefDestroy, (m_texref));
    ./src/cpp/cuda.hpp:          CUDAPP_CALL_GUARDED_CLEANUP(cuModuleUnload, (m_module));
    ./src/cpp/cuda.hpp:    CUDAPP_CALL_GUARDED_CLEANUP(cuMemFree, (devptr));
    ./src/cpp/cuda.hpp:            CUDAPP_CALL_GUARDED_CLEANUP(cuIpcCloseMemHandle, (m_devptr));
    ./src/cpp/cuda.hpp:    CUDAPP_CALL_GUARDED_CLEANUP(cuMemFreeHost, (ptr));
    ./src/cpp/cuda.hpp:    CUDAPP_CALL_GUARDED_CLEANUP(cuMemHostUnregister, (ptr));
    ./src/cpp/cuda.hpp:          CUDAPP_CALL_GUARDED_CLEANUP(cuEventDestroy, (m_event));
    ./src/cpp/cuda_gl.hpp:            CUDAPP_CALL_GUARDED_CLEANUP(cuGLUnregisterBufferObject, (m_handle));
    ./src/cpp/cuda_gl.hpp:            CUDAPP_CALL_GUARDED_CLEANUP(cuGLUnmapBufferObject, (m_buffer_object->handle()));
    ./src/cpp/cuda_gl.hpp:            CUDAPP_CALL_GUARDED_CLEANUP(
    ./src/cpp/cuda_gl.hpp:            CUDAPP_CALL_GUARDED_CLEANUP(cuGraphicsUnmapResources,

After context lines for some::

    simon:pycuda blyth$ find . -name '*.hpp' -exec grep -A 1 -H CUDAPP_CALL_GUARDED_CLEANUP {} \;
    ...
    ./src/cpp/cuda_gl.hpp:            CUDAPP_CALL_GUARDED_CLEANUP(cuGLUnregisterBufferObject, (m_handle));
    ./src/cpp/cuda_gl.hpp-            m_valid = false;
    --
    ./src/cpp/cuda_gl.hpp:            CUDAPP_CALL_GUARDED_CLEANUP(cuGLUnmapBufferObject, (m_buffer_object->handle()));
    ./src/cpp/cuda_gl.hpp-            m_valid = false;
    --
    ./src/cpp/cuda_gl.hpp:            CUDAPP_CALL_GUARDED_CLEANUP(
    ./src/cpp/cuda_gl.hpp-                cuGraphicsUnregisterResource, (m_resource));
    --
    ./src/cpp/cuda_gl.hpp:            CUDAPP_CALL_GUARDED_CLEANUP(cuGraphicsUnmapResources,
    ./src/cpp/cuda_gl.hpp-                (1, &res, s_handle));



launch_kernel
--------------
::

    1345 #if CUDAPP_CUDA_VERSION >= 4000
    1346       void launch_kernel(py::tuple grid_dim_py, py::tuple block_dim_py,
    1347           py::object parameter_buffer,
    1348           unsigned shared_mem_bytes, py::object stream_py)
    1349       {
    1350         const unsigned axis_count = 3;
    1351         unsigned grid_dim[axis_count];
    1352         unsigned block_dim[axis_count];
    1353 
    1354         for (unsigned i = 0; i < axis_count; ++i)
    1355         {
    1356           grid_dim[i] = 1;
    1357           block_dim[i] = 1;
    1358         }
    1359 
    1360         pycuda_size_t gd_length = py::len(grid_dim_py);
    1361         if (gd_length > axis_count)
    1362           throw pycuda::error("function::launch_kernel", CUDA_ERROR_INVALID_HANDLE,
    1363               "too many grid dimensions in kernel launch");
    1364 
    1365         for (unsigned i = 0; i < gd_length; ++i)
    1366           grid_dim[i] = py::extract<unsigned>(grid_dim_py[i]);
    1367 
    1368         pycuda_size_t bd_length = py::len(block_dim_py);
    1369         if (bd_length > axis_count)
    1370           throw pycuda::error("function::launch_kernel", CUDA_ERROR_INVALID_HANDLE,
    1371               "too many block dimensions in kernel launch");
    1372 
    1373         for (unsigned i = 0; i < bd_length; ++i)
    1374           block_dim[i] = py::extract<unsigned>(block_dim_py[i]);
    1375 
    1376         PYCUDA_PARSE_STREAM_PY;
    1377 
    1378         const void *par_buf;
    1379         PYCUDA_BUFFER_SIZE_T py_par_len;
    1380         if (PyObject_AsReadBuffer(parameter_buffer.ptr(), &par_buf, &py_par_len))
    1381           throw py::error_already_set();
    1382         size_t par_len = py_par_len;
    1383 
    1384         void *config[] = {
    1385           CU_LAUNCH_PARAM_BUFFER_POINTER, const_cast<void *>(par_buf),
    1386           CU_LAUNCH_PARAM_BUFFER_SIZE, &par_len,
    1387           CU_LAUNCH_PARAM_END
    1388         };
    1389 
    1390         CUDAPP_CALL_GUARDED(
    1391             cuLaunchKernel, (m_function,
    1392               grid_dim[0], grid_dim[1], grid_dim[2],
    1393               block_dim[0], block_dim[1], block_dim[2],
    1394               shared_mem_bytes, s_handle, 0, config
    1395               ));
    1396       }
    1397 
    1398 #endif



guarded cleanup with/without catch
-------------------------------------

In current (2014/03/13) pycuda the catch does nothing anyhow::

    2046   // {{{ event
    2047   class event : public boost::noncopyable, public context_dependent
    2048   {
    2049     private:
    2050       CUevent m_event;
    2051 
    2052     public:
    2053       event(unsigned int flags=0)
    2054       { CUDAPP_CALL_GUARDED(cuEventCreate, (&m_event, flags)); }
    2055 
    2056       event(CUevent evt)
    2057       : m_event(evt)
    2058       { }
    2059 
    2060       ~event()
    2061       {
    2062         try
    2063         {
    2064           scoped_context_activation ca(get_context());
    2065           CUDAPP_CALL_GUARDED_CLEANUP(cuEventDestroy, (m_event));
    2066         }
    2067         CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(event);
    2068       }


