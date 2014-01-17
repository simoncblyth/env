cube example
===============

Trying out snippet from http://chroma.bitbucket.org/render.html

Usage::

   chroma-
   cd ~/e/chroma/test
   ./cube.py 


Controls on macbook pro
-------------------------

* rotate camera around object: one finger drag around
* rotate camera: control + one finger drag
* zoom: two finger drag in/out
* pan: shift + one finger drag 

* some function key bindings are obscured by OSX system ones brightness/volume etc.. 

Issues
-------

Needs nvcc in PATH
~~~~~~~~~~~~~~~~~~~~~~~~

Initially got a giant stack trace in ipython ending::

    ExecError: error invoking 'nvcc --preprocess --use_fast_math -I/usr/local/env/chroma_env/src/chroma/chroma/cuda -arch sm_30 -m64 -I/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/cuda /var/folders/qm/1p5gh0x94l3b0xqc8dpr9    yn40000gn/T/tmpx1J3Va.cu --compiler-options -P': [Errno 2] No such file or directory

This was fixed by adding "cuda-" to "chroma-" environment setup providing 
access to nvcc via PATH.  Perhaps that should be in VIRTUAL_ENV/env.d ?


CUDA Compilation Warnings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Succeeds to pull up a window containing a 3D cube that can 
interact with, but gives warnings::

    (chroma_env)delta:test blyth$ ./cube.py 
    /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/characterize.py:40: UserWarning: The CUDA compiler succeeded, but said the following:
    kernel.cu(45): warning: variable "NCHILD_MASK" was declared but never referenced
    kernel.cu(45): warning: variable "NCHILD_MASK" was declared but never referenced
    ...  % (preamble, type_name), no_extern_c=True)
    /usr/local/env/chroma_env/src/chroma/chroma/gpu/tools.py:32: UserWarning: The CUDA compiler succeeded, but said the following:
    kernel.cu(198): warning: integer conversion resulted in a change of sign
    kernel.cu(198): warning: integer conversion resulted in a change of sign
    ... no_extern_c=True)

Warnings only show up on first run, presumably the compilation is cached somewhere::

    (chroma_env)delta:test blyth$ ./cube.py 


Initial Crash, no device
~~~~~~~~~~~~~~~~~~~~~~~~~

Initially crashed when run from macports ipython, with::

    In [1]: import chroma
    In [2]: chroma.view(chroma.make.cube(100))
    Merging 24 nodes to 8 parents
    Merging 8 nodes to 2 parents
    Merging 2 nodes to 1 parent
    Process Camera-1:
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/multiprocessing/process.py", line 258, in _bootstrap
        self.run()
      File "/usr/local/env/chroma_env/src/chroma/chroma/camera.py", line 629, in run
        self.init_gpu()
      File "/usr/local/env/chroma_env/src/chroma/chroma/camera.py", line 82, in init_gpu
        self.context = gpu.create_cuda_context(self.device_id)
      File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/tools.py", line 129, in create_cuda_context
        cuda.init()
    RuntimeError: cuInit failed: no device

Perhaps this is first access timeout or somesuch ? 


Working
---------

Subsequently is working using either::

    (chroma_env)delta:test blyth$ ipython
    WARNING: Attempting to work in a virtualenv. If you encounter problems, please install IPython inside the virtualenv.
    Python 2.7.6 (default, Nov 18 2013, 15:12:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.1.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: import chroma

    In [2]: chroma.view(chroma.make.cube(100))

OR::

    In [1]: run cube.py


