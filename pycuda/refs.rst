PyCUDA References
==================


* http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions
* https://pypi.python.org/pypi/pudb


How about multiple GPUs?
-------------------------

* http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions#How_about_multiple_GPUs.3F


Two ways:

* Allocate two contexts, juggle (pycuda.driver.Context.push and
  pycuda.driver.Context.pop) them from that one process.

* Work with several processes or threads, using MPI, multiprocesing or
  threading. As of Version 0.90.2, PyCUDA will release the Global Interpreter
  Lock while it is waiting for CUDA operations to finish. As of version 0.93,
  PyCUDA will actually work when used together with threads. Also see threading,
  below.


How does PyCUDA handle threading?
-----------------------------------

* http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions#How_does_PyCUDA_handle_threading.3F

As of version 0.93, PyCUDA supports threading. There is an example of how this
can be done in examples/multiple_threads.py in the PyCUDA distribution. When
you use threading in PyCUDA, you should be aware of one peculiarity, though.
Contexts in CUDA are a per-thread affair, and as such all contexts associated
with a thread as well as GPU memory, arrays and other resources in that context
will be automatically freed when the thread exits. PyCUDA will notice this and
will not try to free the corresponding resource--it's already gone after all.

There is another, less intended consequence, though: If Python's garbage
collector finds a PyCUDA object it wishes to dispose of, and PyCUDA, upon
trying to free it, determines that the object was allocated outside of the
current thread of execution, then that object is quietly leaked. This properly
handles the above situation, but it mishandles a situation where:

* You use reference cycles in a GPU driver thread, necessitating the GC (over
  just regular reference counts).  
* You require cleanup to be performed before thread exit.  
* You rely on PyCUDA to perform this cleanup.  

To entirely avoid the problem, do one of the following:

* Use multiprocessing instead of threading.  
* Explicitly call free on the objects you want cleaned up.


How do I use the NVidia Visual Profiler with PyCUDA applications?
--------------------------------------------------------------------

* http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions#How_do_I_use_the_NVidia_Visual_Profiler_with_PyCUDA_applications.3F

In the Visual Profiler, create a new session with File set to your Python
executable and Arguments set to your main Python script.

There is one more important change you have to make before the timeline will
show any events. Before your application quits, make the call

pycuda.driver.stop_profiler() so that the profiling data buffers get flushed to
file.


