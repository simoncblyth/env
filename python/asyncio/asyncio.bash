asyncio-source(){   echo ${BASH_SOURCE} ; }
asyncio-edir(){ echo $(dirname $(asyncio-source)) ; }
asyncio-ecd(){  cd $(asyncio-edir); }
asyncio-dir(){  echo $LOCAL_BASE/env/python/asyncio/asyncio ; }
asyncio-cd(){   cd $(asyncio-edir); }
asyncio-vi(){   vi $(asyncio-source) ; }
asyncio-env(){  elocal- ; }
asyncio-usage(){ cat << EOU



API changes

Python3.7. asyncio.async() function is deprecated, use ensure_future()

Ff you know that you have a coroutine and you want it to be scheduled, the
correct API to use is create_task(). The only time when you should be calling
ensure_future() is when you are providing an API (like most of asyncio's own
APIs) that accepts either a coroutine or a Future and you need to do something
to it that requires you to have a Future.


* https://www.integralist.co.uk/posts/python-asyncio/

In older versions of Python, if you were going to manually create your own
Future and schedule it onto the event loop, then you would have used
asyncio.ensure_future (now considered to be a low-level API), but with Python
3.7+ this has been superseded by asyncio.create_task.




talk showing yield underpinnings of asyncio
----------------------------------------------

* https://www.youtube.com/watch?v=M-UcUs7IMIM&feature=youtu.be

  Get to grips with asyncio in Python 3 - Robert Smallshire


task queues
--------------

* https://www.fullstackpython.com/task-queues.html



add_done_callback seems crucial, but not often referred to 
------------------------------------------------------------

* https://www.integralist.co.uk/posts/python-asyncio/

::

  1 import asyncio
  2 
  3 async def foo():
  4     await asyncio.sleep(10)
  5     return "Foo!"
  6 
  7 def got_result(future):
  8     print(f"got the result! {future.result()}")
  9 
 10 async def hello_world():
 11     task = asyncio.create_task(foo())
 12     task.add_done_callback(got_result)
 13     print(task)
 14     await asyncio.sleep(5)
 15     print("Hello World!")
 16     await asyncio.sleep(10)
 17     print(task)
 18 
 19 asyncio.run(hello_world())


Other ways to chain

* https://stackoverflow.com/questions/44345139/python-asyncio-add-done-callback-with-async-def





* https://luminousmen.com/post/asynchronous-programming-blocking-and-non-blocking



Reactor and Proactor patterns 
------------------------------

* https://luminousmen.com/post/asynchronous-programming-cooperative-multitasking

The reactor interface says, "Give me a bunch of your sockets and your
callbacks, and when that socket is ready for I/O, I will call your callback
functions. A reactor job is to react to I/O events by delegating all the
processing to the appropriate handler(worker). The handlers perform processing,
so there is no need to block I/O, as long as handlers or callbacks for events
are registered to take care of them.

The combination works best because cooperative multitasking usually wins,
especially if your connections hang up for a long time. For example, a web
socket is a long-lasting connection. If you allocate a single process or a
single thread to handle a single web socket, you significantly limit the number
of connections to one backend server at a time. And because the connection will
last a long time, it's important to keep many simultaneous connections, while
each connection will have little work to do.

Proactor
--------

* https://www.dre.vanderbilt.edu/~schmidt/PDF/Proactor.pdf
* ~/opticks_refs/Proactor.pdf


asyncio and curio
-------------------

* https://vorpus.org/blog/some-thoughts-on-asynchronous-api-design-in-a-post-asyncawait-world/

* https://asgi.readthedocs.io/en/latest/

* https://docs.python.org/3/library/asyncio-protocol.html


curio and trio
-----------------

* https://curio.readthedocs.io/en/latest/tutorial.html

* https://trio.readthedocs.io/en/stable/

Trio was built from the ground up to take advantage of the latest Python
features, and draws inspiration from many sources, in particular Dave Beazley’s
Curio. 


cheat tips
------------

* https://cheat.readthedocs.io/en/latest/python/asyncio.html

Async code can only run inside an event loop. The event loop is the driver code
that manages the cooperative multitasking.

If it’s useful for some reason, you can create multiple threads and run
different event loops in each of them. For example, Django uses the main thread
to wait for incoming requests, so we can’t run an asyncio event loop there, but
we can start a separate worker thread for our event loop.

To schedule a callback from a different thread, the
AbstractEventLoop.call_soon_threadsafe() method should be used. Example:

loop.call_soon_threadsafe(callback, *args)

Running blocking code in another thread

If you need to call some blocking code from a coroutine, and don’t want to
block the whole thread, you can make it run in another thread using coroutine
AbstractEventLoop.run_in_executor(executor, func, *args):

fn = functools.partial(method, *args)
result = await loop.run_in_executor(None, fn)



pymotw.com/3/asyncio/executors.html
--------------------------------------

* https://pymotw.com/3/asyncio/executors.html


asyncio related modules
---------------------------


* https://www.oreilly.com/library/view/using-asyncio-in/9781492075325/ch04.html


janus : single queue that works with sync_q and async_q APIs 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/aio-libs/janus

sanic : fast http server using asyncio uvloop and ujson 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/huge-success/sanic
* https://sanic.readthedocs.io/en/latest/sanic/getting_started.html


Get to grips with asyncio in Python 3 - Robert Smallshire
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.youtube.com/watch?v=M-UcUs7IMIM&feature=youtu.be


:google:`asyncio continuous message queue monitoring`
-------------------------------------------------------

* https://www.roguelynn.com/words/asyncio-true-concurrency/
* https://www.roguelynn.com/words/asyncio-we-did-it-wrong/

* https://speakerdeck.com/roguelynn/advanced-asyncio-solving-real-world-production-problems?slide=224
* https://www.youtube.com/watch?v=bckD_GK80oY&lc=Ugw9rZpVVSqT9WrCBEJ4AaABAg


concurrent.futures : high level way to use Thread or Process pool with asyncio
---------------------------------------------------------------------------------

* https://docs.python.org/3.8/library/concurrent.futures.html#concurrent.futures.Executor

The concurrent.futures module provides a high-level interface for asynchronously executing callables.

The asynchronous execution can be performed with threads, using
ThreadPoolExecutor, or separate processes, using ProcessPoolExecutor. Both
implement the same interface, which is defined by the abstract Executor class.



can use alternate run loop : uvloop 
-------------------------------------

* https://libuv.org
* http://docs.libuv.org/en/v1.x/





* https://realpython.com/courses/python-3-concurrency-asyncio-module/

* https://docs.python.org/3/library/asyncio-task.html

* https://medium.com/@yeraydiazdiaz/asyncio-for-the-working-python-developer-5c468e6e2e8e

* https://realpython.com/async-io-python/

* https://docs.python.org/3/library/queue.html#module-queue

  A synchronized queue class

* https://docs.python.org/3/library/asyncio-queue.html

  asyncio queues are designed to be similar to classes of the queue module.
  Although asyncio queues are not thread-safe, they are designed to be used
  specifically in async/await code.


* https://www.youtube.com/watch?reload=9&v=0kXaLh8Fz3k&feature=youtu.be&t=10m30s

  John Reese - Thinking Outside the GIL with AsyncIO and Multiprocessing - PyCon 2018

* https://www.youtube.com/watch?v=9zinZmE3Ogk

  Raymond Hettinger, Keynote on Concurrency, PyBay 2017

  52:36 threading.local() wrap any mutable global state when multi-threading  

  54:19 cannot kill threads because they may be holding a lock in which case all other threads deadlock  
   
  55:04 need to message the thread somehow so it can exit gracefully when desired, making 
        sure to release any locks   

  processes are an advantage when there is a need to kill them, as there is no deadlocking 

* https://pybay.com/site_media/slides/raymond2017-keynote/index.html


* :google:`python asyncio and boost asio`

* https://github.com/DmitryKuk/asynchronizer





EOU
}
asyncio-get(){
   local dir=$(dirname $(asyncio-dir)) &&  mkdir -p $dir && cd $dir

}
