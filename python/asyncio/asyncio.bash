asyncio-source(){   echo ${BASH_SOURCE} ; }
asyncio-edir(){ echo $(dirname $(asyncio-source)) ; }
asyncio-ecd(){  cd $(asyncio-edir); }
asyncio-dir(){  echo $LOCAL_BASE/env/python/asyncio/asyncio ; }
asyncio-cd(){   cd $(asyncio-dir); }
asyncio-vi(){   vi $(asyncio-source) ; }
asyncio-env(){  elocal- ; }
asyncio-usage(){ cat << EOU



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


EOU
}
asyncio-get(){
   local dir=$(dirname $(asyncio-dir)) &&  mkdir -p $dir && cd $dir

}
