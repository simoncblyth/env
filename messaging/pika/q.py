#!/usr/bin/env python
"""
    http://www.ibm.com/developerworks/aix/library/au-threadingpython/

    http://docs.python.org/library/queue.html#QueueObjects

By setting daemonic threads to true, it allows the main thread, or program, 
to exit if only daemonic threads are alive. 
This creates a simple way to control the flow of the program, 
because you can then join on the queue, or wait until the queue is empty, before exiting.


"""
import Queue
import threading
import urllib2
import time
          
hosts = ["http://yahoo.com", "http://google.com", "http://amazon.com", "http://ibm.com", "http://apple.com"]
          
          
class ThreadUrl(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
    
    nam = property( lambda self:"%s:%s" % ( self.__class__.__name__ , self.getName() ))
      
    def run(self):
        print "launch %s " % self.nam
        while True:
            host = self.queue.get()
            url = urllib2.urlopen(host)
            #print url.read(1024)
            self.queue.task_done()  ## decrement a count ... to eventually allow join to complete 
            print "completed for %s " % host        
 
 
if __name__ == '__main__':
    start = time.time()

    #q = Queue.Queue()   # fifo
    q = Queue.LifoQueue()  # lifo   from py26

    for i in range(5):
        t = ThreadUrl(q)
        t.setDaemon(True)
        t.start()
              
    #populate queue with data   
    for host in hosts:
        q.put(host)
           
    #wait on the queue until everything has been processed     
    q.join()
    
    print "Elapsed Time: %s" % (time.time() - start)



