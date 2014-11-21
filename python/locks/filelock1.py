#!/usr/bin/env python
"""
Based on

* http://code.activestate.com/recipes/519626-simple-file-based-mutex-for-very-basic-ipc/
* http://www.evanfosmark.com/2009/01/cross-platform-file-locking-support-in-python/

Observations:

* killing the holder of the lock with `kill -9` leaves a stale lockfile 

* deleting stale locks will work in the kill case, 
  but what happens to a stuck process holding the lock 
  that gets removed from under it ?

"""

import os, time, logging
log = logging.getLogger(__name__)

class FileLock:
    def __init__(self, filename, max_locking_minutes=0.5):
        self.filename = filename
        self.fd = None
        self.pid = os.getpid()
        self.max_locking_minutes = max_locking_minutes

    def remove_stale(self):
        if not os.path.exists(self.filename):return
        assert os.path.isfile(self.filename)

        age = self.locking_minutes()
        if age > self.max_locking_minutes:
            log.info("removing stale lockfile %s : age %5.2f exceeds max_locking_minutes %5.2f " % ( self.filename, age, self.max_locking_minutes))
            log.info("process was %s " % self.locking_process())
            assert len(self.filename.split("/")) > 1 and len(self.filename) > 5, "sanity check failure for filename %s  " % self.filename
            os.remove(self.filename)
        else:
            log.debug("lockfile remains valid ")      
        pass

    def locking_pid(self):
        return file(self.filename).read()

    def locking_time(self):
        return time.ctime(os.path.getmtime(self.filename))

    def locking_minutes(self):
        mt = os.path.getmtime(self.filename)
        return (time.time() - mt)/60.

    def locking_process(self):
        return "PID %s since %s thats %5.2f minutes " % (self.locking_pid(), self.locking_time(), self.locking_minutes())  

    def acquire(self):
        try:
            self.fd = os.open(self.filename, os.O_CREAT|os.O_EXCL|os.O_RDWR)
            os.write(self.fd, "%d" % self.pid)
            rc = 1    # return ints so this can be used in older Pythons
        except OSError:
            self.fd = None
            rc = 0
        return rc


    def release(self):
        if not self.fd:
            return 0
        try:
            os.close(self.fd)
            os.remove(self.filename)
            self.fd = None
            return 1
        except OSError:
            return 0


    def __del__(self):
        self.release()


def main():
    logging.basicConfig(level=logging.INFO)
    lock = FileLock("/tmp/lock.file")
    while 1:
        lock.remove_stale()
        if lock.acquire():
            raw_input("[%s] acquired lock : %s : Press ENTER to release:" % (lock.pid, lock.locking_process()) )
            lock.release()
            raw_input("[%s] released lock : Press ENTER to try to aquire again:" % lock.pid )
        else:
            raw_input("[%s] Unable to acquire lock. Held by %s  : Press ENTER to retry:" % (lock.pid, lock.locking_process()) )

if __name__ == "__main__":
    main()

