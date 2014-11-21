#!/usr/bin/env python
"""
https://docs.python.org/2/library/fcntl.html#fcntl.lockf

ttp://amix.dk/blog/post/19531

f = open(path)
rv = fcntl.fcntl(f, fcntl.F_SETFL, os.O_NDELAY)  
lockdata = struct.pack('hhllhh', fcntl.F_WRLCK, 0, 0, 0, 0, 0)
rv = fcntl.fcntl(f, fcntl.F_SETLKW, lockdata)


"""
import fcntl, os, time
import logging
log = logging.getLogger(__name__)

def have_filelock(path):
    log.info("filelock open for %s " % path )
    fp = open(path, 'w')
    try:
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ok = True
    except IOError:
        ok = False 
    pass
    log.info("filelock ok: %s " % ok)
    return ok


def main():
    logging.basicConfig(level=logging.INFO)
    import sys
    path = sys.argv[1] if len(sys.argv)>1 else "/tmp/env-python-filelock.tmp"

    if have_filelock(path):
        log.info("ok we have the lock") 
        time.sleep(10)
    else:
        log.info("nope we dont have the lock")
        sys.exit(0)
    pass


if __name__ == '__main__':
    main()





