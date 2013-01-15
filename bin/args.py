#!/usr/bin/env python
"""
Simple script to log passed arguments, 
initially created for testing mysql UDF script calling from triggers

NB must change ownership, to allow the UDF invoked call to append to the log::

   sudo chown -R mysql:mysql /tmp/args.log 

Otherwise fails with permission denied message in the mysql error log

"""
import os, sys, logging
log = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        filename='/tmp/args.log',
        filemode='a', # append
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S', 
        )
    log.info("sys.argv[1:] %r " % sys.argv[1:])
    print sys.argv[1:]


