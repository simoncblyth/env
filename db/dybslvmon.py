#!/usr/bin/env python
"""
Ideas:

#. big file and mainly interested in the tail, so just parse the last 10000 lines or smth 
#. last timestamp in the log not older than a cut
#. counts of error patterns dont exceed cuts 

Error patterns:

IOError 
        too many from /dev/stdout 


"""
import sys, logging
log = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    log.info(sys.argv)

