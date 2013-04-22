#!/usr/bin/env python
"""

IOError 
        too many from /dev/stdout 


"""
import sys, logging
log = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    log.info(sys.argv)

