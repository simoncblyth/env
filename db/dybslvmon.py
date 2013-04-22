#!/usr/bin/env python
"""

"""
import sys, logging
log = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    log.info(sys.argv)

