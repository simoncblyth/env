#!/usr/bin/env python
"""

"""
import sys, logging
from env.doc.bash2rst import main

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    assert len(sys.argv) == 2 , sys.argv
    main(sys.argv[1])

