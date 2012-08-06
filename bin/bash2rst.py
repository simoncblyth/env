#!/usr/bin/env python
"""

"""
import sys, logging
from env.doc.bash2rst import main

if __name__ == '__main__':
    import sys
    try:
        logging.basicConfig(level=logging.INFO)
    except:
        logging.getLogger().setLevel(logging.INFO)

    assert len(sys.argv) == 2 , sys.argv
    main(sys.argv[1])

