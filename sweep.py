#!/usr/bin/env python
"""
home sweep uses

    from env.tools.svnsweep import main

"""

import logging
from env.tools.svnsweep import Sweeper
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)	
    #swp = Sweeper("~/env","~/G4PB/edocs")
    swp = Sweeper("~/env","~/simoncblyth.bitbucket.io/env")
    print swp
