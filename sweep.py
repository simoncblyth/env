#!/usr/bin/env python
import logging
from env.tools.svnsweep import Sweeper

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)	
    swp = Sweeper("~/env","~/G4PB/edocs")
    print swp
