#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.environ['HOME'])   # use alternate to modifying PYTHONPATH as need to capture that 
from env.base.envcap.envcap import main
if __name__ == '__main__':
    main()

