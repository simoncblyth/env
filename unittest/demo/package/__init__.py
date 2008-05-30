"""package docstring in the __init__.py file"""
import os
import sys
sys.path.append(os.path.join(os.environ['ENV_HOME'], "unittest/context" ) )

import logging 

logging.basicConfig()
log =  logging.getLogger(__name__)
log.setLevel(logging.INFO)

from context import ctx as ctx
from context import present as present 

present(ctx(globals()))

def setup():
    present(ctx(globals()))
    
def teardown():
    present(ctx(globals()))





