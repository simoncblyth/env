#!/usr/bin/env python
"""
http://docs.oracle.com/cd/E17276_01/html/api_reference/CXX/frame.html
"""
import logging
log = logging.getLogger(__name__)

from bsddb3.db import *
from dbxml import *

# must import after dbxml

import pyextfun as pef

if __name__ == '__main__':
    print dir(pef.MyExternalFunctionPow)
    print dir(pef.MyExternalFunctionSqrt)

    p = pef.MyExternalFunctionPow()
    print p


