#!/usr/bin/env python
"""
http://docs.oracle.com/cd/E17276_01/html/api_reference/CXX/frame.html
"""
import logging
log = logging.getLogger(__name__)

from bsddb3.db import *
from dbxml import *

from pyextfun import MyExternalFunctionPow as Pow
from pyextfun import MyExternalFunctionSqrt as Sqrt

if __name__ == '__main__':
    print dir(Pow)
    print dir(Sqrt)

    p = Pow()
    print p


