#!/usr/bin/env python

import os
import sys

import xmlplug
import dybplug

from nose.plugins.capture import Capture

from nose.core import run

def dump():
    for p in sys.path:
        print p
    for k,v in os.environ.items():
        if k.find("NOSE")>-1:
            print k, v

def main(args):
    """Provides a standard way of running nose tests to facilitate automated invokation and comparisons.
       ... actually eventually better to do away with this ... if can do all customisations from the DybWrap plugin 
    
    """
    result = run(argv=args,plugins=[xmlplug.XmlOutput(),dybplug.DybWrap(),Capture()] )

    
if __name__=='__main__':
    sys.exit(main(sys.argv))

