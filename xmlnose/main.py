#!/usr/bin/env python

import os
import sys

import xmlnose
from nose.plugins.capture import Capture
from nose.core import run

def dump():
    for p in sys.path:
        print p
    for k,v in os.environ.items():
        if k.find("NOSE")>-1:
            print k, v

def main(args):
    """
       Usage: 
           cd $ENV_HOME/unittest/demo
           python $ENV_HOME/xmlnose/main.py --with-xml-output 
           OR _xmlnose
       
        Equivalent to :
           nosetests --with-xml-output 
    
        but allows usage without the need to install the 
        xmlnose plugin into PYTHON_SITE or NOSE_HOME
        
        Note if you do have xmlnose installed already then you will
        get a warning about multiple xmlnose modules in sys.path
    
        NB when running with this approach the full range of nose
           plugins are not availble, just those that are explicitly 
           fed in 
    
    
    """
    result = run(argv=args,plugins=[xmlnose.XmlOutput(),Capture()] )

    
if __name__=='__main__':
    sys.exit(main(sys.argv))

