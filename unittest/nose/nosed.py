#!/usr/bin/env python

import sys
import os
from optparse import OptionParser

import xmlplug
from nose.core import run


def dump():
    for p in sys.path:
        print p
    for k,v in os.environ.items():
        if k.find("NOSE")>-1:
            print k, v


def main(args):
    """Provides a standard way of running nose tests to facilitate automated invokation and comparisons.
    """
    
    op = OptionParser(usage="usage: %prog [options] [foo.py] [bar.py] [/absolute/path] [relative/path] ", version="%prog 0.1"  , description=main.__doc__ )
    
    choices=["test","summarize"]
    op.add_option("--action"  ,   default="test" , choices=choices , help="choose one of: %s   default:[%%default] " % ", ".join(choices) )
   
    
    # 
    #op.add_option("--searchdir" , default=os.environ['PWD'], type="string" , help="starting directory from which to look for tests, default:[%default] ")
    
    op.add_option("--tofile"    , default=False                           , help="write outputs to file rather than stdout, default:[%default] " ) 
    
    op.add_option("--reportdir" , default="tests/nose"     , type="string" , help="absolute or relative to searchdir path to store output, default:[%default]  ")   
   
    op.add_option("--dryrun",     action="store_true" ,  help="discover and report tests found but do not run them, default:[%default] ")    
    op.set_defaults( dryrun=False ) 
    
    op.add_option("--xml"   ,   action="store_true"  ,    help="generate test results in xml, default:[%default] "  )
    op.set_defaults( xml=False )
    
    op.add_option("--html"  ,   action="store_true"   ,   help="generate test results in html, xml creation is forced when this option is chosen, default:[%default]" )
    op.set_defaults( html=False )
    
    op.add_option("--auto"  ,  action="store_true"   ,     help="collective option that implies --tofile, --html, --xml, --quiet , default:[%default]")
    op.set_defaults( auto=False )

    # would be nicer to avoid this script knowing anything about Dyb 
    #op.add_option("--siteroot" , default=os.environ['SITEROOT'] )
       
    #op.add_option("-v","--verbose",  action="store_true",  dest="verbose")
    #op.add_option("-q","--quiet"  ,  action="store_false", dest="verbose")    
    # the standard is fine
    
    op.set_defaults( verbose=True )     
         
           
    (cfg , prgs) = op.parse_args(args[1:])
    cfg.base = os.path.dirname(args[0])
    
    print cfg
    
    argv=[__file__]
    argv.extend( prgs )
    plugins=[]
    
    if cfg.html or cfg.xml:
        argv.append( "--with-xml-output" )
        plugins=[xmlplug.XmlOutput()]
    
    result = run(argv=argv,plugins=plugins )


    
if __name__=='__main__':
    sys.exit(main(sys.argv))

