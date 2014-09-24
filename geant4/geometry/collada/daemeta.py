#!/usr/bin/env python
"""
DAEMETA
=========

Presents metadata from G4DAE reconstructed Geant4 volumes contolled via a format string. 
Usage example::

    daemeta.py -p g4_00.dae -f "%(p_index)s %(n_polysmry)s" > g4_00.dae.txt

The first character of the format key, indicates the source of the metadata

**p** for nodeprop 
     p_index,p_id"
**n** for node metadata
     n_polysmry,n_copyNo
**g** for geometry matadata
     g_ErrorBooleanProcess,g_NumberOfRotationSteps,g_cerr,g_cout

TODO:

#. selection expressions based on metadata values, eg dump call with non blank g_cerr 

"""
import os, sys, logging, re
log = logging.getLogger(__name__)
from env.geant4.geometry.collada.g4daenode import DAENode


class Defaults(object):
    logpath = None
    loglevel = "INFO"
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    daepath = "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae"
    summary = False
    format = "%(p_index)s %(n_polysmry)s %(g_NumberOfRotationSteps)s"

def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=defopts.logpath , help="logging path" )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG. Default %default"  )
    op.add_option(      "--logformat", default=defopts.logformat , help="logging format" )
    op.add_option("-p", "--daepath", default=defopts.daepath , help="Path to the original geometry file. Default %default ")
    op.add_option("-y", "--summary", action="store_true", default=defopts.summary , help="Path to the original geometry file. Default %default ")
    op.add_option("-f", "--format",  default=defopts.format , help="Output format string. Default %default ")

    opts, args = op.parse_args()
    del sys.argv[1:]   # avoid confusing webpy with the arguments
    level = getattr( logging, opts.loglevel.upper() )
    if opts.logpath:  # logs to file as well as console, needs py2.4 + (?)
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(opts.logformat)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)  # add the handler to the root logger
    else:
        logging.basicConfig(format=opts.logformat,level=level)
    pass
    log.info(" ".join(sys.argv))
    opts.daepath = os.path.expandvars(os.path.expanduser(opts.daepath))

    return opts, args


def main():
    opts, args = parse_args(__doc__) 
    DAENode.parse( opts.daepath )
    if opts.summary:
        DAENode.summary()
    pass     
    keys = DAENode.format_keys( opts.format )
    for node in DAENode.registry:
        print node.format(opts.format, keys)
    pass    

if __name__ == '__main__':
    main()
 



