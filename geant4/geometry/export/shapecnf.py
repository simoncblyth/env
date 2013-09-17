#!/usr/bin/env python

import logging, sys, os
log = logging.getLogger(__name__)

class Defaults(object):
    mode = "gen" 
    query = "select id from shape where id in (1,2,3)"
    scale = None
    around = None
    like = None
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    loglevel = "INFO"
    logpath = None
    maxids = 1000
    dbpath = 'g4_01.db'
    chunksize = 100
    nameshape = False   # no longer needed as all volumes and materials now named in vrml2file.py 
    group = None
    distance = '12000'

def parse_args(doc):
    """
    """
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=defopts.logpath )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-f", "--logformat", default=defopts.logformat )
    op.add_option("-m", "--mode", default=defopts.mode, help="Mode of output: ori, dup, gen " )
    op.add_option("-q", "--query", default=defopts.query, help="An SQL query that returns shape id integers to print." )
    op.add_option("-c", "--center", action="store_true", help="Before any scaling subtract the average coordinates of a shape/shapeset from all points therein, in order to center the shapeset." )
    op.add_option("-s", "--scale", default=defopts.scale, help="After translations have been done, scale all point coordinates by this factor. Default %default" )
    op.add_option("-e", "--distance", default=defopts.distance, help="Viewpoint distance. Default %default" )
    op.add_option("-a", "--around", default=defopts.around, help="Four floats delimited by commas x,y,z,d . Volumes with centroids within the box are included in output. Default %default. Using this replaces manual SQL query option. " )  
    op.add_option("-k", "--like",  default=defopts.like, help="Comma delimited SQLite like string for volume selection by name. Default %default. Using this replaces manual SQL query option. " )  
    op.add_option("-n", "--dryrun", action="store_true", help="Just do volume identification, not WRL generation. For debugging `around` OR `query` options without time consuming steps.")
    op.add_option("-x", "--maxids", default=defopts.maxids, help="Maximum number of shapes to allow dumping. Default %default.")
    op.add_option("-z", "--chunksize", type="int", default=defopts.chunksize, help="Maximum number of shapes to group by query at once. Default %default.")
    op.add_option("-g", "--group", default=defopts.group, help="Name of node in which all others are placed, or None for no such group. A common choice of name is \"root\". Default %default.")
    op.add_option(      "--nonameshape", action="store_false", dest="nameshape", default=defopts.nameshape, help="Name the shapes in VRML eg S1,S2 etc... Default %default.")
    op.add_option("-d", "--dbpath", default=defopts.dbpath, help="Path to shape DB file. Either an absolute path beginning with a '/' or a source directory relative path not beginning with '/'. Default %default.")
    op.add_option("-T", "--TEST", action="store_true", help="Duplication testing  multiple randomly chosen shapes." )
    op.add_option("-D", "--DUMP", action="store_true", help="Debug dumping." )
    opts, args = op.parse_args()
    level = getattr( logging, opts.loglevel.upper() )

    if opts.logpath:  # logs to file as well as console, needs py2.4 + (?)
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(opts.logformat)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)  # add the handler to the root logger
    else:
        try: 
            logging.basicConfig(format=opts.logformat,level=level)
        except TypeError:
            hdlr = logging.StreamHandler()              # py2.3 has unusable basicConfig that takes no arguments
            formatter = logging.Formatter(opts.logformat)
            hdlr.setFormatter(formatter)
            log.addHandler(hdlr)
            log.setLevel(level)
        pass
    pass
    log.info(" ".join(sys.argv))
    if not opts.dbpath[0] == '/':
        opts.dbpath = os.path.join(os.path.dirname(__file__),opts.dbpath)
    pass    
    return opts, args


