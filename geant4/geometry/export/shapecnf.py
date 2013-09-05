#!/usr/bin/env python

import logging, sys
log = logging.getLogger(__name__)

def parse_args(doc):
    """
    """
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=None )
    op.add_option("-l", "--loglevel",   default="INFO", help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-f", "--logformat", default="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    op.add_option("-m", "--mode", default="gen", help="Mode of output: ori, dup, gen " )
    op.add_option("-q", "--query", default="select id from shape where id in (1,2,3)", help="An SQL query that returns shape id integers to print." )
    op.add_option("-c", "--center", action="store_true", help="Before any scaling subtract the average coordinates of a shape/shapeset from all points therein, in order to center the shapeset." )
    op.add_option("-s", "--scale", default=None, help="After translations have been done, scale all point coordinates by this factor. Default %default" )
    op.add_option("-a", "--around", default=None, help="Four floats delimited by commas x,y,z,d . Volumes with centroids within the box are included in output. Default %default. Using this replaces manual SQL query option. " )  
    op.add_option("-n", "--dryrun", action="store_true", help="Just do volume identification, not WRL generation. For debugging `around` OR `query` options without time consuming steps.")
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
    #logging.getLogger().setLevel(loglevel)
    return opts, args


