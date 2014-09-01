#!/usr/bin/env python
"""
digest.py
==========

Emit hexdigest of resources within a folder

::

    cd /tmp/t/env
    digest.py .hg hgrc


OSX/Linux file naming difference is scrambling the digest order::

    [blyth@cms01 env]$ l .hg/store/data/base/.ssh.bash.swp.i 
    -rw-rw-r--  1 blyth blyth 1560 Aug 29 17:53 .hg/store/data/base/.ssh.bash.swp.i

    (adm_env)delta:env blyth$ l .hg/store/data/base/~2essh.bash.swp.i 
    -rw-r--r--  3 blyth  staff  1560 Aug 29 16:46 .hg/store/data/base/~2essh.bash.swp.i


"""
# dont use argparse/optparse as want to stay ancient python compatible 
from __future__ import with_statement
import os, sys, time, stat
try: 
    from hashlib import md5
except ImportError: 
    from md5 import md5

import logging
log = logging.getLogger(__name__)

cumulative = None

def digest_(path):
    """
    :param path:
    :return: md5 hexdigest of the content of the path or None if non-existing path

    http://stackoverflow.com/questions/1131220/get-md5-hash-of-a-files-without-open-it-in-python

    Confirmed to give same hexdigest as commandline /sbin/md5::

        md5 /Users/blyth/workflow/notes/php/property/colliers-4q2011.pdf 
        MD5 (/Users/blyth/workflow/notes/php/property/colliers-4q2011.pdf) = 3a63b5232ff7cb6fa6a7c241050ceeed

    """
    global cumulative

    if not os.path.exists(path):return None
    if os.path.isdir(path):return None
    dig = md5()

    if cumulative is None:
        cumulative = md5() 

    with open(path,'rb') as f: 
        for chunk in iter(lambda: f.read(8192),''): 
            dig.update(chunk)
            cumulative.update(chunk)
        pass
    return dig.hexdigest()


def linux_name( name ):
    """
    Fix an OSX/Linux file naming difference, that would otherwise scramble  
    ordering, messing up the cumulative digest.

    * http://www.readynas.com/forum/viewtopic.php?f=28&t=28265

    """
    if name[0:3] == "~2e":
        name = name.replace("~2e",".")
    return name


def crawler(root, directory, dir_, leaf_, opts):
    """
    Recursive crawler.  

    Typically start recursion with `root` and `directory` the same

    :param root: root directory that passes along unchanged through the recursion
    :param directory: absolute directory inside the root, can also be the root
    :param dir_:
    :param leaf_: 

    """
    assert directory.startswith(root)
    rrdir = directory[len(root):]  

    leaves = {}

    # name shenanigans is workaround for an OSX/Linux file naming bug, from long ago ? 
    names = os.listdir(directory)
    lnames = map(linux_name,names)   # linux names
    namemap = dict(zip(lnames,names))  # keyed by the linux name 

    for lname in sorted(namemap):    # in linux name order
        name = namemap[lname]        # original name
        path = os.path.join(directory,name)

        if os.path.isdir(path):
            if not name in opts.exclude_dirs:
                subleaves = crawler( root, path, dir_, leaf_, opts )
                leaves.update(subleaves)
        else:
            relpath = path[len(root):]
            if opts.only:
                if relpath in opts.rels:
                    log.info("only relpath %s %s " % (relpath,repr(opts.rels)) ) 
                    leaves[relpath] = leaf_( path )
                else:
                    log.info("only skip relpath %s " % relpath ) 
            else:
                if relpath in opts.rels:
                    log.debug("skipping relpath %s " % relpath ) 
                else:
                    leaves[relpath] = leaf_( path )
                pass
        pass
    pass
    if len(leaves) == 0 and opts.skipempty:
        log.debug("skipempty rrdir %s " % rrdir )
    else:   
        dir_(rrdir, leaves)
    pass
    return leaves

def digest(rootdir, opts):

    def psize( sz ):
        sz = float(sz)/(1024*1024) 
        return "%.2fM" % sz

    def dir_( path, leaves ):
        pass
        #size = sum( map( lambda k:leaves[k]['size'], leaves ))
        #print "dir_ %s %s  " % ( psize(size), path )

    def leaf_( path ):
        st = os.stat(path)
        return {'size':st[stat.ST_SIZE],'digest':digest_(path)}

    leaves = crawler(rootdir, rootdir, dir_, leaf_, opts)
    if opts.verbose:
        print "\n".join(["%s : %s " % (leaves[k]['digest'], k) for k in sorted(leaves,key=lambda k:k)])  
    pass

    if cumulative is None:
        print "oops no digests"
    else:
        print cumulative.hexdigest()



def args_(doc):
    from optparse import OptionParser
    parser = OptionParser(usage=doc)
    parser.add_option("-l", "--loglevel", default="INFO", help="logging level")
    parser.add_option("-v", "--verbose", action="store_true", help="Dump the component digests.")
    parser.add_option("-o", "--only", action="store_true", help="Instead of skipping relative paths provides in args, switch to only giving digests for them.")
    opts, args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,opts.loglevel.upper()))
    return opts, args

def main():

    opts, args = args_(__doc__) 
    rootdir = args[0] + '/'

    opts.rels = args[1:]
    opts.exclude_dirs = []
    opts.skipempty = True

    digest(rootdir, opts)


if __name__ == '__main__':
    main()





