#!/usr/bin/env python
"""
resources.py
=============

Check size of resources within a folder, with totals
for all subfolders and size sorted list of of all leaves.

::

    resources.py ${APACHE_HTDOCS}/env


#. Easily below 1GB limit when skip mov,m4v

   * rather than create workflow to exclude those, 
     maybe permanently move them elsewhere ? 
     Create a specific dropbox account perhaps.


Bitbucket repo size limits 
---------------------------

* https://confluence.atlassian.com/pages/viewpage.action?pageId=273877699

::

    1GB soft limit
    2GB hard limit

"""
import os, stat, argparse, logging
log = logging.getLogger(__name__)

def parse(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-v","--verbose", action="store_true")
    parser.add_argument("--loglevel",  default="INFO", help="")
    parser.add_argument("-K","--skiptype",  help="comma delimited list of filetypes to skip", default=".mov,.m4v")
    parser.add_argument("rootdir", nargs=1 )
    parser.add_argument("-i","--ipython", action="store_true", help=" "  )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.loglevel.upper()))
    return args 


def crawler(root, directory, dir_, leaf_, exclude_dirs=[".svn"], skipempty=False, skiptype=[]):
    """
    Typically start recursion with `root` and `directory` the same

    :param root: root directory that passes along unchanged through the recursion
    :param directory: absolute directory inside the root, can also be the root
    :param dir_:
    :param exclude_dirs: 

    Recursive crawler.  
    """
    assert directory.startswith(root)
    rrdir = directory[len(root):]  

    leaves = {}
    for name in os.listdir(directory):
        path = os.path.join(directory,name)

        if os.path.isdir(path):
            if not name in exclude_dirs:
                subleaves = crawler( root, path, dir_, leaf_, exclude_dirs, skipempty, skiptype )
                leaves.update(subleaves)
        else:
            base, ext = os.path.splitext(name)
            if ext in skiptype:
                log.info("skipping %s " % path ) 
            else:
                leaves[path[len(root):]] = leaf_( path )
        pass
    pass
    if len(leaves) == 0 and skipempty:
        log.debug("skipempty rrdir %s " % rrdir )
    else:   
        dir_(rrdir, leaves)
    pass
    return leaves

def resources(rootdir, skiptype):

    def psize( sz ):
        sz = float(sz)/(1024*1024) 
        return "%.2fM" % sz

    def dir_( path, leaves ):
        size = sum( map( lambda k:leaves[k]['size'], leaves ))
        print "dir_ %s %s  " % ( psize(size), path )

    def leaf_( path ):
        st = os.stat(path)
        return {'size':st[stat.ST_SIZE]}

    leaves = crawler(rootdir, rootdir, dir_, leaf_ , skiptype=skiptype )
    print "\n".join([" %15s : %s " % (psize(leaves[k]['size']), k) for k in sorted(leaves,key=lambda k:leaves[k]['size'])])  

def main():
    args = parse(__doc__) 
    rootdir = args.rootdir[0] + '/'
    skiptype = args.skiptype.split(",")
    resources(rootdir, skiptype)


if __name__ == '__main__':
    main()





