#!/usr/bin/env python
"""
Compare Project File Trees
============================

Finds files that differ and files that only 
appear in left or right trees. Generated files
can be ignored via ``ignore_`` arguments.

Commandline example::

   diff.py MeshLabSrc_AllInc_v132 meshlab -z mlab.zip

This recursively look for differences between
directory trees and adds new files on the **right** 
(ie from ``meshlab``) as well as common files that 
differ into the zipfile.

This allows quick archiving of ad-hoc developments 
in an unmanaged(not in SCM) tree. 

NB the directory argument order determines which new files get 
added to the zip. The default files to write to the zip:

#. files in common between left and right that differ
#. new files on the right

Due to this remember to distclean on the right to avoid zipping
generated files.

Usage from python example::

    from env.base.diff import Diff
    dqt = Diff.qt( "meshlab_pristine", "meshlab" )
    for label, ls in dqt.items():
        print label
        for _ in ls: 
            print " " * 5, _

Implementation based on filecmp.dircmp


"""
import os, sys, logging
from filecmp import dircmp
from zipfile import ZipFile
log = logging.getLogger(__name__)


class Defaults(object):
    logpath = None
    loglevel = "INFO"
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    zippath = None  
    report = False

def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=defopts.logpath , help="logging path" )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG. Default %default"  )
    op.add_option("-f", "--logformat", default=defopts.logformat , help="logging format" )
    op.add_option("-z", "--zippath", default=defopts.zippath , help="Path to zipfile to be created. Default %default ")
    op.add_option("-r", "--report", action="store_true", default=defopts.report , help="Full closure report. Default %default ")

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
        logging.basicConfig(format=opts.logformat,level=level)
    pass
    log.info(" ".join(sys.argv))
    if not opts.zippath is None:
        opts.zippath = os.path.expandvars(os.path.expanduser(opts.zippath))
    return opts, args


class Diff(dict):
    """
    For comparing similar directory trees, looking for files
    that differ.
    """ 
    @classmethod
    def qt(cls, l, r ):
        """
        Compare project directories, ignoring files typically generated
        when building qt projects with qmake.
        """
        dqt = cls(l, r,
                    ignore_file_heads=["moc_","ui_","qrc_","Makefile."], 
                    ignore_file_types=[".o",".so",".dylib",".app"], 
                    ignore_file_names=["macx","Makefile",".DS_Store"])
        return dqt

    def __init__(self, l,r, **kwa):
        dict.__init__(self, **kwa )
        self.update(diff=[],left=[],right=[])
        top = dircmp(l,r, self['ignore_file_names'])
        self.top = top
        self.recurse(top)

    def collect(self, label, rec):
        if label not in self:
            self[label] = []
        self[label].append(rec)

    def visit_only( self, label, cf, name):
        igtype = self.ignore_type(name)
        ighead = self.ignore_head(name)
        if igtype:
            log.debug("igtype only %s : %s " % (label, name) )
        elif ighead:
            log.debug("ighead only %s : %s " % (label, name) )
        else:
            base = getattr(cf, label)
            path = "%s/%s" % ( base, name )
            log.debug("only %s : %s " % (label, path) )
            self.collect(label, (path,))

    def visit_diff(self, cf):
        for name in cf.diff_files:
            assert cf.left.startswith(self.top.left)
            assert cf.right.startswith(self.top.right)
            _left  = cf.left[len(self.top.left)+1:]
            _right = cf.right[len(self.top.right)+1:]
            assert _left == _right , ( name, _left, _right ) 
            path = "%s/%s" % ( _left, name ) 
            lpath = "%s/%s" % ( cf.left, name )
            rpath = "%s/%s" % ( cf.right, name )
            pass
            self.collect( 'diff', (path, lpath, rpath,) )

    def ignore_type(self, name):
        base, ext = os.path.splitext(name)
        return ext in self['ignore_file_types']

    def ignore_head(self, name):
        for ighead in self['ignore_file_heads']:
            if name.startswith(ighead):return True
        return False

    def visit(self, cf):
        self.visit_diff(cf)
        for name in cf.left_only:
            self.visit_only("left", cf, name)
        for name in cf.right_only:
            self.visit_only("right", cf, name)

    def recurse(self, cf):
        self.visit(cf)
        for sf in cf.subdirs.values():    
            self.recurse(sf)



def zipdiff( diff, zippath, keys ):
    """
    # CAUTION : SINGLE LEVEL ASSUMPTION FOR NEW DIR, 
    # TODO: RECURSIVE ADDING 
    """
    zf = ZipFile(zippath,"w")
    for k in keys:
        for _ in diff[k]:
            path = _[-1]
            if os.path.isdir(path):
                for name in os.listdir(path):
                    print name
                    subpath = os.path.join(path, name)
                    print subpath
                    zf.write(subpath) 
            else:
                zf.write(path) 
        pass
    pass    
    zf.close()



def main():
    opts, args = parse_args(__doc__) 

    dqt = Diff.qt( *args )
    if opts.report:
        print dqt.top.report_full_closure()

    for label, ls in dqt.items():
        print label
        for _ in ls:
            print " " * 5, _

    if not opts.zippath is None:
        zippath = opts.zippath
        keys = "diff right".split()
        log.info("writing %s to zippath %s " % ( keys, zippath ))
        zipdiff( dqt, zippath, keys )


if __name__ == '__main__':
    main()


