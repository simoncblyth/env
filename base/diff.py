#!/usr/bin/env python
"""
Compare Project File Trees
============================

Finds files that differ and files that only 
appear in left or right trees. Generated files
can be ignored via ``ignore_`` arguments.

Usage example::

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
log = logging.getLogger(__name__)
       
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
            self.collect_only(label, path)

    def collect_only(self, label, path):
        if label not in self:
            self[label] = []
        self[label].append(path)

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
            self.collect_diff( path, lpath, rpath )

    def collect_diff(self, path, lpath, rpath, label='diff'):
        if label not in self:
            self[label] = []
        rec =  ( path , lpath, rpath, )
        self[label].append( rec )
        log.debug(label + " %s : %s %s  " % rec)

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


def main():
    logging.basicConfig(level=logging.INFO)
    dqt = Diff.qt( *sys.argv[1:] )
    #print dqt.top.report_full_closure()
    for label, ls in dqt.items():
        print label
        for _ in ls:
            print " " * 5, _

if __name__ == '__main__':
    main()


