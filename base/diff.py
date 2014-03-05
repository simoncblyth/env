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

    from env.base.diff import Diff, qt 
    dqt = Diff( "meshlab_pristine", "meshlab", ignore=qt() )
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
    patch = False
    ignoreflavor = "g4" 
    patchdir = 'patches'
    name = None


def name_from_args(args):
    assert len(args) == 2 , "expecting 2 directory name arguments"
    a, b = args
    name = None
    log.info("a [%s]%s b [%s]%s " % ( len(a), a, len(b), b, ))
    if len(a) > len(b):
        log.debug("a > b  ")
        if a.startswith(b):
            log.debug("a > b : a startswith b ")
            name = b
    else:
        if b.startswith(a):
            log.debug("a < b : b startswith a ")
            name = a
    return name

def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=defopts.logpath , help="logging path" )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG. Default %default"  )
    op.add_option("-f", "--logformat", default=defopts.logformat , help="logging format" )
    op.add_option("-t", "--ignoreflavor", default=defopts.ignoreflavor , help="Ignore flavor string, eg qt/cmt " )
    op.add_option("-z", "--zippath", default=defopts.zippath , help="Path to zipfile to be created. Default %default ")
    op.add_option("-d", "--patchdir", default=defopts.patchdir , help="Path to patchdir to be created if not existing and populated. Default %default ")
    op.add_option("-p", "--patch",  action="store_true", default=defopts.patch , help="Switch on patch creation. Default %default ")
    op.add_option("-r", "--report", action="store_true", default=defopts.report , help="Full closure report. Default %default ")
    op.add_option("-i", "--name",  default=defopts.name , help="Identifier for the diff . Default %default ")

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

    if opts.name is None:
        opts.name = name_from_args( args )
    assert opts.name is not None, "unable to discern a name for the diff from arguments %s , see manually with -i name" % args

    log.info("diff name %s " % opts.name)
    return opts, args



class Diff(dict):
    """
    For comparing similar directory trees, looking for files
    that differ.
    """ 
    def __init__(self, l,r ):
        dict.__init__(self)
        self.update(diff=[],left=[],right=[])
        top = dircmp(l,r, self.ignore.names )
        self.top = top
        self.recurse(top)

    def collect(self, label, rec):
        if label not in self:
            self[label] = []
        self[label].append(rec)

    def skip(self, name, label):
        igtype = self.ignore_type(name)
        ighead = self.ignore_head(name)
        if igtype:
            log.debug("igtype only %s : %s " % (label, name) )
            return True
        elif ighead:
            log.debug("ighead only %s : %s " % (label, name) )
            return True
        else:
            return False 

    def visit_only( self, label, cf, name):
        if self.skip(name, label):
            log.debug("skip %s " % name)
            return
        only = getattr(cf, label)
        top = getattr(self.top, label)
        _only = only[len(top)+1:]
        if len(_only) > 0:
            path = "%s/%s" % (_only, name) 
        else:
            path = name

        if self.ignore_relpath(path):
            log.debug("ignore_relpath %s " % path )
            return

        apath = "%s/%s" % ( top, path )
        log.debug("only : label %s top %s _only %s path %s " % (label, top, _only, apath) )
        self.collect(label, (apath,))

    def visit_diff(self, cf):
        for name in cf.diff_files:
            if self.skip(name, "diff"):
                continue
            assert cf.left.startswith(self.top.left)
            assert cf.right.startswith(self.top.right)
            _left  = cf.left[len(self.top.left)+1:]
            _right = cf.right[len(self.top.right)+1:]
            assert _left == _right , ( name, _left, _right ) 
            path = "%s/%s" % ( _left, name ) 

            if self.ignore_relpath(path):
                log.debug("ignore_relpath %s " % path )
                continue

            lpath = "%s/%s" % ( cf.left, name )
            rpath = "%s/%s" % ( cf.right, name )
            pass
            self.collect( 'diff', (path, lpath, rpath,) )

    def ignore_type(self, name):
        base, ext = os.path.splitext(name)
        return ext in self.ignore.types

    def ignore_head(self, name):
        for ighead in self.ignore.heads:
            if name.startswith(ighead):return True
        return False

    def ignore_relpath(self, path):
        for igrelpath in self.ignore.relpath:
            if path.startswith(igrelpath):return True
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


class qt(object):
    """
    Specification of files typically generated
    when building qt projects with qmake. Which 
    should be ignored when interested in source only.
    """
    heads=["moc_","ui_","qrc_","Makefile."]
    types=[".o",".so",".dylib",".app"]
    names=["macx","Makefile",".DS_Store"]
    relpath=[]

class cmt(object):
    heads=["Makefile."]
    types=[".o",".so",".dylib",".app",".pyo",".pyc",".d","map",]
    names=["setup.sh","setup.csh","cleanup.sh","cleanup.csh","Makefile",".DS_Store"]
    relpath=[]


class g4(cmt):
    relpath=['include','tmp','lib','bin',]


class QtDiff(Diff):
    ignore = qt()

class CMTDiff(Diff):
    ignore = cmt()

class G4Diff(Diff):
    ignore = g4()





def patch( diff, opts ):
    """
    Create separate patch files for each file with differences
    Usage example::

        [blyth@belle7 g4checkpatch]$ diff.py geant4.9.2.p01.orig geant4.9.2.p01 -p
        [blyth@belle7 g4checkpatch]$ l patches/
        total 92
        -rw-rw-r-- 1 blyth blyth  388 Mar  4 20:15 geant4.9.2.p01_environments_g4py_config_module.gmk.patch
        -rw-rw-r-- 1 blyth blyth  418 Mar  4 20:15 geant4.9.2.p01_environments_g4py_configure.patch
        -rw-rw-r-- 1 blyth blyth  384 Mar  4 20:15 geant4.9.2.p01_source_digits_hits_utils_src_G4ScoreLogColorMap.cc.patch
        -rw-rw-r-- 1 blyth blyth  407 Mar  4 20:15 geant4.9.2.p01_source_digits_hits_utils_src_G4VScoreColorMap.cc.patch
        -rw-rw-r-- 1 blyth blyth 1230 Mar  4 20:15 geant4.9.2.p01_source_geometry_solids_Boolean_src_G4SubtractionSolid.cc.patch
        -rw-rw-r-- 1 blyth blyth  680 Mar  4 20:15 geant4.9.2.p01_source_materials_include_G4MaterialPropertiesTable.hh.patch
        -rw-rw-r-- 1 blyth blyth  888 Mar  4 20:15 geant4.9.2.p01_source_materials_include_G4MaterialPropertyVector.hh.patch
        -rw-rw-r-- 1 blyth blyth  934 Mar  4 20:15 geant4.9.2.p01_source_materials_src_G4MaterialPropertiesTable.cc.patch
        -rw-rw-r-- 1 blyth blyth 4080 Mar  4 20:15 geant4.9.2.p01_source_materials_src_G4MaterialPropertyVector.cc.patch
        -rw-rw-r-- 1 blyth blyth  429 Mar  4 20:15 geant4.9.2.p01_source_persistency_gdml_include_G4GDMLWrite.hh.patch
        -rw-rw-r-- 1 blyth blyth 2346 Mar  4 20:15 geant4.9.2.p01_source_persistency_gdml_src_G4GDMLWrite.cc.patch
        -rw-rw-r-- 1 blyth blyth 1517 Mar  4 20:15 geant4.9.2.p01_source_processes_electromagnetic_lowenergy_src_G4hLowEnergyLoss.cc.patch
        -rw-rw-r-- 1 blyth blyth  393 Mar  4 20:15 geant4.9.2.p01_source_processes_hadronic_processes_include_G4ElectronNuclearProcess.hh.patch
        -rw-rw-r-- 1 blyth blyth  383 Mar  4 20:15 geant4.9.2.p01_source_processes_hadronic_processes_include_G4PhotoNuclearProcess.hh.patch
        -rw-rw-r-- 1 blyth blyth  898 Mar  4 20:15 geant4.9.2.p01_source_processes_hadronic_processes_include_G4PositronNuclearProcess.hh.patch
        -rw-rw-r-- 1 blyth blyth  765 Mar  4 20:15 geant4.9.2.p01_source_processes_hadronic_processes_src_G4ElectronNuclearProcess.cc.patch
        -rw-rw-r-- 1 blyth blyth  737 Mar  4 20:15 geant4.9.2.p01_source_processes_hadronic_processes_src_G4PhotoNuclearProcess.cc.patch
        -rw-rw-r-- 1 blyth blyth 1307 Mar  4 20:15 geant4.9.2.p01_source_processes_optical_include_G4OpBoundaryProcess.hh.patch
        -rw-rw-r-- 1 blyth blyth  406 Mar  4 20:15 geant4.9.2.p01_source_visualization_HepRep_include_cheprep_DeflateOutputStreamBuffer.h.patch
        -rw-rw-r-- 1 blyth blyth  666 Mar  4 20:15 geant4.9.2.p01_source_visualization_VRML_GNUmakefile.patch
        -rw-rw-r-- 1 blyth blyth  663 Mar  4 20:15 geant4.9.2.p01_source_visualization_VRML_include_G4VRML2FileSceneHandler.hh.patch
        -rw-rw-r-- 1 blyth blyth 1634 Mar  4 20:15 geant4.9.2.p01_source_visualization_VRML_src_G4VRML2FileSceneHandler.cc.patch
        -rw-rw-r-- 1 blyth blyth  711 Mar  4 20:15 geant4.9.2.p01_source_visualization_VRML_src_G4VRML2SceneHandlerFunc.icc.patch

    """
    if not os.path.exists( opts.patchdir ):
        log.info("creating patchdir %s " % opts.patchdir )
        os.makedirs(opts.patchdir)

    def identity(_):
        return "%s_%s" % ( opts.name, _[0].replace("/","_") )
 
    for _ in diff['diff']:
        ctx = dict(left=_[1], right=_[2], id=identity(_), patchdir=opts.patchdir )
        patch = "%(patchdir)s/%(id)s.patch " % ctx 
        dff = "diff -u -r %(left)s %(right)s " % ctx 
        cmd = "echo %s > %s && %s >> %s " % ( dff, patch, dff, patch )  
        for line in os.popen(cmd).read():
            print line


def main():
    opts, args = parse_args(__doc__) 

    clsmap = dict(qt=QtDiff, cmt=CMTDiff, g4=G4Diff)
    if opts.ignoreflavor in clsmap:
        cls = clsmap[opts.ignoreflavor]
    else:
        assert 0, "unhandled ignoretype %s " % opts.ignoretype

    diff = cls( *args )
    if opts.report:
        print diff.top.report_full_closure()

    for label, ls in diff.items():
        print label
        for _ in ls:
            print " " * 5, _

    if not opts.zippath is None:
        zippath = opts.zippath
        keys = "diff right".split()
        log.info("writing %s to zippath %s " % ( keys, zippath ))
        zipdiff( diff, zippath, keys )

    if opts.patch:
        patch( diff, opts )
       


if __name__ == '__main__':
    main()


