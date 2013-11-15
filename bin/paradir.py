#!/usr/bin/env python
"""
``paradir.py`` parallel data directory handling
=================================================

Clears working copy of data files with particular file extensions 
by moving them to a parallel tree.

Usage::

   cd ~/env/geant4/geometry/gdml
   paradir.py
   paradir.py | sh 

Subseqently sync the tree between machines with something like:

   rsync -razvt  N:$(local-base N)/env/geant4/geometry/gdml/ $(local-base)/env/geant4/geometry/gdml/

"""
import os, logging
log = logging.getLogger(__name__)

def find_paradir( home, para ):
    """
    :param home: root of source tree
    :param para: corresponding root of parallel data tree
    :param exts: list of file extensions to transfer from home to para 

    #. invoking directory must be inside ``home``
    #. finds sub-directories of invoking directory that 
       have a parallel counterpart directory
    """
    for name in os.listdir("."):
        if os.path.isdir(name) and name[0] != ".":
            path = os.path.abspath(name)
            assert path[0:len(home)] == home, (path, home)
            rdir = path[len(home)+1:]
            paradir = os.path.join(para, rdir)
            if os.path.exists(paradir):
                yield name, rdir, path, paradir
            else:
                log.info("not existing: %s " % paradir ) 
                cmd = "mkdir -p %(paradir)s " % locals()
                print cmd
        

def move_files( from_ , to_ , exts ):
    """
    Prepare commands to move files with matched extension
    """
    cmds = []
    for name in os.listdir(from_):
        base, ext = os.path.splitext(name)
        if ext in exts:
            cmd = "mv %(from_)s/%(name)s %(to_)s/%(name)s " % locals()
            cmds.append(cmd)
    return "\n".join(cmds)

         
def main():
    logging.basicConfig(level=logging.INFO)
    home = os.path.expandvars('$ENV_HOME')
    para = os.path.expandvars('$LOCAL_BASE/env')
    exts = ".db .wrl .dae .gdml .png".split()
    for name, rdir, path, paradir in find_paradir(home, para):
        print move_files( path, paradir, exts)

         
if __name__ == '__main__':
    main()







