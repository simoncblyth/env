G4PY Examples
================

Try using g4py from Chroma Geant4.

::

    (chroma_env)delta:event blyth$ ll $VIRTUAL_ENV/src/geant4.9.5.p01/environments/g4py/
    total 80
    -rwxr-xr-x   1 blyth  staff  15855 Mar 27  2012 configure
    -rw-r--r--   1 blyth  staff   6817 Mar 27  2012 History
    -rw-r--r--   1 blyth  staff    596 Mar 27  2012 GNUmakefile
    -rw-r--r--   1 blyth  staff    152 Mar 27  2012 AUTHORS
    -rw-r--r--   1 blyth  staff   4668 Mar 27  2012 00README
    drwxr-xr-x   5 blyth  staff    170 Mar 27  2012 ..
    drwxr-xr-x  13 blyth  staff    442 Mar 27  2012 .
    drwxr-xr-x   6 blyth  staff    204 Mar 27  2012 tools
    drwxr-xr-x  26 blyth  staff    884 Mar 27  2012 tests
    drwxr-xr-x  24 blyth  staff    816 Mar 27  2012 source
    drwxr-xr-x  11 blyth  staff    374 Mar 27  2012 site-modules
    drwxr-xr-x   9 blyth  staff    306 Mar 27  2012 examples
    drwxr-xr-x  12 blyth  staff    408 Mar 27  2012 config


emplot
-------

Surprisingly it works, run from ipython to prevent the plot 
from immediately disappearing as the process exits.

::
   
    chroma- 
    cd /usr/local/env/chroma_env/src/geant4.9.5.p01/environments/g4py/examples/emplot

    (chroma_env)delta:emplot blyth$ ipython
    Python 2.7.6 (default, Nov 18 2013, 15:12:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: run eplot.py

    *************************************************************
     Geant4 version Name: geant4-09-05-patch-01    (20-March-2012)
                          Copyright : Geant4 Collaboration
                          Reference : NIM A 506 (2003), 250-303
                                WWW : http://cern.ch/geant4
    *************************************************************





