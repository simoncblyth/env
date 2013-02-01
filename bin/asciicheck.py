#!/usr/bin/env python
"""
Prints lines and linenumbers containing non-ascii characters, usage::

    asciicheck.py b2charm_end_of_2011_v005_tables.tex
    rc=$?  ## rc is count of non-ascii lines

Optionally a 2nd argument is matched against lines to provide 
context to non-ascii containing lines, for example if the document 
contains http references followed by content use::

    asciicheck.py b2charm_end_of_2011_v005_tables.tex http://

TODO:

#. more flexible context matching via regexp	

"""
from __future__ import with_statement

nascii_ = lambda _:not(0 <= ord(_) < 128) 

def highlight(line):
    out = ""
    for c in line:
        if nascii_(c):
            out += "[" + c + "]"
        else:
            out += c
    return out

def asciicheck(path, ctxmatch=None):
    """
    :param path: path to file to check for non-ascii characters
    :param ctxmatch: beginning of line to output for providing context to ascii violating lines
    :return:  count of non-ascii char containing lines in the file
    """  
    naline = 0 
    ctx = None
    with open(path,"r") as fp:
        for i, line in enumerate(fp.readlines()):
            nascii = filter(nascii_, line )
            if ctxmatch and line.startswith(ctxmatch):
                ctx = "%s %s" % ( i+1, line )
            if len(nascii)>0:
                if ctx:
                    print ctx,
                ctx = None
                print path, i+1, highlight(line)
                naline += 1
    return naline

if __name__ == '__main__':
    import sys
    sys.exit(asciicheck(*sys.argv[1:]))


