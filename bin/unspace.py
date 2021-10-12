#!/usr/bin/env python
"""
unspace.py 
==========

Emit to stdout commands to rename all files in pwd with spaces
in their names to names with the spaces removed. 

Pipe the output to the shell to do this renaming.

"""
import glob 

if __name__ == '__main__':

    exts = "*" 
    for ext in exts.split():
        for name in glob.glob("*.%s"%ext):
            if " " in name:
                newname = name.replace(" ","") 
                cmd = "mv \"%s\" \"%s\" " % (name, newname)
                print(cmd) 
            pass


