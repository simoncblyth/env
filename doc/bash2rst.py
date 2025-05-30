#!/usr/bin/env python
"""
"""
#from __future__ import with_statement
import os, logging
log = logging.getLogger(__name__)

def bashtoc( content ):
    """
    :param content:  the documentation content
    :return: list of relative paths extracted from single toctree

    Find the toctree and collect the paths::

        [0  ] .. toctree:: 
        [0  ]  
        [1  ]     python/python 
        [2  ]     python/python 
        [-1 ]  

    TODO:

    #. generalize to support the titled form of toctree referencing

    """
    count = -1
    paths = []
    for line in content.split("\n"):
        if line.startswith(".. toctree::"):
            count = 0
        elif len(line) == 0:   # blank line
            if count == 0:     # immediately following the toctree
                pass    
            else:              # terminating blank 
                count = -1         
        else:
            if count > -1:
                count += 1
            pass
        if count > 0:
            path = line.strip().rstrip()
            if path.endswith('/'):
                print("skip %s " % path)
            elif len(path) == 0:
                pass     
            else:
                paths.append(path)
        else:
            path = ""
        #print "[%-3s] %s [%s]" % ( count, line, path )
        pass
    return paths


class Bash(list):
    """
    Problems:

    #. backticks as needed for rst referencing have special meaning for bash
   
       #. maybe by un-shell-escaping here   
       #. could run the bash function, in order to fill out vars 
       #. just live with it, almost never actually use fn-usage anyhow typically use fn-vi

    The docs are organized such that a repeated name implicitly indicates an "index" eg "tools/tools.bash" 
    references "tools/sleepwatcher.bash" and plays the role of index.
    This is not acted upon, perhaps it could and result in generation of "tools/index" ? Which would
    avoid the extra level in the resultant URLs

    """
    def __init__(self, path ):
        """
        :param path: to bash function file
        """
        log.debug("path %s " % path )

        apath = os.path.abspath(path)
        rdir = os.path.dirname(apath)
        root = os.getcwd()
        rpath = apath[len(root)+1:]
        rname, type = os.path.splitext( rpath )
        assert type == ".bash", (type, path )

        if rname.endswith('/'):
            print("[%s]" % rname)

        content = self._rst_read(apath)
        paths = bashtoc(content)

        log.info("path %s => %s " % (path, repr(paths) ))

        self.root = root
        gpath = self._genpath( rname )

        self.path = path
        self.rdir = rdir
        self.apath = apath
        self.rpath = rpath         # relative to root, ie cwd
        self.gpath = gpath
        self.content = content
        self.extend( paths ) 

    def _genpath(self, rname ):
        """
        :param rname: root relative name
        :return: generated absolute rst path, with index swap-ins
        """
        iname = self.place_index(rname)
        return os.path.join( self.root, "_docs" , iname + ".rst" )

    def remove_index(self, rname):
        """
        Replace a toctree reference to an index like "mobkp/index" with 
        the bash function argot of "mobkp/mobkp"

        Actually need to write the index
        """
        parts = rname.split("/")
        if len(parts)>1:
            if parts[-1] == "index":
                parts[-1] = parts[-2]    
        return "/".join(parts)

    def place_index(self, rname):
        """
        Looking for names like::

        green/red/red

        where the last 2 leaves are the same, implying index behavior return green/red/index
        """
        parts = rname.split("/")
        if len(parts) > 1: 
            if parts[-2] == parts[-1]:
                parts[-1] = "index"    
        return "/".join(parts)


    def _rst_read(self, path, delim="EOU"):
        """
        Extract rst documentation from the bash function file

        :return: extracted content as a single string 
        """
        if not os.path.exists(path):
            log.warn("path %s does not exist " % path )
            return ""

        fp = open(path,"r") 
        content = fp.read()
        fp.close()

        bits = content.split(delim)
        assert len(bits) == 3, "expect 3 bits delimited by %s in %s not %s " % ( delim, path, len(bits)) 
        return bits[1]

    def write_rst(self):
        """
        Write the content to the supplied path 
        """
        gpath = self.gpath
        gdir = os.path.dirname(gpath)
        if not os.path.exists(gdir):
            os.makedirs(gdir)
            log.info("writing %s " % gpath )    
        fp = open(gpath,"w") 
        fp.write(self.content)
        fp.close()
        return gpath

    def Walk(cls, path):
        """
        Recursive conversion of a tree of bash function usage strings
        into a tree of Sphinx rst documentation files. Where the linkage
        is defined by toctree directives within the usage content.
        NB only files linked via toctree are walked

        :param path: to bash file eg ``env.bash``
        """
        parent = cls(path)
        parent.write_rst()
        log.debug("paths %s " % (repr(parent)) )
        for child in parent:
            a = os.path.join( parent.rdir, child )	
            log.debug("child %s a %s " % (child, a) )

            parts = a.split("/")
            if len(parts) > 1:
                if parts[-1] == "index":
                    parts[-1] = parts[-2]
            aa = "/".join(parts)

            cls.Walk( aa + ".bash" )
    Walk = classmethod(Walk)



def main(root):
    Bash.Walk(root)

if __name__ == '__main__':
    main("env.bash")

