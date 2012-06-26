#!/usr/bin/env python
from __future__ import with_statement
import os, logging
log = logging.getLogger(__name__)


def bashdoc(path, base, delim):

   if not path.startswith("/"):
       abspath = os.path.join(base,path)	   
   else:
       abspath = path

   with open(abspath,"r") as fp:
       content = fp.read()

   bits = content.split(delim)
   assert len(bits) == 3, "expect 3 bits delimited by %s in %s not %s " % ( delim, path, len(bits)) 

   relp = abspath[len(base)+1:]
   name,type = os.path.splitext(relp)
   assert type == '.bash', "expecting .bash fileext %s  " % path 

   return bits[1], name




def bashtoc( content ):
    """
    Find the toctree and collect the paths::

        [0  ] .. toctree:: 
	[0  ]  
	[1  ]     python/python 
	[2  ]     python/python 
	[-1 ]  
	[-1 ]  

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
	    paths.append(path)
        else:
	    path = ""	

        #print "[%-3s] %s [%s]" % ( count, line, path )
        pass
    return paths



def bashrst(path,base,delim="EOU",gbase=None, kids=False):
   """
   :param path: absolute or relative to base path to bash function file
   :param base: root of the sphinx build, which must contain the `_build` directory 
   :return:  the path to the generated file

   Extract the usage message from the bash function file and writes to a file 
   in `_build/bashrst/`

   Problems:

   #. backticks as needed for rst referencing have special meaning for bash
   
      #. maybe by un-shell-escaping here   
      #. could run the bash function, in order to fill out vars 

   """
   #assert gbase, "gbase must be defined"
   if not gbase:
       log.warn("gbase not defined path %s " % path )	   
       gbase = os.path.join(base, "_build", "bashrst" )
  
   content, name = bashdoc( path, base, delim)

   outp = os.path.join( gbase, name + ".rst")
   odir = os.path.dirname(outp)
   if not os.path.exists(odir):
       os.makedirs(odir)

   with open(outp,"w") as fp:
       fp.write(content)

   if kids:
       paths = bashtoc(content)
       return outp, paths	   
   else:
       return outp   



if __name__ == '__main__':
    path = os.path.expandvars("$ENV_HOME/env.bash")
    base = os.path.expandvars("$ENV_HOME")
    content, name = bashdoc(path, base, "EOU")
    print "content of %s \n %s \n" % (name, content )
    bashtoc(content)

